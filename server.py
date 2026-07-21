import asyncio
import hashlib
import json
import logging
import os
import traceback
import uuid
from datetime import datetime

from aiohttp import web
from aiortc import (
    RTCConfiguration,
    RTCDataChannel,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.contrib.media import MediaBlackhole
from av.audio.resampler import AudioResampler

from audio_player.audio_player import AudioPlayer
from auth_store import (
    GUEST_CONVERSATION_LIMIT,
    GUEST_TOKEN_TTL_SECONDS,
    add_message,
    authenticate_user,
    backup_to_supabase,
    create_conversation,
    create_guest,
    create_session,
    create_user,
    decrement_conversation_count,
    decrement_guest_conversation_count,
    end_conversation,
    get_conversation_usage,
    get_guest_conversation_count,
    get_session_user,
    increment_conversation_count,
    increment_guest_conversation_count,
    init_auth_db,
)
from config.constants import *
from config.logging_config import setup_logging
from utils import *


class TranscriptLoggingQueue(asyncio.Queue):
    """Wraps the ASR->LLM queue to persist each final transcript as a user message."""

    def __init__(self, conversation_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conversation_id = conversation_id

    async def put(self, item):
        if item and self._conversation_id is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, add_message, self._conversation_id, "user", item)
        await super().put(item)


class ResponseLoggingQueue(asyncio.Queue):
    """
    Wraps the LLM->TTS queue to persist each streamed response as one assistant message.

    LLM output arrives as chunks with a trailing None marking the end of a turn, so chunks
    are buffered and saved as a single message once the None terminator is seen.
    """

    def __init__(self, conversation_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conversation_id = conversation_id
        self._buffer = ""

    async def put(self, item):
        if item is None:
            if self._buffer:
                if self._conversation_id is not None:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        None, add_message, self._conversation_id, "assistant", self._buffer
                    )
                self._buffer = ""
        else:
            self._buffer += item
        await super().put(item)


class WebPage:
    """
    WebPage class to handle WebRTC connections and media streaming.
    """
    def __init__(self):
        """
        Initialize the WebPage class.
        """
        self.logger = logging.getLogger(__name__)
        self.pcs = set()
        self.args = self.generate_args()
        patch_ice_gather_timeout()
        init_auth_db()
        backup_to_supabase()
        self.logger.info("Logging is set up.")

    def _set_session_cookie(self, response: web.StreamResponse, request: web.Request, token: str):
        response.set_cookie(
            "session_token",
            token,
            max_age=3600,
            httponly=True,
            samesite="Lax",
            secure=request.secure,
            path="/",
        )

    def _set_guest_cookie(self, response: web.StreamResponse, request: web.Request, token: str):
        response.set_cookie(
            "guest_token",
            token,
            max_age=GUEST_TOKEN_TTL_SECONDS,
            httponly=True,
            samesite="Lax",
            secure=request.secure,
            path="/",
        )

    def _get_current_user(self, request: web.Request):
        token = request.cookies.get("session_token")
        return get_session_user(token)

    async def introduction(self, request: web.Request) -> web.Response:
        """
        Serve the introduction HTML file. This is now the front page: visitors can read
        about AIRTC, try it as a guest, or log in via the button in the top-right corner.
        Args:
            request: The HTTP request object.
        Returns:
            web.Response: The response containing the HTML content.
        """
        content = open(os.path.join(ROOT, "webpage/introduction.html"), encoding="utf-8").read()
        return web.Response(content_type="text/html", text=content, charset='utf-8')

    async def login_page(self, request: web.Request) -> web.Response:
        """
        Serve the login/signup HTML file.
        Args:
            request: The HTTP request object.
        Returns:
            web.Response: The response containing the HTML content.
        """
        content = open(os.path.join(ROOT, "webpage/index.html"), encoding="utf-8").read()
        return web.Response(content_type="text/html", text=content, charset='utf-8')

    async def generate(self, request: web.Request) -> web.Response:
        """
        Serve the generate HTML file. Reachable without logging in — guests get a
        limited number of trial conversations, enforced in offer().
        Args:
            request: The HTTP request object.
        Returns:
            web.Response: The response containing the HTML content.
        """
        content = open(os.path.join(ROOT, "webpage/generate.html"), encoding="utf-8").read()
        return web.Response(content_type="text/html", text=content, charset='utf-8')

    async def javascript(self, request: web.Request) -> web.Response:
        """
        Serve the JavaScript client file.
        Args:
            request: The HTTP request object.
        Returns:
            web.Response: The response containing the JavaScript content.
        """
        content = open(os.path.join(ROOT, "webpage/static/js/main.js")).read()
        return web.Response(content_type="application/javascript", text=content)

    async def check_password(self, request: web.Request) -> web.Response:
        """
        Check the provided password against the secret key.
        Args:
            request: The HTTP request object.
        Returns:
            web.Response: The response indicating success or failure.
        """
        return await self.login(request)

    async def login(self, request: web.Request) -> web.Response:
        params = await request.json()
        username = params.get("username", "")
        password = params.get("password", "")
        user = authenticate_user(username, password)
        if user is None:
            return web.json_response({"ok": False, "message": "Invalid username or password"}, status=403)

        token = create_session(user["id"])
        response = web.json_response({"ok": True, "username": user["username"]})
        self._set_session_cookie(response, request, token)
        return response

    async def signup(self, request: web.Request) -> web.Response:
        params = await request.json()
        username = params.get("username", "")
        password = params.get("password", "")

        try:
            user = create_user(username, password)
        except ValueError as exc:
            return web.json_response({"ok": False, "message": str(exc)}, status=400)

        token = create_session(user["id"])
        response = web.json_response({"ok": True, "username": user["username"]}, status=201)
        self._set_session_cookie(response, request, token)
        return response

    async def offer(self, request: web.Request) -> web.Response:
        """
        Manage WebRTC connection.
        """
        user = self._get_current_user(request)
        guest_token = None

        if user is not None:
            conversation_count, conversation_limit = get_conversation_usage(user["id"])
            if conversation_count >= conversation_limit:
                return web.json_response({"ok": False, "message": "Conversation limit reached"}, status=429)
            increment_conversation_count(user["id"])
            conversation_id = create_conversation(user["id"])
        else:
            # No account: track trial usage against an opaque guest cookie instead of a user row.
            guest_token = request.cookies.get("guest_token")
            guest_count = get_guest_conversation_count(guest_token)
            if guest_count is None:
                guest_token = create_guest()
                guest_count = 0
            if guest_count >= GUEST_CONVERSATION_LIMIT:
                response = web.json_response(
                    {"ok": False, "message": "Guest trial used up — sign up for more conversations."},
                    status=429,
                )
                self._set_guest_cookie(response, request, guest_token)
                return response
            increment_guest_conversation_count(guest_token)
            # Guest conversations aren't persisted (no account to attach them to).
            conversation_id = None

        # Everything from here on can fail midway through negotiation (bad SDP, a slow/dead
        # STUN or TURN server, a client pipeline error, etc). If it does, we clean up whatever
        # got created, refund the conversation slot charged above (a failed connection
        # shouldn't cost the user part of their limited quota), and return a clean error
        # instead of a raw 500 crash page or a hung request.
        pc = None
        try:
            params = await request.json()
            processing_mode = params.get("processingMode", "online")
            # located in js
            offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
            self.logger.info(f"Received offer with processing mode: {processing_mode}")
            # Local mode (no STUN/TURN) is currently disabled; always connect online.
            pc = RTCPeerConnection(configuration=RTCConfiguration(
                    iceServers=[
                        RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
                        RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
                        # Does not work outside of China
                        # RTCIceServer(urls=["stun:stun.qq.com:3478"]),
                        RTCIceServer(
                            urls=[
                                "turn:relay.metered.ca:80",
                                "turn:relay.metered.ca:443",
                                "turn:relay.metered.ca:443?transport=tcp",
                            ],
                            username=os.getenv("TURN_USERNAME"),
                            credential=os.getenv("TURN_CREDENTIAL"),
                        ),
                    ]
                )
            )
            pc_id = f"PC-{uuid.uuid4().hex[:8]}"
            self.pcs.add(pc)
            log_channel = pc.createDataChannel("log")
            pc.log_channel = log_channel
            self.logger.info(f"Current {pc_id}: New connection established")

            # Create session-specific queues
            # asr_queue: transfer results of ASR to LLM
            # llm_queue: transfer results of LLM to TTS
            # audio_queue: transfer audio data from TTS to WebRTC browser player
            asr_queue = TranscriptLoggingQueue(conversation_id, maxsize=20)
            llm_queue = ResponseLoggingQueue(conversation_id, maxsize=20)
            audio_queue = asyncio.Queue(maxsize=100)

            # Create components
            stop_event = asyncio.Event()
            interrupt_event = asyncio.Event()
            recorder = MediaBlackhole()
            audio_player = AudioPlayer(
                audio_queue,
                sample_rate=AUDIO_SAMPLE_RATE,
                samples_per_frame=SAMPLES_PER_FRAME,
                format=FORMAT,
                layout=LAYOUT,
            )
            pc._audio_player = audio_player
            vad = create_client("vad",
                                self.args.vad,
                                threshold=VAD_THRESHOLD,
                                )
            # some ASR may require different sample rate!
            resampler = AudioResampler(rate=ASR_SAMPLE_RATE,
                                       layout=LAYOUT,
                                       format=FORMAT,
                                      )
            # Each ASR platform uses its own language code format:
            #   google  → BCP-47 tag  e.g. "zh-CN", "en-US"  (set asr_language in config.yaml)
            #   whisper → ISO-639-1   e.g. "zh", "en", or None for auto-detect
            #             (set whisper_language in config.yaml; ~ / null = auto-detect)
            asr_language_map = {
                "whisper": WHISPER_LANGUAGE_CODES,  # None triggers Whisper's built-in auto-detect
            }
            asr_language = asr_language_map.get(self.args.asr, ASR_LANGUAGE)
            asr_client = create_client("asr",
                                       platform=self.args.asr,
                                       rate=ASR_SAMPLE_RATE,
                                       language_code=asr_language,
                                       alternative_language_codes=ASR_ALTERNATIVE_LANGUAGES,
                                       chunk_size=ASR_CHUNK_SIZE,
                                       stop_word=STOP_WORD,
                                       pc=pc,
                                       )
            llm_credentials = {
                "google": dict(model=GOOGLE_LLM_MODEL, api_key=GOOGLE_AI_API_KEY, base_url=GOOGLE_LLM_BASE_URL),
                "baidu":  dict(model=LLM_MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL),
            }
            llm_cfg = llm_credentials.get(self.args.llm, llm_credentials["baidu"])
            llm_client = create_client("llm",
                                        platform=self.args.llm,
                                        pc=pc,
                                        **llm_cfg,
                                        )
            tts_client = create_client("tts",
                                        platform=self.args.tts,
                                        voice=AZURE_TTS_VOICE,
                                        key=AZURE_TTS_KEY,
                                        region=AZURE_TTS_REGION,
                                        pc=pc,
                                        )

            pc.addTrack(audio_player)

            @pc.on("datachannel")
            def on_datachannel(channel):
                @channel.on("message")
                def on_message(message):
                    try:
                        data = json.loads(message)
                        if data.get("type") == "clear_audio" and hasattr(pc, '_audio_player'):
                            pc._audio_player.request_interrupt()
                            self.logger.info(f"{pc_id}: Audio buffer cleared by client request")
                    except Exception:
                        pass

            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                self.logger.info(f"Current {pc_id}: Connection state: {pc.connectionState}")
                if pc.connectionState == 'connected':
                    msg = '\n' + '=' * 50 + '\nStart Recording\n' + '=' * 50
                    self.logger.info(msg)
                    server_to_client(pc.log_channel, msg)
                if pc.connectionState in ['closed', 'failed', 'disconnected']:
                    # Cancel all tasks
                    if hasattr(pc, '_stop_event'):
                        pc._stop_event.set()

                    tasks = []
                    for attr in ['_asr_task', '_llm_task', '_tts_task']:
                        if hasattr(pc, attr):
                            task = getattr(pc, attr)
                            if not task.done():
                                task.cancel()
                                tasks.append(task)

                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)

                    await pc.close()
                    self.pcs.discard(pc)
                    if conversation_id is not None:
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(None, end_conversation, conversation_id)
                    self.logger.info(f"{pc_id}: Connection cleaned up")

            @pc.on("track")
            async def on_track(track):
                self.logger.info(f"Current {pc_id}: Received {track.kind} track")
                if track.kind == "audio":
                    recorder.addTrack(track)
                    # Create control events to stop or interrupt connection
                    pc._stop_event = stop_event
                    pc._interrupt_event = interrupt_event
                    try:
                        pc._asr_task = asyncio.create_task(
                            asr_client.generate(
                                track,
                                asr_queue,
                                audio_player,
                                stop_event,
                                interrupt_event,
                                vad=vad,
                                resampler=resampler,
                                timeout=TIMEOUT,
                            )
                        )

                        pc._llm_task = asyncio.create_task(
                            llm_client.generate(
                                asr_queue,
                                llm_queue,
                                stop_event,
                                interrupt_event,
                                system=SYSTEM_PROMPT,
                                max_tokens=MAX_TOKENS,
                                timeout=TIMEOUT,
                            )
                        )

                        pc._tts_task = asyncio.create_task(
                            tts_client.generate(
                                llm_queue,
                                audio_queue,
                                stop_event,
                                interrupt_event,
                                samples_per_frame=SAMPLES_PER_FRAME,
                                timeout=TIMEOUT,
                            )
                        )

                    except Exception as e:
                        self.logger.error(f"Current {pc_id}: Pipeline error: {e}")
                        traceback.print_exc()

                @track.on("ended")
                async def on_ended():
                    msg = f"Current {pc_id}: Track ended"
                    self.logger.info(msg)
                    server_to_client(pc.log_channel, msg)
                    if hasattr(pc, "_stop_event"):
                        pc._stop_event.set()
                    await recorder.stop()

            # Complete WebRTC setup
            await pc.setRemoteDescription(offer)
            await recorder.start()

            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            response = web.Response(
                content_type="application/json",
                text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
            )
        except Exception:
            self.logger.error(f"Failed to establish WebRTC connection: {traceback.format_exc()}")
            if pc is not None:
                self.pcs.discard(pc)
                await pc.close()
            if user is not None:
                decrement_conversation_count(user["id"])
            else:
                decrement_guest_conversation_count(guest_token)
            response = web.json_response(
                {"ok": False, "message": "Failed to establish connection — please try again."},
                status=500,
            )
            if guest_token is not None:
                self._set_guest_cookie(response, request, guest_token)
            return response

        if guest_token is not None:
            self._set_guest_cookie(response, request, guest_token)
        return response

    async def on_shutdown(self, app: web.Application) -> None:
        """
        Handle the shutdown of the application.
        Args:
            app: The web application instance.
        """
        coros = [pc.close() for pc in self.pcs]
        await asyncio.gather(*coros)
        self.pcs.clear()

    def run_web_app(self, ssl_context=None):
        """
        Run the web application.
        Args:
            ssl_context: The SSL context for HTTPS (if any).
        """
        app = web.Application()
        app.on_shutdown.append(self.on_shutdown)
        print(os.path.join(ROOT, "webpage"))
        app.router.add_get("/", self.introduction)
        app.router.add_get("/introduction.html", self.introduction)
        app.router.add_get("/index.html", self.login_page)
        app.router.add_get("/generate.html", self.generate)
        app.router.add_post("/offer", self.offer)
        app.router.add_post("/check-password", self.check_password)
        app.router.add_post("/login", self.login)
        app.router.add_post("/signup", self.signup)
        app.router.add_static("/static/", path=os.path.join(ROOT, "webpage", "static"), show_index=True)
        if PORT:
            port = int(PORT)
        else:
            port = self.args.port
        web.run_app(app, access_log=None, host=self.args.host, port=port, ssl_context=ssl_context)

    def generate_args(self: object) -> argparse.Namespace:
        """
        Generate command-line arguments for webpage WebRTC
        Args:
            self (object): class instance
        Return:
            argparse.Namespace: The parsed command-line arguments.
        """
        parser = argparse.ArgumentParser(description="Minimal WebRTC audio logger")
        parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
        parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
        parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)")
        parser.add_argument("--port", type=int, default=8081, help="Port for HTTP server (default: 8081)")
        parser.add_argument("--asr", type=str, default="google", help="ASR model to use")
        parser.add_argument("--llm", type=str, default="google", help="LLM model to use")
        parser.add_argument("--tts", type=str, default="azure", help="TTS model to use")
        parser.add_argument("--vad", type=str, default="simple", help="VAD model to use")
        return parser.parse_args()

if __name__ == "__main__":
    setup_logging()
    webpage = WebPage()
    ssl_context = get_ssl_context(webpage.args)
    webpage.run_web_app(ssl_context=ssl_context)
