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
from aiortc.mediastreams import MediaStreamError
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

# How long a pre-connected ("warm") peer connection may sit without the client
# sending "activate" before the server closes it. Warm connections are opened on
# page load, so every visitor — including bots — holds one; this bounds that cost.
WARM_IDLE_TIMEOUT_S = 300


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
        Warm-connect: negotiate the WebRTC transport only.

        Called on page load, before the user clicks Start, so the slow part of the
        handshake (ICE/DTLS) is already done by the time they want to talk. Nothing
        here charges quota or constructs ASR/LLM/TTS clients — all of that happens
        when the client sends an "activate" message over the data channel (see
        _activate_session), so a page view alone costs neither a conversation slot
        nor any external API session.
        """
        user = self._get_current_user(request)
        guest_token = None
        if user is None:
            # Resolve (or mint) the guest identity now, while we still have HTTP
            # cookies in hand — the later "activate" arrives over the data channel,
            # which can't set cookies. No usage is charged here.
            guest_token = request.cookies.get("guest_token")
            if get_guest_conversation_count(guest_token) is None:
                guest_token = create_guest()

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
            self.logger.info(f"Current {pc_id}: New warm connection")

            # Identity and session state stashed for activation time.
            pc._user = user
            pc._guest_token = guest_token
            pc._activated = False
            pc._conversation_id = None
            pc._client_track = None
            pc._stop_event = asyncio.Event()
            pc._interrupt_event = asyncio.Event()

            # The return-audio track must be in the SDP answer, so the player is
            # attached at warm time — a local object that costs nothing until TTS
            # actually feeds its queue.
            audio_queue = asyncio.Queue(maxsize=100)
            audio_player = AudioPlayer(
                audio_queue,
                sample_rate=AUDIO_SAMPLE_RATE,
                samples_per_frame=SAMPLES_PER_FRAME,
                format=FORMAT,
                layout=LAYOUT,
            )
            pc._audio_player = audio_player
            pc._audio_queue = audio_queue
            pc.addTrack(audio_player)

            @pc.on("datachannel")
            def on_datachannel(channel):
                @channel.on("message")
                def on_message(message):
                    try:
                        data = json.loads(message)
                    except Exception:
                        return
                    msg_type = data.get("type")
                    if msg_type == "clear_audio" and hasattr(pc, '_audio_player'):
                        pc._audio_player.request_interrupt()
                        self.logger.info(f"{pc_id}: Audio buffer cleared by client request")
                    elif msg_type == "activate":
                        asyncio.create_task(self._activate_session(pc, pc_id))

            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                self.logger.info(f"Current {pc_id}: Connection state: {pc.connectionState}")
                if pc.connectionState in ['closed', 'failed', 'disconnected']:
                    pc._stop_event.set()

                    for attr in ['_idle_task', '_drain_task']:
                        task = getattr(pc, attr, None)
                        if task is not None and not task.done():
                            task.cancel()

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
                    if pc._conversation_id is not None:
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(None, end_conversation, pc._conversation_id)
                    self.logger.info(f"{pc_id}: Connection cleaned up")

            @pc.on("track")
            async def on_track(track):
                self.logger.info(f"Current {pc_id}: Received {track.kind} track")
                if track.kind == "audio":
                    pc._client_track = track
                    # Nothing consumes the track until activation; drain it so
                    # aiortc's per-track frame queue doesn't grow without bound.
                    pc._drain_task = asyncio.create_task(self._drain_track(track))

                @track.on("ended")
                async def on_ended():
                    msg = f"Current {pc_id}: Track ended"
                    self.logger.info(msg)
                    server_to_client(pc.log_channel, msg)
                    pc._stop_event.set()

            # Complete WebRTC setup
            await pc.setRemoteDescription(offer)

            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            pc._idle_task = asyncio.create_task(self._close_if_never_activated(pc, pc_id))

            response = web.Response(
                content_type="application/json",
                text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
            )
        except Exception:
            self.logger.error(f"Failed to establish WebRTC connection: {traceback.format_exc()}")
            if pc is not None:
                self.pcs.discard(pc)
                await pc.close()
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

    async def _drain_track(self, track):
        """Consume and discard frames from a track nobody is using yet."""
        try:
            while True:
                await track.recv()
        except (MediaStreamError, asyncio.CancelledError):
            pass

    async def _close_if_never_activated(self, pc, pc_id):
        """Reap warm connections whose page view never turned into a session."""
        try:
            await asyncio.sleep(WARM_IDLE_TIMEOUT_S)
        except asyncio.CancelledError:
            return
        if not pc._activated and pc.connectionState not in ("closed", "failed"):
            self.logger.info(
                f"{pc_id}: Warm connection idle for {WARM_IDLE_TIMEOUT_S}s without activation, closing"
            )
            # Detach ourselves first so the close-triggered cleanup handler doesn't
            # cancel this task in the middle of its own pc.close() call.
            pc._idle_task = None
            await pc.close()
            self.pcs.discard(pc)

    async def _activate_session(self, pc, pc_id):
        """
        Handle the client's "activate" data-channel message: charge quota, build the
        ASR/LLM/TTS pipeline, and start it on the already-negotiated connection.

        This is the second half of what offer() used to do in one shot. It only runs
        when the user actually clicks Start, so external API clients are never
        constructed for a visitor who merely loads the page.
        """
        def reply(ok, message=""):
            try:
                pc.log_channel.send(json.dumps({"type": "activate_result", "ok": ok, "message": message}))
            except Exception:
                self.logger.warning(f"{pc_id}: Could not deliver activate_result")

        if pc._activated:
            reply(True)
            return
        if pc._client_track is None:
            reply(False, "No audio track on this connection — please reload the page.")
            return

        user = pc._user
        guest_token = pc._guest_token
        conversation_id = None
        charged = False
        try:
            # Quota is checked and charged here, not at negotiation time, so page
            # views (and failed handshakes) never cost a conversation slot.
            if user is not None:
                conversation_count, conversation_limit = get_conversation_usage(user["id"])
                if conversation_count >= conversation_limit:
                    reply(False, "Conversation limit reached")
                    return
                increment_conversation_count(user["id"])
                charged = True
                conversation_id = create_conversation(user["id"])
            else:
                guest_count = get_guest_conversation_count(guest_token)
                if guest_count is None or guest_count >= GUEST_CONVERSATION_LIMIT:
                    reply(False, "Guest trial used up — sign up for more conversations.")
                    return
                increment_guest_conversation_count(guest_token)
                charged = True
                # Guest conversations aren't persisted (no account to attach them to).

            pc._activated = True
            pc._conversation_id = conversation_id

            for attr in ('_idle_task', '_drain_task'):
                task = getattr(pc, attr, None)
                if task is not None and not task.done():
                    task.cancel()
            # Make sure the drain task has released track.recv() before ASR takes over.
            drain_task = getattr(pc, '_drain_task', None)
            if drain_task is not None:
                await asyncio.gather(drain_task, return_exceptions=True)
                pc._drain_task = None

            # Create session-specific queues
            # asr_queue: transfer results of ASR to LLM
            # llm_queue: transfer results of LLM to TTS
            # audio_queue (from warm-connect): transfer audio from TTS to browser player
            asr_queue = TranscriptLoggingQueue(conversation_id, maxsize=20)
            llm_queue = ResponseLoggingQueue(conversation_id, maxsize=20)
            audio_queue = pc._audio_queue
            audio_player = pc._audio_player
            stop_event = pc._stop_event
            interrupt_event = pc._interrupt_event

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

            pc._asr_task = asyncio.create_task(
                asr_client.generate(
                    pc._client_track,
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

            reply(True)
            msg = '\n' + '=' * 50 + '\nStart Recording\n' + '=' * 50
            self.logger.info(msg)
            server_to_client(pc.log_channel, msg)
            self.logger.info(f"{pc_id}: Session activated")
        except Exception:
            self.logger.error(f"{pc_id}: Activation failed: {traceback.format_exc()}")
            pc._activated = False
            pc._conversation_id = None
            if charged:
                # A failed activation shouldn't cost part of a limited quota.
                if user is not None:
                    decrement_conversation_count(user["id"])
                else:
                    decrement_guest_conversation_count(guest_token)
            # Put the connection back into a consistent warm state so the client
            # can retry: resume draining and re-arm the idle reaper.
            if pc._client_track is not None and pc.connectionState not in ("closed", "failed"):
                pc._drain_task = asyncio.create_task(self._drain_track(pc._client_track))
                pc._idle_task = asyncio.create_task(self._close_if_never_activated(pc, pc_id))
            reply(False, "Failed to start the session — please try again.")

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
