import asyncio
import json
import logging
import os
import ssl
import uuid

from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole

ROOT = os.path.dirname(__file__)
logger = logging.getLogger("pc")
pcs = set()

class AudioPrinterTrack(MediaStreamTrack):
    kind = "audio"
    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        pcm = frame.to_ndarray()
        print(f"Received audio chunk: shape={pcm.shape}, dtype={pcm.dtype} with sample rate {frame.sample_rate}")
        return frame

async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

async def javascript(request):
    content = open(os.path.join(ROOT, "js/client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    recorder = MediaBlackhole()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            audio_track = AudioPrinterTrack(track)
            recorder.addTrack(audio_track)

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(offer)
    await recorder.start()

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
    )

async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Minimal WebRTC audio logger")
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8081, help="Port for HTTP server (default: 8081)")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    ssl_context = None
    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/js/client.js", javascript)
    app.router.add_post("/offer", offer)
    app.router.add_static("/", ".", show_index=True)
    web.run_app(app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context)