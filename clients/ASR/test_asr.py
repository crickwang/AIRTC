import asyncio
import hashlib
import json
import logging
import os
import traceback
import uuid

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

from audio_player.audio_player import MockAudioPlayer
from config.constants import *
from utils import *
from utils import create_client

asr_client = create_client("asr",
                            platform="google",
                            rate=ASR_SAMPLE_RATE,
                            language_code=ASR_LANGUAGE,
                            chunk_size=ASR_CHUNK_SIZE,
                            stop_word=STOP_WORD,
                            )



