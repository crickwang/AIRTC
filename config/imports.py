
import asyncio
import json
import logging
import os
import ssl
import traceback
import uuid
import yaml
import argparse
import numpy as np
import threading
import queue
import time
import re
from av import AudioFrame
from pydub import AudioSegment
import io
from fractions import Fraction
import sounddevice as sd
import pyaudio as pa
from collections import OrderedDict, deque

from dotenv import load_dotenv
from aiohttp import web
from aiortc import AudioStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole
from av.audio.resampler import AudioResampler
from aiortc.mediastreams import MediaStreamError
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import torch.hub

# ASR imports
from google.cloud import speech_v1 as speech, texttospeech
from google.protobuf import duration_pb2
from google.api_core import client_options
from asr.google_transcription import AudioStream, listen_print_loop

# FunASR imports
from asr.funasr_stream import ParaformerStreaming
from funasr import AutoModel 

# # OpenAI RealTime WebRTC import
# from asr.openai_transcription import OpenAITranscription

# LLM import - OpenAI, Baidu Proxy
from llm.baidu import BaiduClient 

# EdgeTTS import
from tts.edge_tts_stream import EdgeTTS

import azure.cognitiveservices.speech as speechsdk