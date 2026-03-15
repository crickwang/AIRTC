
import argparse
import asyncio
import io
import json
import logging
import os
import queue
import re
import ssl
import threading
import time
import traceback
import unicodedata as ucd
import uuid
from collections import OrderedDict, deque
from fractions import Fraction

import azure.cognitiveservices.speech as speechsdk
import numpy as np
import pyaudio as pa
import sounddevice as sd
import speech_recognition as sr
import torch.hub
import yaml
from aiohttp import web
from aiortc import AudioStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame
from av.audio.resampler import AudioResampler
from dotenv import load_dotenv
from funasr import AutoModel
from google.api_core import client_options

# ASR imports
from google.cloud import speech_v1 as speech
from google.cloud import texttospeech
from google.protobuf import duration_pb2
from openai import OpenAI
from pydub import AudioSegment
from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

# FunASR imports
from asr.funasr_stream import ParaformerStreaming
from asr.google_transcription import AudioStream, listen_print_loop

# # OpenAI RealTime WebRTC import
# from asr.openai_transcription import OpenAITranscription
# LLM import - OpenAI, Baidu Proxy
from llm.baidu import BaiduClient

# EdgeTTS import
from tts.edge_tts_stream import EdgeTTS
