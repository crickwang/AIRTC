from ollama import chat
import asyncio
import edge_tts.submaker as submaker
import edge_tts.communicate as communicate
import yaml
import pywhispercpp.constants
from pywhispercpp.model import Model
import pywhispercpp
from pydub import AudioSegment
import os
from pathlib import Path

with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

llm_model = config.get('llm_model', 'llama3:8b')
asr_model = config.get('asr_model', 'large-v3-turbo-q8')
