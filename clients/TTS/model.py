from abc import ABC, abstractmethod
from email.mime import text
import azure.cognitiveservices.speech as speechsdk
import edge_tts.submaker as submaker
import edge_tts.communicate as communicate
from google.cloud import texttospeech
import asyncio
import time
import queue
import threading
import numpy as np
from register import register
from config.constants import TIMEOUT
from pathlib import Path
import re
from config.constants import AUDIO_CHUNK_SIZE
from clients.utils import log_to_client

class TTSClient(ABC):
    def __init__(self, **kwargs):
        super().__init__()
        
    @abstractmethod
    async def generate(
        self: object, 
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        interrupt_event: asyncio.Event,
        **kwargs
    ):
        """
        Generate speech from text using the TTS service.
        The workflow of this method is to: \n
        1. Receive LLM-generated text from the input queue.
        2. Process the text and generate speech.
        3. Send the speech to the output queue.
        4. Use the stop_event to signal when to terminate the program.
        5. Use the interrupt_event to signal when to interrupt other clients.
        
        Args:
            self (object): The TTS client instance.
            input_queue (asyncio.Queue): The input queue containing LLM-generated text.
            output_queue (asyncio.Queue): The output queue for sending audio chunks.
            stop_event (asyncio.Event): The event to signal stopping the generation.
            interrupt_event (asyncio.Event): The event to signal interrupting the TTS.
            **kwargs: Additional keyword arguments.
        """
        pass
  
@register.add_model("tts", "azure")
class AzureTTS(TTSClient):
    def __init__(self, voice: str, key: str, region: str, **kwargs):
        """
        Initialize AzureTTS instance.
        Args:
            voice (str): The voice name to use for TTS.
            key (str): The API key for Azure Cognitive Services.
            region (str): The Azure region for the service.
        """
        super().__init__()
        self.voice = voice
        self.config = speechsdk.SpeechConfig(
            subscription=key, 
            region=region,
        )
        self.config.speech_synthesis_voice_name = voice
        self.config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm
        )
        self.synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.config,
            audio_config=None,
        )
        self.connection = speechsdk.Connection.from_speech_synthesizer(self.synthesizer)
        self.connection.open(True)
        self.pc = kwargs.get('pc', None)
        
    async def generate(
        self: object,
        input_queue: asyncio.Queue, 
        output_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        interrupt_event: asyncio.Event,
        samples_per_frame: int,
        timeout: int = TIMEOUT
    ) -> None:
        """
        Azure TTS generation
        Args:
            input_queue (asyncio.Queue): Queue for incoming text chunks from LLM.
            output_queue (asyncio.Queue): Queue for outgoing audio chunks to WebRTC peer.
            stop_event (asyncio.Event): Event to signal stopping the program.
            interrupt_event (asyncio.Event): Event to signal interrupting the TTS.
            samples_per_frame (int): Number of audio samples per frame.
            timeout (float, optional): Timeout for waiting on input. Default to 600 seconds.
        Returns:
            None
        Raises:
            Exception: If an error occurs during processing.
        """
        total_samples_generated = 0
        sentence_count = 0
        
        try:
            current_tts_task = None
            while not stop_event.is_set():
                try:
                    # Get text from LLM
                    text = await asyncio.wait_for(input_queue.get(), timeout=timeout)
                    
                    if stop_event.is_set():
                        msg = "TTS: Stopping"
                        print(msg)
                        if self.pc:
                            log_to_client(self.pc.log_channel, msg)
                        break
                    
                    if interrupt_event.is_set():
                        msg = "TTS: Interrupted, clearing buffer"
                        print(msg)
                        if self.pc:
                            log_to_client(self.pc.log_channel, msg)
                        # Clear output queue
                        try:
                            while True:
                                output_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        continue
                    
                    if text is None:
                        msg = "TTS: End of input from LLM"
                        print(msg)
                        if self.pc:
                            log_to_client(self.pc.log_channel, msg)
                        # Wait for current TTS to finish
                        if current_tts_task and not current_tts_task.done():
                            await current_tts_task
                        continue
                    
                    print(f"TTS: Received text: '{text[:50]}...'")
                    sentences = text.split()
                    
                    for sentence in sentences:
                        if stop_event.is_set() or interrupt_event.is_set():
                            msg = "TTS: Interrupted during sentence processing"
                            print(msg)
                            if self.pc:
                                log_to_client(self.pc.log_channel, msg)
                            break
                        
                        if sentence.strip():
                            sentence_count += 1
                            
                            # Wait for previous TTS to finish before starting new one
                            if current_tts_task and not current_tts_task.done():
                                await current_tts_task
                            
                            # Start new streaming TTS
                            current_tts_task = asyncio.create_task(
                                self.process_text(
                                    sentence.strip(), 
                                    output_queue, 
                                    samples_per_frame, 
                                    stop_event, 
                                    interrupt_event, 
                                    sentence_count
                                )
                            )
                            
                            # For the first sentence, wait a bit to get audio flowing
                            if sentence_count == 1:
                                try:
                                    await asyncio.wait_for(asyncio.shield(current_tts_task), timeout=0.5)
                                except asyncio.TimeoutError:
                                    pass  # Continue processing
                            
                            samples = await current_tts_task
                            total_samples_generated += samples
                    
                except asyncio.TimeoutError:
                    print("TTS: Timeout waiting for input")
                    break
                except Exception as e:
                    print(f"TTS: Error processing text: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            print(f"TTS: Completed - {sentence_count} sentences, {total_samples_generated} samples")
            
        finally:
            await output_queue.put(None)
            msg = "TTS: end of processing"
            print(msg)
            if self.pc:
                log_to_client(self.pc.log_channel, msg)


    async def process_text(
        self: object,
        sentence: str, 
        output_queue: asyncio.Queue, 
        samples_per_frame: int,
        stop_event: asyncio.Event,
        interrupt_event: asyncio.Event,
        sentence_id: int,
    ) -> int:
        """
        Process TTS with Azure true streaming - output audio chunks as they arrive.
        Uses the synthesizing event for immediate audio streaming.
        """
        
        if stop_event.is_set() or interrupt_event.is_set():
            print(f"TTS: Sentence {sentence_id} skipped due to interruption")
            return

        print(f"TTS: Processing sentence {sentence_id}: '{sentence[:50]}...'")

        if not sentence.strip():
            return 0
        
        start_time = time.time()
        total_samples = 0
        first_chunk_time = None
        
        # Thread-safe containers
        audio_chunk_queue = queue.Queue()
        error_container = [None]
        synthesis_complete = threading.Event()
        
        try:
            msg = f"TTS: Starting streaming for sentence {sentence_id}: '{sentence}...'"
            print(msg)
            if self.pc:
                log_to_client(self.pc.log_channel, msg)

            def on_synthesizing(evt):
                """Handle audio chunks as they arrive - TRUE STREAMING"""
                nonlocal first_chunk_time, total_samples
                
                audio_data = evt.result.audio_data
                
                if interrupt_event.is_set() or stop_event.is_set():
                    print(f"TTS: Interrupt event set during synthesis of sentence {sentence_id}")
                    error_container[0] = "Interrupted or stopped"
                    audio_chunk_queue.put(('error', "Interrupted or stopped"))
                    synthesis_complete.set()
                    return
                
                if audio_data and len(audio_data) > 0:
                    # Track first chunk timing
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        latency = first_chunk_time - start_time
                        # print(f"TTS: First audio chunk for sentence {sentence_id} in {latency:.3f}s")
                    
                    # Put audio chunk in queue immediately
                    audio_chunk_queue.put(('audio', audio_data))
                    total_samples += len(audio_data) // 2  # 16-bit samples
                    # print(f"TTS: Streamed {len(audio_data)} bytes for sentence {sentence_id}")
            
            def on_synthesis_completed(evt):
                """Handle synthesis completion"""
                if evt.result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                    error_msg = f"Synthesis failed: {evt.result.reason}"
                    error_container[0] = error_msg
                    audio_chunk_queue.put(('error', error_msg))
                # else:
                #     # Log latency metrics if available
                #     if evt.result.properties:
                #         first_byte_latency = evt.result.properties.get_property(
                #             speechsdk.PropertyId.SpeechServiceResponse_SynthesisFirstByteLatencyMs
                #         )
                #         if first_byte_latency:
                #             print(f"TTS: Sentence {sentence_id} - First byte latency: {first_byte_latency}ms")
                
                audio_chunk_queue.put(('done', None))
                synthesis_complete.set()
            
            def on_synthesis_cancelled(evt):
                """Handle synthesis cancellation"""
                error_msg = f"Synthesis cancelled: {evt.result.cancellation_details.error_details}"
                error_container[0] = error_msg
                audio_chunk_queue.put(('error', error_msg))
                synthesis_complete.set()
            
            # Connect event handlers for streaming
            self.synthesizer.synthesizing.connect(on_synthesizing)
            self.synthesizer.synthesis_completed.connect(on_synthesis_completed)
            self.synthesizer.synthesis_canceled.connect(on_synthesis_cancelled)

            # Start synthesis (non-blocking)
            def start_synthesis():
                self.synthesizer.speak_text_async(sentence)

            # Run synthesis in background thread
            synthesis_thread = threading.Thread(target=start_synthesis, daemon=True)
            synthesis_thread.start()
            
            # Process audio chunks as they arrive
            while not synthesis_complete.is_set() or not audio_chunk_queue.empty():
                try:
                    # Get next chunk with short timeout
                    chunk_type, chunk_data = audio_chunk_queue.get(timeout=0.1)
                    
                    if chunk_type == 'error':
                        print(f"TTS: Error in sentence {sentence_id}: {chunk_data}")
                        break
                    elif chunk_type == 'done':
                        print(f"TTS: Streaming complete for sentence {sentence_id}")
                        break
                    elif chunk_type == 'audio':
                        # Convert bytes to numpy array (16-bit PCM)
                        audio_np = np.frombuffer(chunk_data, dtype=np.int16)
                        
                        if len(audio_np) > 0:
                            # Apply audio improvements
                            #audio_np = apply_audio_improvements(audio_np)
                            
                            # Send immediately
                            await self.send_audio(audio_np, output_queue, samples_per_frame)
                    
                except queue.Empty:
                    # Continue waiting for more chunks
                    await asyncio.sleep(0.01)
            
            # Disconnect event handlers
            self.synthesizer.synthesizing.disconnect_all()
            self.synthesizer.synthesis_completed.disconnect_all()
            self.synthesizer.synthesis_canceled.disconnect_all()

            processing_time = time.time() - start_time
            print(f"TTS: Sentence {sentence_id} completed in {processing_time:.2f}s, {total_samples} samples")

            return total_samples
            
        except Exception as e:
            print(f"TTS: Error in Azure streaming TTS for sentence {sentence_id}: {e}")
            import traceback
            traceback.print_exc()
            return 0

    async def send_audio(
        self: object, 
        audio_np: np.ndarray, 
        output_queue: asyncio.Queue, 
        samples_per_frame: int
    ) -> None:
        """Send audio chunks immediately without waiting for complete sentence."""
        if len(audio_np) == 0:
            return

        chunk_size = samples_per_frame  # 10ms chunks
        
        for i in range(0, len(audio_np), chunk_size):
            chunk = audio_np[i:min(i + chunk_size, len(audio_np))]

            if len(chunk) < samples_per_frame:
                chunk = np.pad(chunk, (0, samples_per_frame - len(chunk)), 'constant')

            await output_queue.put(chunk)
            await asyncio.sleep(0.001)

class EdgeTTS(TTSClient):
    def __init__(
        self: object, 
        voice: str, 
        output_file: str = None, 
        srt_file: str = None
    ):
        self.voice = voice
        if output_file is not None:
            self.output_file = Path(output_file)
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            self.output_file.open("w").close()
        else:
            self.output_file = None
        if srt_file is not None:
            self.srt_file = Path(srt_file)
            self.srt_file.parent.mkdir(parents=True, exist_ok=True)
            self.srt_file.open("w").close()
        else:
            self.srt_file = None
            
    async def generate(
        self: object,
        input_queue: asyncio.Queue, 
        output_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        interrupt_event: asyncio.Event,
        samples_per_frame: int,
        timeout: int = TIMEOUT
    ) -> None:
        """
        Edge TTS generation
        Args:
            input_queue (asyncio.Queue): Queue for incoming text chunks from LLM.
            output_queue (asyncio.Queue): Queue for outgoing audio chunks to WebRTC peer.
            stop_event (asyncio.Event): Event to signal stopping the program.
            interrupt_event (asyncio.Event): Event to signal interrupting the TTS.
            samples_per_frame (int): Number of audio samples per frame.
            timeout (float, optional): Timeout for waiting on input. Default to 600 seconds.
        Returns:
            None
        Raises:
            Exception: If an error occurs during processing.
        """
        total_samples_generated = 0
        sentence_count = 0
        
        try:
            current_tts_task = None
            while not stop_event.is_set():
                try:
                    # Get text from LLM
                    text = await asyncio.wait_for(input_queue.get(), timeout=timeout)
                    
                    if stop_event.is_set():
                        print("TTS: Stopping")
                        break
                    
                    if interrupt_event.is_set():
                        print("TTS: Interrupted, clearing buffer")
                        # Clear output queue
                        try:
                            while True:
                                output_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        continue
                    
                    if text is None:
                        print("TTS: End of input from LLM")
                        # Wait for current TTS to finish
                        if current_tts_task and not current_tts_task.done():
                            await current_tts_task
                        continue
                    
                    print(f"TTS: Received text: '{text[:50]}...'")
                    sentences = text.split()
                    
                    for sentence in sentences:
                        if stop_event.is_set() or interrupt_event.is_set():
                            print("TTS: Interrupted during sentence processing")
                            break
                        
                        if sentence.strip():
                            sentence_count += 1
                            
                            # Wait for previous TTS to finish before starting new one
                            if current_tts_task and not current_tts_task.done():
                                await current_tts_task
                            
                            # Start new streaming TTS
                            current_tts_task = asyncio.create_task(
                                self.process_text(
                                    sentence.strip(), 
                                    output_queue, 
                                    samples_per_frame, 
                                    stop_event, 
                                    interrupt_event, 
                                    sentence_count
                                )
                            )
                            
                            # For the first sentence, wait a bit to get audio flowing
                            if sentence_count == 1:
                                try:
                                    await asyncio.wait_for(asyncio.shield(current_tts_task), timeout=0.5)
                                except asyncio.TimeoutError:
                                    pass  # Continue processing
                            
                            samples = await current_tts_task
                            total_samples_generated += samples
                    
                except asyncio.TimeoutError:
                    print("TTS: Timeout waiting for input")
                    break
                except Exception as e:
                    print(f"TTS: Error processing text: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            print(f"TTS: Completed - {sentence_count} sentences, {total_samples_generated} samples")
            
        finally:
            await output_queue.put(None)
            print("TTS: Sent end marker")

    async def process_text(
        self: object,
        sentence: str, 
        output_queue: asyncio.Queue, 
        samples_per_frame: int,
        stop_event: asyncio.Event,
        interrupt_event: asyncio.Event,
        sentence_id: int,
        ) -> int:
        """
        Process TTS with Azure true streaming - output audio chunks as they arrive.
        Uses the synthesizing event for immediate audio streaming.
        """
        
        if stop_event.is_set() or interrupt_event.is_set():
            print(f"TTS: Sentence {sentence_id} skipped due to interruption")
            return

        print(f"TTS: Processing sentence {sentence_id}: '{sentence[:50]}...'")

        if not sentence:
            return
        try:
            chunks = []
            com = communicate.Communicate(text=sentence, voice=self.voice)
            sub = submaker.SubMaker() if self.srt_file else None
            if self.output_file is None:
                async for chunk in com.stream():
                    if chunk["type"] == "audio":
                        chunks.append(chunk["data"])

            else:
                with self.output_file.open("ab") as file:
                    if self.srt_file is None:
                        async for chunk in com.stream():
                            if chunk["type"] == "audio":
                                file.write(chunk["data"])
                                chunks.append(chunk["data"])

                    else:
                        async for chunk in com.stream():
                            if chunk["type"] == "audio":
                                file.write(chunk["data"])
                                chunks.append(chunk["data"])
                            elif chunk["type"] == "subtitle":
                                sub.feed(chunk)
                        self.srt_file.write_text(sub.get_srt(), encoding='utf-8')
            self.send_audio(np.array(chunks), output_queue, samples_per_frame)
            
        except Exception as e:
            print(f"TTS: Error in Azure streaming TTS for sentence {sentence_id}: {e}")
            import traceback
            traceback.print_exc()
            return 0

    async def send_audio(
        self: object, 
        audio_np: np.ndarray, 
        output_queue: asyncio.Queue, 
        samples_per_frame: int
    ) -> None:
        """Send audio chunks immediately without waiting for complete sentence."""
        if len(audio_np) == 0:
            return

        chunk_size = samples_per_frame  # 10ms chunks
        
        for i in range(0, len(audio_np), chunk_size):
            chunk = audio_np[i:min(i + chunk_size, len(audio_np))]

            if len(chunk) < samples_per_frame:
                chunk = np.pad(chunk, (0, samples_per_frame - len(chunk)), 'constant')

            await output_queue.put(chunk)
            await asyncio.sleep(0.001)
            
class GoogleTTS(TTSClient):
    def __init__(self, voice: str, language: str):
        """
        Initialize GoogleTTS instance.
        Args:
            voice (str): The voice name to use for TTS.
            language (str): The language code to use for TTS.
        """
        super().__init__()
        self.client = texttospeech.TextToSpeechClient()
        self.config = texttospeech.StreamingSynthesizeConfig(
            voice=texttospeech.VoiceSelectionParams(
                name=voice,
                language_code=language
            )
        )
        self.request = texttospeech.StreamingSynthesizeRequest(
            streaming_config=self.config
        )

    async def generate(
        self,
        input_queue: asyncio.Queue, 
        output_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        interrupt_event: asyncio.Event,
        timeout=TIMEOUT,
    ) -> None:
        """
        Azure TTS
        Args:
            input_queue (asyncio.Queue): Queue for incoming text chunks from LLM.
            output_queue (asyncio.Queue): Queue for outgoing audio chunks to WebRTC peer.
            stop_event (asyncio.Event): Event to signal stopping the program.
            interrupt_event (asyncio.Event): Event to signal interrupting the TTS.
            timeout (float, optional): Timeout for waiting on input. Default to 600 seconds.
        Returns:
            None
        Raises:
            Exception: If an error occurs during processing.
        """
        try:
            text_buffer = ""
            current_sentence_id = 0
            
            while not stop_event.is_set():
                try:
                    # Get text from LLM
                    text_chunk = await asyncio.wait_for(input_queue.get(), timeout=timeout)
                    
                    if stop_event.is_set():
                        print("TTS: Stopping")
                        break
                    
                    if interrupt_event.is_set():
                        print("TTS: Interrupted, clearing buffer")
                        text_buffer = ""
                        # Clear output queue
                        try:
                            while True:
                                output_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        continue
                    
                    if text_chunk is None:
                        print("TTS: End of input")
                        # Process remaining buffer
                        if text_buffer.strip():
                            current_sentence_id += 1
                            await self.process_text(
                                text_buffer.strip(), 
                                output_queue, 
                                stop_event,
                                interrupt_event, 
                                current_sentence_id
                            )
                        break
                    
                    text_buffer += text_chunk
                    
                    # Extract complete sentences
                    sentences, text_buffer = self.extract_complete_sentences(text_buffer)
                    
                    for sentence in sentences:
                        if stop_event.is_set() or interrupt_event.is_set():
                            print("TTS: Interrupted during sentence processing")
                            break
                        
                        if sentence.strip():
                            current_sentence_id += 1
                            await self.process_text(
                                sentence.strip(), 
                                output_queue, 
                                stop_event,
                                interrupt_event, 
                                current_sentence_id
                            )
                    
                except asyncio.TimeoutError:
                    print("TTS: Timeout waiting for input")
                    break
                except Exception as e:
                    print(f"TTS: Error: {e}")
                    break
        
        finally:
            await output_queue.put(None)
            print("TTS: Pipeline ended")


    async def process_text(
        self: object,
        text: str, 
        output_queue: asyncio.Queue, 
        stop_event: asyncio.Event,
        interrupt_event: asyncio.Event, 
        sentence_id: int
    ) -> None:
        """
        Process TTS text with two threads to ensure low-latency audio streaming.
        """
        if stop_event.is_set() or interrupt_event.is_set():
            print(f"TTS: Sentence {sentence_id} skipped due to interruption")
            return
        
        print(f"TTS: Processing sentence {sentence_id}: '{text[:50]}...'")
        
        try:
            # Create streaming request
            def request_generator():
                yield self.request
                yield texttospeech.StreamingSynthesizeRequest(
                    input=texttospeech.StreamingSynthesisInput(text=text)
                )

            # Process TTS with interruption checking
            audio_chunk_queue = queue.Queue()
            
            def run_streaming_tts():
                try:
                    streaming_responses = self.client.streaming_synthesize(request_generator())
                    for response in streaming_responses:
                        if interrupt_event.is_set():
                            print(f"TTS: Sentence {sentence_id} interrupted during generation")
                            break
                        if response.audio_content:
                            audio_chunk_queue.put(('audio', response.audio_content))
                    audio_chunk_queue.put(('done', None))
                except Exception as e:
                    audio_chunk_queue.put(('error', str(e)))
            
            tts_thread = threading.Thread(target=run_streaming_tts, daemon=True)
            tts_thread.start()
            
            # Process audio chunks with interruption checking
            while not interrupt_event.is_set():
                try:
                    chunk_type, chunk_data = audio_chunk_queue.get(timeout=5.0)
                    
                    if chunk_type == 'done':
                        break
                    elif chunk_type == 'error':
                        print(f"TTS: Error in sentence {sentence_id}: {chunk_data}")
                        break
                    elif chunk_type == 'audio':
                        if interrupt_event.is_set():
                            print(f"TTS: Sentence {sentence_id} interrupted during audio processing")
                            break
                        
                        audio_np = np.frombuffer(chunk_data, dtype=np.int16)
                        if len(audio_np) > 0:
                            await self.send_audio_chunks(audio_np, output_queue, interrupt_event)
                    
                except queue.Empty:
                    print(f"TTS: Timeout waiting for audio in sentence {sentence_id}")
                    break
            
            if interrupt_event.is_set():
                print(f"TTS: Sentence {sentence_id} was interrupted")
            else:
                print(f"TTS: Sentence {sentence_id} completed")
                
        except Exception as e:
            print(f"TTS: Error processing sentence {sentence_id}: {e}")


    async def send_audio_chunks(
        self: object, 
        audio_np: np.ndarray, 
        output_queue: asyncio.Queue, 
        interrupt_event: asyncio.Event
    ):
        """Send audio chunks to WebRTC."""
        chunk_size = AUDIO_CHUNK_SIZE
        
        for i in range(0, len(audio_np), chunk_size):
            if interrupt_event.is_set():
                break
            
            chunk = audio_np[i:min(i + chunk_size, len(audio_np))]
            if len(chunk) < chunk_size:
                # Pad last chunk
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
            
            await output_queue.put(chunk)
            await asyncio.sleep(0.001)  # Small delay to prevent overwhelming


    def extract_complete_sentences(self, text: str) -> tuple:
        """Extract complete sentences for processing."""
        
        # Split on sentence endings
        sentences = re.split(r'([.!?。！？ ]+)', text)
        
        complete_sentences = []
        remaining = ""
        
        i = 0
        while i < len(sentences) - 1:
            sentence = sentences[i]
            if i + 1 < len(sentences) and re.match(r'[.!?。！？ ]+', sentences[i + 1]):
                sentence += sentences[i + 1]
                if sentence.strip():
                    complete_sentences.append(sentence.strip())
                i += 2
            else:
                remaining += sentence
                i += 1
        
        # Add any remaining parts
        if i < len(sentences):
            remaining += sentences[i]
        
        return complete_sentences, remaining

class TTSClientFactory:
    @staticmethod
    def create(platform: str, **kwargs) -> TTSClient:
        """
        Create a TTS client instance.
        Args:
            platform (str): The platform name to create the TTS client for.
        Returns:
            TTSClient: The created TTS client.
        """
        try:
            client = register.get_model("tts", platform)
            return client(**kwargs)
        except Exception as e:
            print(f"Error creating TTS client: {e}")
            return None