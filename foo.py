#!/usr/bin/env python3
"""
Simple WebRTC Audio Demo
Receives audio from client, processes it (adds echo effect), and sends it back
"""

import asyncio
import json
import os
import ssl
import uuid
import numpy as np
import argparse
from fractions import Fraction
import time

from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole
from av import AudioFrame
from av.audio.resampler import AudioResampler

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = "s16"

# Global sets to track peer connections
pcs = set()

class AudioProcessor(MediaStreamTrack):
    """
    Audio track that processes incoming audio and sends it back with effects
    """
    kind = "audio"
    
    def __init__(self, input_track):
        super().__init__()
        self.input_track = input_track
        self.resampler = AudioResampler(rate=SAMPLE_RATE, layout='mono', format=FORMAT)
        self._timestamp = 0
        self.audio_buffer = []
        self.buffer_size = 4800  # 300ms at 16kHz
        
    async def recv(self):
        # Get frame from input track
        frame = await self.input_track.recv()
        
        # Resample to our target format
        print(type(frame), frame)
        resampled_frames = self.resampler.resample(frame)
        if not resampled_frames:
            # If no frames returned, create silence
            return self._create_silence_frame(1600)  # 100ms of silence
            
        audio_frame = resampled_frames[0]
        pcm_data = audio_frame.to_ndarray().flatten()
        
        print(type(pcm_data), pcm_data.shape, pcm_data.dtype, pcm_data[:10])  # Debug output
        
        # Add to buffer for echo effect
        self.audio_buffer.extend(pcm_data)
        
        # Process audio (add simple echo effect)
        processed_audio = self._process_audio(pcm_data)
        
        # Create output frame
        output_frame = self._create_audio_frame(pcm_data)
        
        return output_frame
    
    def _process_audio(self, pcm_data):
        """Simple audio processing: add echo effect and amplify"""
        processed = pcm_data.copy().astype(np.float32)
        
        # Add echo if we have enough buffer
        if len(self.audio_buffer) > self.buffer_size:
            delay_samples = self.buffer_size // 3  # Echo delay
            echo_start = len(self.audio_buffer) - len(pcm_data) - delay_samples
            
            if echo_start >= 0:
                echo_data = np.array(self.audio_buffer[echo_start:echo_start + len(pcm_data)])
                processed += echo_data.astype(np.float32) * 0.3  # 30% echo
        
        # Keep buffer size manageable
        if len(self.audio_buffer) > self.buffer_size * 2:
            self.audio_buffer = self.audio_buffer[-self.buffer_size:]
        
        # Amplify and add some reverb-like effect
        processed *= 1.5  # Amplify
        
        # Simple reverb simulation
        if len(processed) > 100:
            for i in range(100, len(processed)):
                processed[i] += processed[i-100] * 0.1
        
        # Clip to prevent distortion
        processed = np.clip(processed, -32767, 32767)
        
        return processed.astype(np.int16)
    
    def _create_audio_frame(self, pcm_data):
        """Create AudioFrame from PCM data"""
        # Reshape for AudioFrame (channels, samples)
        audio_array = pcm_data.reshape(1, -1)
        
        frame = AudioFrame.from_ndarray(audio_array, format=FORMAT, layout='mono')
        frame.sample_rate = SAMPLE_RATE
        frame.time_base = Fraction(1, SAMPLE_RATE)
        frame.pts = self._timestamp
        
        self._timestamp += len(pcm_data)
        
        return frame
    
    def _create_silence_frame(self, samples):
        """Create a frame of silence"""
        silence = np.zeros((1, samples), dtype=np.int16)
        frame = AudioFrame.from_ndarray(silence, format=FORMAT, layout='mono')
        frame.sample_rate = SAMPLE_RATE
        frame.time_base = Fraction(1, SAMPLE_RATE)
        frame.pts = self._timestamp
        self._timestamp += samples
        return frame

class TestToneGenerator(MediaStreamTrack):
    """
    Generate a test tone for testing without input audio
    """
    kind = "audio"
    
    def __init__(self, frequency=440, sample_rate=16000):
        super().__init__()
        self.frequency = frequency
        self.sample_rate = sample_rate
        self._timestamp = 0
        self.samples_per_frame = sample_rate // 10  # 100ms frames
        
    async def recv(self):
        # Generate sine wave
        duration = self.samples_per_frame / self.sample_rate
        t = np.linspace(0, duration, self.samples_per_frame, False)
        
        # Create a sine wave that changes frequency over time for interest
        freq_mod = self.frequency + 50 * np.sin(self._timestamp / self.sample_rate * 0.5)
        sine_wave = np.sin(2 * np.pi * freq_mod * t)
        print(type(sine_wave), sine_wave.shape, sine_wave.dtype, sine_wave[:10])  # Debug output
        # Convert to 16-bit PCM
        pcm_data = (sine_wave * 16383).astype(np.int16)  # 50% volume
        print(type(pcm_data), pcm_data.shape, pcm_data.dtype, pcm_data[:10])  # Debug output
        audio_array = pcm_data.reshape(1, -1)
        
        # Create AudioFrame
        frame = AudioFrame.from_ndarray(audio_array, format='s16', layout='mono')
        frame.sample_rate = self.sample_rate
        frame.time_base = Fraction(1, self.sample_rate)
        frame.pts = self._timestamp
        self._timestamp += self.samples_per_frame
        print(type(frame), frame.sample_rate, frame.time_base, frame.pts, frame)  # Debug output
        sd.play(frame.to_ndarray().flatten(), samplerate=frame.sample_rate)
        return frame

async def index(request):
    """Serve the HTML page"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>WebRTC Audio Demo</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; background: #f0f0f0; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        button { padding: 10px 20px; margin: 10px; font-size: 16px; border: none; border-radius: 4px; cursor: pointer; }
        .start { background: #4CAF50; color: white; }
        .stop { background: #f44336; color: white; }
        .test { background: #2196F3; color: white; }
        .status { padding: 10px; margin: 10px 0; border-radius: 4px; }
        .info { background: #e3f2fd; border: 1px solid #2196F3; }
        .success { background: #e8f5e9; border: 1px solid #4CAF50; }
        .error { background: #ffebee; border: 1px solid #f44336; }
        audio { width: 100%; margin: 10px 0; }
        .controls { text-align: center; margin: 20px 0; }
        #log { background: #f5f5f5; padding: 10px; border-radius: 4px; height: 300px; overflow-y: scroll; font-family: monospace; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>WebRTC Audio Processing Demo</h1>
        
        <div class="status info">
            <strong>Instructions:</strong>
            <ol>
                <li>Click "Start Audio Processing" to begin with microphone input (will add echo effect)</li>
                <li>Or click "Test Tone" to hear a generated test tone</li>
                <li>The processed audio will play in the audio element below</li>
                <li>Click "Stop" to end the session</li>
            </ol>
        </div>
        
        <div class="controls">
            <button class="start" onclick="start()">Start Audio Processing</button>
            <button class="test" onclick="startTestTone()">Test Tone</button>
            <button class="stop" onclick="stop()">Stop</button>
        </div>
        
        <div id="status" class="status info">Ready to start...</div>
        
        <h3>Processed Audio Output:</h3>
        <audio id="remoteAudio" controls></audio>
        
        <h3>Connection Log:</h3>
        <div id="log"></div>
    </div>

    <script>
        let pc = null;
        let localStream = null;
        let audioContext = null;
        let testToneMode = false;

        function log(message) {
            const logDiv = document.getElementById('log');
            const timestamp = new Date().toLocaleTimeString();
            logDiv.innerHTML += `[${timestamp}] ${message}<br>`;
            logDiv.scrollTop = logDiv.scrollHeight;
            console.log(message);
        }

        function updateStatus(message, type = 'info') {
            const statusDiv = document.getElementById('status');
            statusDiv.className = `status ${type}`;
            statusDiv.textContent = message;
            log(`Status: ${message}`);
        }

        function ensureAudioContext() {
            if (!audioContext) {
                const AudioCtx = AudioContext || webkitAudioContext;
                audioContext = new AudioCtx();
                log('Created AudioContext');
            }
            
            if (audioContext.state === 'suspended') {
                audioContext.resume().then(() => {
                    log('AudioContext resumed');
                });
            }
        }

        function createPeerConnection() {
            const config = { iceServers: [] };
            pc = new RTCPeerConnection(config);

            pc.onconnectionstatechange = () => {
                log(`Connection state: ${pc.connectionState}`);
                if (pc.connectionState === 'connected') {
                    updateStatus('Connected - Audio processing active', 'success');
                } else if (['disconnected', 'failed', 'closed'].includes(pc.connectionState)) {
                    updateStatus('Disconnected', 'error');
                    stop();
                }
            };

            pc.oniceconnectionstatechange = () => {
                log(`ICE state: ${pc.iceConnectionState}`);
            };

            pc.ontrack = (event) => {
                log(`Received ${event.track.kind} track`);
                
                if (event.track.kind === 'audio') {
                    const audio = document.getElementById('remoteAudio');
                    const stream = event.streams[0];
                    
                    ensureAudioContext();
                    
                    audio.srcObject = stream;
                    audio.play().then(() => {
                        log('Audio playback started');
                        updateStatus('Playing processed audio', 'success');
                    }).catch(e => {
                        log(`Audio play failed: ${e.message}`);
                        updateStatus('Click anywhere to enable audio', 'error');
                        
                        // Try again on user interaction
                        document.addEventListener('click', () => {
                            audio.play().then(() => {
                                log('Audio playback started after user interaction');
                                updateStatus('Playing processed audio', 'success');
                            }).catch(console.error);
                        }, { once: true });
                    });
                }
            };

            return pc;
        }

        async function negotiate(includeAudio = true) {
            log('Starting negotiation...');
            
            const offer = await pc.createOffer({
                offerToReceiveAudio: true,
                offerToReceiveVideo: false
            });
            
            await pc.setLocalDescription(offer);
            log('Set local description');
            
            // Wait for ICE gathering
            if (pc.iceGatheringState !== 'complete') {
                await new Promise(resolve => {
                    pc.addEventListener('icegatheringstatechange', function checkState() {
                        if (pc.iceGatheringState === 'complete') {
                            pc.removeEventListener('icegatheringstatechange', checkState);
                            resolve();
                        }
                    });
                });
            }
            
            log('Sending offer to server');
            const response = await fetch('/offer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    sdp: pc.localDescription.sdp, 
                    type: pc.localDescription.type,
                    testTone: testToneMode
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const answer = await response.json();
            await pc.setRemoteDescription(new RTCSessionDescription(answer));
            log('Negotiation complete');
        }

        async function start() {
            try {
                updateStatus('Starting...', 'info');
                testToneMode = false;
                
                ensureAudioContext();
                
                if (pc) stop();
                pc = createPeerConnection();
                
                log('Requesting microphone access...');
                localStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: false,
                        noiseSuppression: false,
                        autoGainControl: false,
                        sampleRate: 16000,
                        channelCount: 1
                    }
                });
                
                log('Got microphone access');
                localStream.getTracks().forEach(track => {
                    pc.addTrack(track, localStream);
                    log(`Added ${track.kind} track`);
                });
                
                await negotiate();
                updateStatus('Processing microphone audio with echo effect', 'success');
                
            } catch (error) {
                log(`Error: ${error.message}`);
                updateStatus(`Error: ${error.message}`, 'error');
            }
        }

        async function startTestTone() {
            try {
                updateStatus('Starting test tone...', 'info');
                testToneMode = true;
                
                ensureAudioContext();
                
                if (pc) stop();
                pc = createPeerConnection();
                
                await negotiate();
                updateStatus('Playing test tone', 'success');
                
            } catch (error) {
                log(`Error: ${error.message}`);
                updateStatus(`Error: ${error.message}`, 'error');
            }
        }

        function stop() {
            log('Stopping...');
            
            if (localStream) {
                localStream.getTracks().forEach(track => {
                    track.stop();
                    log(`Stopped ${track.kind} track`);
                });
                localStream = null;
            }
            
            if (pc) {
                pc.close();
                pc = null;
                log('Closed peer connection');
            }
            
            const audio = document.getElementById('remoteAudio');
            if (audio) {
                audio.srcObject = null;
            }
            
            updateStatus('Stopped', 'info');
        }

        // Initialize on user interaction
        document.addEventListener('click', ensureAudioContext, { once: true });
        
        log('Demo loaded and ready');
    </script>
</body>
</html>
    """
    return web.Response(content_type="text/html", text=html_content)

async def offer(request):
    """Handle WebRTC offer"""
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    test_tone_mode = params.get("testTone", False)

    pc = RTCPeerConnection()
    pc_id = f"PeerConnection({uuid.uuid4()})"
    pcs.add(pc)

    print(f"{pc_id} Created for {request.remote}")
    
    recorder = MediaBlackhole()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"{pc_id} Connection state: {pc.connectionState}")
        if pc.connectionState in ['closed', 'failed', 'disconnected']:
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    async def on_track(track):
        print(f"{pc_id} Track {track.kind} received")
        
        if track.kind == "audio":
            # Process the incoming audio and add it back as a track
            processed_track = AudioProcessor(track)
            pc.addTrack(processed_track)
            recorder.addTrack(track)

        @track.on("ended")
        async def on_ended():
            print(f"{pc_id} Track {track.kind} ended")
            await recorder.stop()

    # If test tone mode, add a test tone generator
    if test_tone_mode:
        print(f"{pc_id} Adding test tone generator")
        test_tone = TestToneGenerator(frequency=440)
        pc.addTrack(test_tone)

    await pc.setRemoteDescription(offer)
    await recorder.start()

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
    )

async def on_shutdown(app):
    """Clean up peer connections on shutdown"""
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC Audio Processing Demo")
    parser.add_argument("--host", default="localhost", help="Host (default: localhost)")
    parser.add_argument("--port", type=int, default=8081, help="Port (default: 8081)")
    parser.add_argument("--cert-file", help="SSL certificate file")
    parser.add_argument("--key-file", help="SSL key file")
    args = parser.parse_args()

    # SSL context for HTTPS (required for microphone access in production)
    ssl_context = None
    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)

    print(f"Starting WebRTC Audio Demo server...")
    print(f"Open your browser to: {'https' if ssl_context else 'http'}://{args.host}:{args.port}")
    print("Note: For microphone access, you may need HTTPS in production")

    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)