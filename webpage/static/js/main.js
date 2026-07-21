// Global variables
let pc = null;
let localStream = null;
let localAudioTrack = null;
let audioSender = null;
let audioContext = null;
let logChannel = null;
let silentTrack = null;
let silentAudioSource = null;
let connectWatchdogId = null;
let offerAbortController = null;

// Overall budget from clicking Start to reaching connectionState 'connected'. Covers a hung
// /offer request, a stuck ICE gathering step, or ICE connectivity checks that never succeed -
// the browser's own 'failed' detection can take ~30s, which is too long to leave someone
// staring at "Connecting...".
const CONNECT_TIMEOUT_MS = 15000;

function clearConnectWatchdog() {
    if (connectWatchdogId) {
        clearTimeout(connectWatchdogId);
        connectWatchdogId = null;
    }
}

// Build a MediaStreamTrack that continuously emits very low-amplitude noise (not pure
// silence) instead of relying on the mic track's own `enabled = false`. Disabling a real
// track only signals "send silence" — the browser is free to throttle or stop actually
// transmitting it (e.g. via Opus DTX), which can let NAT/relay bindings on the media path
// go idle and drop. A synthetic track that's always genuinely producing audio keeps real
// packets flowing the whole time we're "paused," independent of whatever the browser
// decides to do with a disabled capture track. Low-level noise (not exact zeros) is used
// so the encoder has no grounds to invoke DTX on this track either.
function getSilentTrack() {
    if (silentTrack) return silentTrack;

    const ctx = ensureAudioContext();
    const buffer = ctx.createBuffer(1, ctx.sampleRate, ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < data.length; i++) {
        data[i] = (Math.random() * 2 - 1) * 0.0005; // inaudible, but non-zero
    }

    silentAudioSource = ctx.createBufferSource();
    silentAudioSource.buffer = buffer;
    silentAudioSource.loop = true;

    const destination = ctx.createMediaStreamDestination();
    silentAudioSource.connect(destination);
    silentAudioSource.start();

    silentTrack = destination.stream.getAudioTracks()[0];
    return silentTrack;
}

function stopSilentTrack() {
    if (silentAudioSource) {
        try {
            silentAudioSource.stop();
        } catch (e) {
            // already stopped
        }
        silentAudioSource.disconnect();
        silentAudioSource = null;
    }
    if (silentTrack) {
        silentTrack.stop();
        silentTrack = null;
    }
}

// True when the box is already scrolled at (or near) its bottom edge.
// Threshold is generous because the browser's native smooth-scroll doesn't always land
// exactly at its target (observed ~70px short in testing), and a tight threshold would
// make the very next message think the user had scrolled away, breaking auto-follow.
function isScrolledToBottom(box, threshold = 120) {
    return box.scrollHeight - box.scrollTop - box.clientHeight <= threshold;
}

// Append a chat bubble to the transcription box: user on the right, AI on the left.
function appendToTranscription(text, role) {
    const box = document.getElementById('transcriptionBox');
    if (!box) return;

    // Boxes start hidden until the first transcript actually arrives.
    document.getElementById('transcriptionContainer')?.classList.remove('hidden');
    document.getElementById('clearBtnContainer')?.classList.remove('hidden');

    // Only auto-follow new messages if the user was already at the bottom —
    // otherwise a live conversation keeps yanking them back down while they read up.
    const wasAtBottom = isScrolledToBottom(box);

    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble ' + (role === 'user' ? 'user' : 'ai');

    const roleLabel = document.createElement('span');
    roleLabel.className = 'chat-role';
    roleLabel.textContent = role === 'user' ? 'You' : 'AI';

    const body = document.createElement('span');
    body.textContent = text;

    bubble.appendChild(roleLabel);
    bubble.appendChild(body);
    box.appendChild(bubble);

    if (wasAtBottom) {
        box.scrollTo({ top: box.scrollHeight, behavior: 'smooth' });
    }
}

// Toggle the connecting spinner/label on the Start button while the initial
// handshake (getUserMedia -> offer/answer -> ICE) is in flight.
function setConnecting(isConnecting) {
    const button = document.getElementById('startButton');
    const label = document.getElementById('startButtonLabel');
    if (!button) return;
    button.classList.toggle('connecting', isConnecting);
    button.disabled = isConnecting;
    if (label) label.textContent = isConnecting ? 'Connecting...' : 'Start';
}

// Toggle the persistent recording indicator (pulsing dot + label). On whenever the mic's
// real audio is actually being sent to the server — after connect, after resuming from a
// pause — off whenever it won't be, e.g. paused or disconnected.
function setRecordingIndicator(isRecording) {
    const status = document.getElementById('recordingStatus');
    if (!status) return;
    status.classList.toggle('active', isRecording);
}

// Add log message to the frontend
function addLogMessage(message, type = 'info') {
    const logContainer = document.getElementById('logContainer');
    if (!logContainer) return;
    
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry log-${type}`;
    
    const timestamp = new Date().toLocaleTimeString();
    logEntry.innerHTML = `<span class="timestamp">[${timestamp}]</span> <span class="message">${message}</span>`;
    
    logContainer.appendChild(logEntry);
    
    // Auto-scroll to bottom
    logContainer.scrollTop = logContainer.scrollHeight;
    
    // Keep only last 100 messages
    while (logContainer.children.length > 100) {
        logContainer.removeChild(logContainer.firstChild);
    }
}

// AudioContext management functions
function ensureAudioContext() {
    if (!audioContext) {
        const AudioCtx = AudioContext || webkitAudioContext;
        audioContext = new AudioCtx();
        console.log("Created new AudioContext, state:", audioContext.state);
    }
    
    if (audioContext.state === 'suspended') {
        audioContext.resume().then(() => {
            console.log("Audio context resumed, state:", audioContext.state);
        }).catch(e => {
            console.error("Failed to resume audio context:", e);
        });
    }
    
    return audioContext;
}

function createPeerConnection() {
    // Local mode (empty ICE servers) is currently disabled; always connect online.
    const config = {
        iceServers: [
            // { urls: 'stun:stun.qq.com:3478' },
            { urls: 'stun:stun.l.google.com:19302' },
            { urls: 'stun:stun1.l.google.com:19302' }
        ],
        iceCandidatePoolSize: 10
    };

    pc = new RTCPeerConnection(config);
    // Captured so late/stale events from a connection we've since replaced or torn down
    // (e.g. a delayed 'closed' event firing after fullStop() already moved on) don't act
    // on whatever pc happens to be current by the time they arrive.
    const thisPc = pc;

    logChannel = pc.createDataChannel('logs', {
        ordered: true
    });

    pc.log_channel = logChannel;

    logChannel.onopen = function() {
        addLogMessage('Data channel opened', 'client');
    };
    
    logChannel.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'transcription') {
                appendToTranscription(data.message, 'user');
            } else if (data.type === 'response') {
                appendToTranscription(data.message, 'ai');
            } else if (data.type === 'stop_word') {
                addLogMessage('Stop word detected — session paused', 'client');
                stop();
            } else if (data.type === 'log') {
                addLogMessage(data.message, 'server');
            }
        } catch (e) {
            addLogMessage('Received: ' + event.data, 'server');
        }
    };
    
    logChannel.onerror = function(error) {
        addLogMessage('Data channel error: ' + error, 'error');
    };
    
    logChannel.onclose = function() {
        addLogMessage('Data channel closed', 'warning');
    };

    pc.onconnectionstatechange = () => {
        if (pc !== thisPc) return; // stale event from a superseded connection
        console.log("Connection state:", pc.connectionState);

        if (pc.connectionState === "failed") {
            // Genuine ICE/DTLS failure - neither STUN nor TURN produced a working path.
            clearConnectWatchdog();
            setConnecting(false);
            addLogMessage('Connection failed — check your network and try again', 'error');
            alert('Connection failed — check your network and try again.');
            fullStop();
        } else if (["disconnected", "closed"].includes(pc.connectionState)) {
            // "closed" is usually our own doing (fullStop() called pc.close()); "disconnected"
            // can be a transient network blip. Neither needs an alert on top of cleanup.
            clearConnectWatchdog();
            setConnecting(false);
            fullStop();
        } else if (pc.connectionState === "connected") {
            clearConnectWatchdog();
            setConnecting(false);
            addLogMessage("WebRTC connection established — you may now speak", 'client');
            setRecordingIndicator(true);
        }
    };

    pc.ondatachannel = function(event) {
        const channel = event.channel;
        addLogMessage(`Received data channel: ${channel.label}`, 'client');
        
        channel.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'transcription') {
                    appendToTranscription(data.message, 'user');
                } else if (data.type === 'response') {
                    appendToTranscription(data.message, 'ai');
                } else if (data.type === 'stop_word') {
                    addLogMessage('Stop word detected — session paused', 'client');
                    stop();
                } else if (data.type === 'log') {
                    addLogMessage(data.message, 'server');
                }
            } catch (e) {
                addLogMessage('Server: ' + event.data, 'server');
            }
        };
    };

    pc.oniceconnectionstatechange = () => {
        console.log("ICE connection state:", pc.iceConnectionState);
    };

    pc.onicegatheringstatechange = () => {
        console.log("ICE gathering state:", pc.iceGatheringState);
    };

    pc.ontrack = (event) => {
        console.log("Received remote track:", event);
        //console.log("Track kind:", event.track.kind, "readyState:", event.track.readyState);
        
        if (event.track.kind === 'audio') {
            let audio = document.getElementById("remoteAudio");
            if (!audio) {
                audio = document.createElement("audio");
                audio.id = "remoteAudio";
                audio.autoplay = true;
                audio.controls = true; // Add controls for debugging
                document.body.appendChild(audio);
            }
            
            const stream = event.streams[0];
            
            // Ensure audio context is running
            const ctx = ensureAudioContext();
            
            audio.srcObject = stream;
            
            // Force audio to load and play
            audio.load();
            audio.play().then(() => {
                console.log("Audio started playing successfully");
            }).catch(e => {
                console.error("Failed to play audio:", e);
                console.log("Audio context state:", ctx.state);
                
                // Try to resume audio context and retry
                if (ctx.state === 'suspended') {
                    ctx.resume().then(() => {
                        console.log("Audio context resumed, retrying play");
                        return audio.play();
                    }).then(() => {
                        console.log("Audio play retry successful");
                    }).catch(retryError => {
                        console.error("Audio play retry failed:", retryError);
                        // Show user message to click
                        showPlaybackMessage();
                    });
                } else {
                    // Show user message to click
                    showPlaybackMessage();
                }
            });
            
            // Add event listeners for debugging
            audio.onloadeddata = () => console.log("Remote audio data loaded");
            audio.oncanplay = () => console.log("Remote audio can start playing");
            audio.onplay = () => console.log("Remote audio started playing");
            audio.onpause = () => console.log("Remote audio paused");
            audio.onended = () => console.log("Remote audio ended");
            audio.onerror = (e) => console.error("Remote audio error:", e);
            audio.onstalled = () => console.log("Remote audio stalled");
            audio.onwaiting = () => console.log("Remote audio waiting");
        }
    };

    return pc;
}

function showPlaybackMessage() {
    // Create a temporary message for user interaction
    const message = document.createElement('div');
    message.id = 'playback-message';
    message.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: #007bff;
        color: white;
        padding: 20px;
        border-radius: 8px;
        z-index: 1000;
        cursor: pointer;
        font-family: Arial, sans-serif;
    `;
    message.innerHTML = 'Click here to enable audio playback';
    
    message.onclick = () => {
        const audio = document.getElementById("remoteAudio");
        if (audio) {
            const ctx = ensureAudioContext();
            ctx.resume().then(() => {
                return audio.play();
            }).then(() => {
                console.log("Audio playback enabled after user interaction");
                document.body.removeChild(message);
            }).catch(e => {
                console.error("Still failed to play audio after user interaction:", e);
            });
        }
    };
    
    document.body.appendChild(message);
    
    // Auto-remove after 10 seconds
    setTimeout(() => {
        if (document.body.contains(message)) {
            document.body.removeChild(message);
        }
    }, 10000);
}

// Full ICE gathering (host + STUN + TURN relay allocation) can take multiple seconds,
// and we don't need every candidate to connect - host/STUN candidates alone are enough
// on most networks. Cap the wait so we send whatever we've got after ICE_GATHERING_TIMEOUT_MS
// rather than blocking the whole handshake on a slow/unreachable TURN allocation.
const ICE_GATHERING_TIMEOUT_MS = 1500;

function waitForIceGathering(pc, timeoutMs) {
    if (pc.iceGatheringState === 'complete') {
        return Promise.resolve();
    }
    return new Promise((resolve) => {
        let settled = false;
        const finish = () => {
            if (settled) return;
            settled = true;
            pc.removeEventListener('icegatheringstatechange', checkState);
            clearTimeout(timer);
            resolve();
        };
        function checkState() {
            console.log("ICE gathering state:", pc.iceGatheringState);
            if (pc.iceGatheringState === 'complete') finish();
        }
        pc.addEventListener('icegatheringstatechange', checkState);
        const timer = setTimeout(() => {
            console.log(`ICE gathering timed out after ${timeoutMs}ms, proceeding with ${pc.iceGatheringState} candidates`);
            finish();
        }, timeoutMs);
    });
}

function negotiate() {
    console.log("Starting negotiation...");

    offerAbortController = new AbortController();
    const signal = offerAbortController.signal;

    return pc.createOffer({
        offerToReceiveAudio: true,  // IMPORTANT: This tells the server we want audio back
        offerToReceiveVideo: false
    }).then((offer) => {
        console.log("Created offer");
        return pc.setLocalDescription(offer);
    }).then(() => {
        console.log("Set local description, waiting for ICE gathering...");
        return waitForIceGathering(pc, ICE_GATHERING_TIMEOUT_MS);
    }).then(() => {
        const offer = pc.localDescription;
        const processingMode = getProcessingMode();
        console.log("Sending offer to server with processing mode:", processingMode);

        return fetch('/offer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
                processingMode: processingMode
            }),
            signal
        });
    }).then((res) => {
        if (!res.ok) {
            return res.json().catch(() => ({})).then((body) => {
                throw new Error(body.message || `HTTP error! status: ${res.status}`);
            });
        }
        return res.json();
    }).then((answer) => {
        console.log("Received answer from server");
        return pc.setRemoteDescription(new RTCSessionDescription(answer));
    }).then(() => {
        console.log("Negotiation complete!");
    }).catch((error) => {
        if (error.name === 'AbortError') {
            // We cancelled this ourselves (watchdog timeout or a fresh start()) -
            // whichever triggered the abort has already shown its own message and cleaned up.
            console.log("Negotiation aborted");
            return;
        }
        console.error("Negotiation failed:", error);
        setConnecting(false);
        addLogMessage('Failed to connect: ' + error.message, 'error');
        alert(error.message);
        fullStop();
    });
}

function start() {
    console.log("Starting WebRTC connection...");

    // Soft restart: PC is still connected, mic was only muted — just unmute.
    if (pc && pc.connectionState === 'connected' && localAudioTrack && !localAudioTrack.enabled) {
        localAudioTrack.enabled = true;
        if (audioSender) {
            audioSender.replaceTrack(localAudioTrack)
                .catch(e => console.error("Failed to restore microphone track:", e));
        }
        console.log("Soft restart: microphone unmuted");
        addLogMessage('Session resumed — you may now speak', 'client');
        setRecordingIndicator(true);
        // Re-attach remote audio stream if needed
        const remoteAudio = document.getElementById("remoteAudio");
        if (remoteAudio && !remoteAudio.srcObject) {
            // The audio player track is still flowing on the server; re-attach
            pc.getReceivers().forEach(receiver => {
                if (receiver.track.kind === 'audio') {
                    const stream = new MediaStream([receiver.track]);
                    remoteAudio.srcObject = stream;
                    remoteAudio.play().catch(e => console.error("Re-attach audio failed:", e));
                }
            });
        }
        return;
    }

    // Get the selected processing mode
    const processingMode = getProcessingMode();
    addLogMessage(`Starting with ${processingMode} processing mode`, 'info');

    setConnecting(true);

    // Ensure audio context is created on user interaction
    ensureAudioContext();

    if (pc) {
        fullStop();
    }

    pc = createPeerConnection();

    // Give the whole handshake a hard deadline instead of leaving the user staring at
    // "Connecting..." until the browser's own (much slower, ~30s) failure detection kicks in.
    clearConnectWatchdog();
    const watchdogPc = pc;
    connectWatchdogId = setTimeout(() => {
        connectWatchdogId = null;
        if (pc === watchdogPc && pc.connectionState !== 'connected') {
            console.warn(`Connection attempt timed out after ${CONNECT_TIMEOUT_MS}ms`);
            addLogMessage('Connection timed out — check your network and try again', 'error');
            alert('Connection timed out — check your network and try again.');
            setConnecting(false);
            fullStop();
        }
    }, CONNECT_TIMEOUT_MS);

    // Audio constraints that match your server
    const constraints = {
        audio: {
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
            sampleRate: 24000,
            channelCount: 1
        },
        video: false
    };

    navigator.mediaDevices.getUserMedia(constraints)
        .then((stream) => {
            console.log("Got user media, tracks:", stream.getTracks().length);
            localStream = stream;
            localAudioTrack = stream.getAudioTracks()[0];
            console.log("Local audio track:", localAudioTrack);
            
            if (localAudioTrack) {
                console.log("Local audio track settings:", localAudioTrack.getSettings());
                
                localAudioTrack.onended = () => {
                    console.log("Local audio track ended");
                };
            }
            
            // Add tracks to peer connection
            stream.getTracks().forEach(track => {
                console.log("Adding track to PC:", track.kind);
                const sender = pc.addTrack(track, stream);
                if (track.kind === 'audio') {
                    audioSender = sender;
                }
                console.log("Track added, sender:", sender);
            });
            
            return negotiate();
        })
        .catch((err) => {
            console.error("Failed to access microphone:", err);
            clearConnectWatchdog();
            alert("Failed to access microphone: " + err.message);
            setConnecting(false);
        });
}

// Hard disconnect: close PC and release all tracks
function fullStop() {
    console.log("Full disconnect...");
    clearConnectWatchdog();
    if (offerAbortController) {
        offerAbortController.abort();
        offerAbortController = null;
    }
    if (localStream) {
        localStream.getTracks().forEach(track => {
            track.stop();
            console.log("Stopped track:", track.kind);
        });
        localStream = null;
        localAudioTrack = null;
    }
    stopSilentTrack();
    audioSender = null;
    setRecordingIndicator(false);
    if (pc) {
        pc.close();
        pc = null;
        logChannel = null;
        console.log("Peer connection closed");
    }
    const remoteAudio = document.getElementById("remoteAudio");
    if (remoteAudio) remoteAudio.srcObject = null;
    const message = document.getElementById('playback-message');
    if (message && document.body.contains(message)) document.body.removeChild(message);
}

function stop() {
    // Soft stop: mute mic and silence remote audio but keep WebRTC connection alive.
    // This avoids ICE renegotiation on the next start().
    if (pc && pc.connectionState === 'connected' && localAudioTrack) {
        console.log("Soft stop: muting microphone");
        localAudioTrack.enabled = false;
        setRecordingIndicator(false);
        if (audioSender) {
            audioSender.replaceTrack(getSilentTrack())
                .catch(e => console.error("Failed to switch to silent track:", e));
        }
        const remoteAudio = document.getElementById("remoteAudio");
        if (remoteAudio) remoteAudio.srcObject = null;
        // Tell server to clear audio buffer so old TTS audio doesn't play on resume
        if (logChannel && logChannel.readyState === 'open') {
            logChannel.send(JSON.stringify({type: 'clear_audio'}));
        }
        addLogMessage('Session paused — microphone muted', 'client');
        return;
    }
    // Fall back to full disconnect if not connected
    fullStop();
}

function clearLogs() {
    const logContainer = document.getElementById('logContainer');
    if (logContainer) {
        logContainer.innerHTML = '';
    }
}

// Initialize audio context on any user interaction
function initAudioOnInteraction() {
    ensureAudioContext();
    console.log("Audio context initialized on user interaction");
}

// Add event listeners for user interaction to initialize audio context
document.addEventListener('click', initAudioOnInteraction, { once: true });
document.addEventListener('keydown', initAudioOnInteraction, { once: true });
document.addEventListener('touchstart', initAudioOnInteraction, { once: true });

// Optional: Add debugging info
function getAudioContextInfo() {
    if (audioContext) {
        console.log("AudioContext state:", audioContext.state);
        console.log("AudioContext sample rate:", audioContext.sampleRate);
        console.log("AudioContext current time:", audioContext.currentTime);
    } else {
        console.log("No AudioContext created yet");
    }
}

// Make debugging function available globally
window.getAudioContextInfo = getAudioContextInfo;

// Function to get the selected processing mode
function getProcessingMode() {
    const selectedOption = document.querySelector('input[name="processingMode"]:checked');
    return selectedOption ? selectedOption.value : 'online'; // default to online; local mode is currently disabled
}

function clearTranscription() {
    const transcriptionBox = document.getElementById('transcriptionBox');
    if (transcriptionBox && transcriptionBox.innerHTML.trim() !== '') {
        if (confirm('Are you sure you want to clear all transcriptions?')) {
            transcriptionBox.innerHTML = '';
            document.getElementById('transcriptionContainer')?.classList.add('hidden');
            document.getElementById('clearBtnContainer')?.classList.add('hidden');
        }
    }
}