// Global variables
let pc = null;
let localStream = null;
let localAudioTrack = null;
let audioContext = null;
let logChannel = null;

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
    // Configure RTCPeerConnection
    const config = {
        iceServers: [
            { urls: 'stun:stun.l.google.com:19302' },
            { urls: 'stun:stun1.l.google.com:19302' }
        ],
        iceCandidatePoolSize: 10
    };
    
    pc = new RTCPeerConnection(config);


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
            if (data.type === 'log') {
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
        console.log("Connection state:", pc.connectionState);
        if (["disconnected", "failed", "closed"].includes(pc.connectionState)) {
            stop();
        }
    };

    pc.ondatachannel = function(event) {
        const channel = event.channel;
        addLogMessage(`Received data channel: ${channel.label}`, 'client');
        
        channel.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'log') {
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

function negotiate() {
    console.log("Starting negotiation...");
    
    return pc.createOffer({
        offerToReceiveAudio: true,  // IMPORTANT: This tells the server we want audio back
        offerToReceiveVideo: false
    }).then((offer) => {
        console.log("Created offer");
        return pc.setLocalDescription(offer);
    }).then(() => {
        console.log("Set local description, waiting for ICE gathering...");
        return new Promise((resolve) => {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                pc.addEventListener('icegatheringstatechange', function checkState() {
                    console.log("ICE gathering state:", pc.iceGatheringState);
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                });
            }
        });
    }).then(() => {
        const offer = pc.localDescription;
        console.log("Sending offer to server");
        
        return fetch('/offer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `${document.getElementById("password").value}`
            },
            body: JSON.stringify({ sdp: offer.sdp, type: offer.type })
        });
    }).then((res) => {
        if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.json();
    }).then((answer) => {
        console.log("Received answer from server");
        return pc.setRemoteDescription(new RTCSessionDescription(answer));
    }).then(() => {
        console.log("Negotiation complete!");
    }).catch((error) => {
        console.error("Negotiation failed:", error);
    });
}

function start() {
    console.log("Starting WebRTC connection...");
    
    // Ensure audio context is created on user interaction
    ensureAudioContext();
    
    if (pc) {
        stop();
    }
    
    pc = createPeerConnection();

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
                console.log("Track added, sender:", sender);
            });
            
            return negotiate();
        })
        .catch((err) => {
            console.error("Failed to access microphone:", err);
            alert("Failed to access microphone: " + err.message);
        });
}

function stop() {
    console.log("Stopping connection...");
    
    if (localStream) {
        localStream.getTracks().forEach(track => {
            track.stop();
            console.log("Stopped track:", track.kind);
        });
        localStream = null;
        localAudioTrack = null;
    }
    
    if (pc) {
        pc.close();
        pc = null;
        console.log("Peer connection closed");
    }
    
    // Clear remote audio
    const remoteAudio = document.getElementById("remoteAudio");
    if (remoteAudio) {
        remoteAudio.srcObject = null;
    }
    
    // Remove any playback messages
    const message = document.getElementById('playback-message');
    if (message) {
        document.body.removeChild(message);
    }
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