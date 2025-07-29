let pc = null;

function createPeerConnection() {
    pc = new RTCPeerConnection();

    pc.onconnectionstatechange = () => {
        console.log("Connection state:", pc.connectionState);
        if (["disconnected", "failed", "closed"].includes(pc.connectionState)) {
            stop();
        }
    };

    return pc;
}

function negotiate() {
    return pc.createOffer().then((offer) => {
        return pc.setLocalDescription(offer);
    }).then(() => {
        return new Promise((resolve) => {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                pc.addEventListener('icegatheringstatechange', function checkState() {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                });
            }
        });
    }).then(() => {
        const offer = pc.localDescription;
        return fetch('/offer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sdp: offer.sdp, type: offer.type })
        });
    }).then((res) => res.json()).then((answer) => {
        return pc.setRemoteDescription(answer);
    }).catch(console.error);
}

function start() {
    pc = createPeerConnection();

    navigator.mediaDevices.getUserMedia({ audio: true, video: false })
        .then((stream) => {
            stream.getTracks().forEach(track => pc.addTrack(track, stream));
            return negotiate();
        })
        .catch((err) => {
            alert("Failed to access microphone: " + err);
        });
}

function stop() {
    if (pc) {
        pc.getSenders().forEach(sender => sender.track.stop());
        pc.close();
        pc = null;
        console.log("Stopped audio stream");
    }
}

