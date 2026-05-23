let pc = null;
let ms = null;
let dc = null;

async function start_session() {
  console.log("Starting OpenAI Realtime session...");
  // Get an ephemeral key from your server - see server code below
  const tokenResponse = await fetch("/session");
  const data = await tokenResponse.json();
  const EPHEMERAL_KEY = data.client_secret.value;

  // Create a peer connection
  if (!pc) {
    pc = new RTCPeerConnection();
  }

  // Set up to play remote audio from the model
  const audioEl = document.createElement("audio");
  audioEl.autoplay = true;
  pc.ontrack = e => audioEl.srcObject = e.streams[0];

  // Add local audio track for microphone input in the browser
  ms = await navigator.mediaDevices.getUserMedia({
    audio: true
  });
  pc.addTrack(ms.getTracks()[0]);

  // Set up data channel for sending and receiving events
  dc = pc.createDataChannel("oai-events");
  dc.addEventListener("message", (e) => {
    // Realtime server events appear here!
    console.log(e);
  });

  // Start the session using the Session Description Protocol (SDP)
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  const baseUrl = "https://api.openai.com/v1/audio/transcriptions";
  const model = "gpt-4o-mini-transcribe";
  const sdpResponse = await fetch(`${baseUrl}?model=${model}`, {
    method: "POST",
    body: offer.sdp,
    headers: {
      Authorization: `Bearer ${EPHEMERAL_KEY}`,
    },
    files: {
      audio: ms.getTracks()[0].mediaStreamTrack,
    },
  });

  const answer = {
    type: "answer",
    sdp: await sdpResponse.text(),
  };
  await pc.setRemoteDescription(answer);
}

function stop_session() {
    if (ms) {
        ms.getTracks().forEach(track => track.stop());
        ms = null;
    }
    if (pc) {
        pc.close();
        pc = null;
    }
    if (dc) {
        dc.close();
        dc = null;
    }
}