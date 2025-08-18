
import sounddevice as sd
import numpy as np
import time
import torch

samplerate=16000
chunk = int(samplerate * 32 / 1000)
_time = 3 # seconds

def energy(audio):
    return np.mean(np.square(audio))

print('start_recording')
time.sleep(0.25)
audio = sd.rec(_time * samplerate, samplerate=samplerate, channels=1, dtype='float32')
sd.wait()
sd.play(audio, samplerate=samplerate)
sd.wait()

model, util = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
audio = audio.reshape(1, -1)
print(energy(audio))
torch_audio = torch.from_numpy(audio)
print(torch_audio.shape, torch_audio[:,0:chunk].shape)
#print(np.mean(audio), np.std(audio))

for i in range(0, samplerate, chunk):
    #print(audio[i:i + chunk].tobytes())
    if torch_audio[:,i:i + chunk].shape[1] == 512:
        #print(energy(torch_audio[:,i:i + chunk]))
        is_speech = model(torch_audio[:,i:i + chunk], samplerate).item()
        print(is_speech)