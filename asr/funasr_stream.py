from funasr import AutoModel
import os
import soundfile
import time

class ParaformerStreaming:
    def __init__(self, model, chunk_size=[0, 10, 5], encoder_chunk_look_back=4, 
                 decoder_chunk_look_back=1):
        '''
        Initialize the ParaformerStreaming model.
        Args:
            model (AutoModel): Pre-trained Paraformer model.
            chunk_size (list): List containing [unknown, frame size in seconds, number of looks back for encoder and decoder].
                               Given a conventional sample rate of 16000Hz and 60ms each frame (idk why, seems that paraformer just set it that way),
                               the first element is unknown, you may go to github issue at [https://github.com/modelscope/FunASR/issues/1026] but there is nothing really. 
                               The second element is the frame size, and the third element is the number of looks back for encoder and decoder.
                                1. chunk_size[0]: Unknown, typically set to 0.
                                2. chunk_size[1]: Frame size in seconds (e.g., 10 means 10 * 60ms = 600ms).
                                3. chunk_size[2]: Number of looks back for encoder and decoder (e.g., 5 means looking back 5 frames).
            encoder_chunk_look_back (int): Number of chunks to look back for encoder self-attention.
            decoder_chunk_look_back (int): Number of encoder chunks to look back for decoder cross-attention.
        '''
        self.model = model
        self.chunk_size = chunk_size
        self.encoder_chunk_look_back = encoder_chunk_look_back
        self.decoder_chunk_look_back = decoder_chunk_look_back
        self.cache = {}
        self.chunk_stride = chunk_size[1] * 960  # Assuming 16kHz sample rate, 60ms frame = 960 samples

    def generate(self, speech):
        '''
        Generate transcription from the input speech.
        Args:
            speech (numpy.ndarray): Input audio signal as a numpy array.
        Returns:
            str: Transcription of the input speech.
        '''
        text = ''
        total_chunk_num = int(len((speech)-1)/self.chunk_stride+1)
        for i in range(total_chunk_num):
            speech_chunk = speech[i*self.chunk_stride:(i+1)*self.chunk_stride]
            is_final = i == total_chunk_num - 1
            res = self.model.generate(input=speech_chunk, 
                                      cache=self.cache, 
                                      is_final=is_final, 
                                      chunk_size=self.chunk_size, 
                                      encoder_chunk_look_back=self.encoder_chunk_look_back, 
                                      decoder_chunk_look_back=self.decoder_chunk_look_back)
            text += res[0]['text'] if res else ''
        return text

if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = "model/paraformer-zh-streaming"
    model = AutoModel(model=os.path.join(parent_dir, model_path), disable_update=True)
    paraformer = ParaformerStreaming(model=model)
    wav_file = os.path.join(parent_dir, "audio/long_mandarin.wav")
    speech, sample_rate = soundfile.read(wav_file)
    start_time = time.time()
    transcription = paraformer.generate(speech)
    end_time = time.time()
    print(f"Transcription: {transcription}")

# Given a conventional sample rate of 16000Hz and 60ms each frame (idk why, seems that paraformer just set it that way), 
# The first element in this list is unknown, you may go to github issue at [https://github.com/modelscope/FunASR/issues/1026]
# but there is nothing really.
# The second element is the frame size, so 10 * 60ms = 600ms audio being decoded in one chunk.
# The third element is the number of looks back for the encoder and decoder, meaning that each chunk will look back at the 
# previous and next 5 frames.
# chunk_size = [0, 20, 4] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
# encoder_chunk_look_back = 4 #number of chunks to lookback for encoder self-attention
# decoder_chunk_look_back = 1 #number of encoder chunks to lookback for decoder cross-attention

# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(parent_dir)
# model = AutoModel(model=os.path.join(parent_dir, "model/paraformer-zh-streaming"), disable_update=True)

# wav_file = os.path.join(parent_dir, 'audio/mandarin.wav')
# speech, sample_rate = soundfile.read(wav_file)
# chunk_stride = chunk_size[1] * 960 # With 60ms frame, the stride is 960 samples (assuming 16kHz sample rate)

# cache = {}
# total_chunk_num = int(len((speech)-1)/chunk_stride+1)
# for i in range(total_chunk_num):
#     speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
#     is_final = i == total_chunk_num - 1
#     res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size, encoder_chunk_look_back=encoder_chunk_look_back, decoder_chunk_look_back=decoder_chunk_look_back)
#     print(res)
