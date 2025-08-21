from dataclasses import dataclass

@dataclass
class Register:
    asr_models = {}
    llm_models = {}
    tts_models = {}
    vads = {}

    def print_available_models(self, model_type=None):
        asr_msg = "Available ASR Models:\n"
        for model in self.asr_models:
            asr_msg += f" - {model}\n"
        llm_msg = "Available LLM Models:\n"
        for model in self.llm_models:
            llm_msg += f" - {model}\n"
        tts_msg = "Available TTS Models:\n"
        for model in self.tts_models:
            tts_msg += f" - {model}\n"
        vad_msg = "Available VAD Models:\n"
        for model in self.vads:
            vad_msg += f" - {model}\n"

        if model_type == "asr":
            print(asr_msg)
        elif model_type == "llm":
            print(llm_msg)
        elif model_type == "tts":
            print(tts_msg)
        elif model_type == "vad":
            print(vad_msg)
        else:
            print(asr_msg)
            print(llm_msg)
            print(tts_msg)
            print(vad_msg)
            
    def get_model(self, model_type, model_name):
        try:
            if model_type == "asr":
                return self.asr_models[model_name]
            elif model_type == "llm":
                return self.llm_models[model_name]
            elif model_type == "tts":
                return self.tts_models[model_name]
            elif model_type == "vad":
                return self.vads[model_name]
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        except ValueError as e:
            print(f"Error: {e}")
            return None
        except Exception as e:
            print(f"Error: Invalid model name: {model_name}")
            self.print_available_models(model_type)
            return None

    def add_model(self, model_type, model_name):
        def decorator(model):
            try:
                if model_type == "asr":
                    self.asr_models[model_name] = model
                elif model_type == "llm":
                    self.llm_models[model_name] = model
                elif model_type == "tts":
                    self.tts_models[model_name] = model
                elif model_type == "vad":
                    self.vads[model_name] = model
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
            except ValueError as e:
                print(f"Error: {e}")
                return None
            except Exception as e:
                print(f"Error: {e}")
                return None
            return model
        return decorator

register = Register()