import yaml
from dotenv import load_dotenv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Settings:
    def __init__(self):
        load_dotenv()
        self.config = self.load_config()

    def load_config(self):
        with open(os.path.join(ROOT, 'config', 'config.yaml'), "r", encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get_config(self, key, default=None):
        return self.config.get(key, default)
    
    def get_env(self, key, default=None):
        return os.getenv(key, default)

# print(Settings().get_config("asr_language", "zh-CN"))  # Example usage to print ASR language setting