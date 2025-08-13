import yaml
from dotenv import load_dotenv
import os

class Settings:
    def __init__(self):
        load_dotenv()
        self.config = self.load_config()

    def load_config(self):
        with open("config.yaml", "r", encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get_config(self, key, default=None):
        return self.config.get(key, default)
    
    def get_env(self, key, default=None):
        return os.getenv(key, default)