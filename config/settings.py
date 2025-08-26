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
        """
        Get a configuration value by key.
        Args:
            key (str): The key of the configuration value.
            default: The default value to return if the key is not found.
        """
        return self.config.get(key, default)
    
    def get_env(self, key, default=None):
        """
        Get an environment variable value by key.
        Args:
            key (str): The key of the environment variable.
            default: The default value to return if the key is not found.
        """
        return os.getenv(key, default)

    def write_config(self, key, value):
        """
        Write a configuration value by key.
        Args:
            key (str): The key of the configuration value.
            value: The value to set for the key.
        """
        self.config[key] = value
        with open(os.path.join(ROOT, 'config', 'config.yaml'), "w", encoding='utf-8') as f:
            yaml.safe_dump(self.config, f)

# print(Settings().get_config("asr_language", "zh-CN"))  # Example usage to print ASR language setting