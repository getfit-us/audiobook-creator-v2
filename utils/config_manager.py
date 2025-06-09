"""
Configuration Manager for Audiobook Creator
Loads settings from environment variables initially, saves to JSON, and allows UI overrides
"""

import os
import json
from typing import Dict, Any,  TypedDict

# Define types for configuration sections
class TTSConfig(TypedDict):
    base_url: str
    api_key: str
    model: str
    max_parallel_requests: int

class LLMConfig(TypedDict):
    base_url: str
    api_key: str
    model_name: str
    no_think_mode: bool

class AppConfig(TypedDict):
    temp_dir: str
    api_output_format: str
    tasks_file: str

class Config(TypedDict):
    tts: TTSConfig
    llm: LLMConfig
    app: AppConfig
from typing import cast
from dotenv import load_dotenv

load_dotenv()

class ConfigManager:
    def __init__(self, config_file: str = "app_config.json"):
        self.config_file: str = config_file
        self.config: Config = {}  # type: ignore
        self.load_config()

    def get_default_config(self) -> Config:
        """Get default configuration from environment variables"""
        return {
            # TTS Settings
            "tts": {
                "base_url": os.environ.get("TTS_BASE_URL", "http://localhost:8880/v1"),
                "api_key": os.environ.get("TTS_API_KEY", "not-needed"),
                "model": os.environ.get("TTS_MODEL", "kokoro"),
                "max_parallel_requests": int(os.environ.get("MAX_PARALLEL_REQUESTS_BATCH_SIZE", "2"))
            },
            # LLM Settings  
            "llm": {
                "base_url": os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1"),
                "api_key": os.environ.get("LLM_API_KEY", "lm-studio"),
                "model_name": os.environ.get("LLM_MODEL_NAME", "qwen3-14b"),
                "no_think_mode": os.environ.get("NO_THINK_MODE", "true").lower() == "true"
            },
            # App Settings
            "app": {
                "temp_dir": "temp",
                "api_output_format": "wav",
                "tasks_file": "tasks.json"
            }
        }

    def load_config(self) -> None:
        """Load configuration from file, or create with defaults from environment"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                # Ensure all default keys exist (in case new settings were added)
                default_config = self.get_default_config()
                self._merge_defaults(self.config, default_config)
            else:
                # First run - create config from environment variables
                self.config = self.get_default_config()
                self.save_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            # Fallback to defaults
            self.config = self.get_default_config()

    def _merge_defaults(self, config: Dict, defaults: Dict) -> None:
        """Recursively merge default values for missing keys"""
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict) and isinstance(config[key], dict):
                self._merge_defaults(config[key], value)

    def save_config(self) -> None:
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self.config.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        """Set a configuration value"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value

    def update_section(self, section: str, values: Dict[str, Any]) -> None:
        """Update multiple values in a section"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section].update(values)
        self.save_config()

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get all values from a section"""
        return cast(Dict[str, Any], self.config.get(section, {}))

    # Convenience methods for common settings
    def get_tts_config(self) -> TTSConfig:
        """Get TTS configuration"""
        return cast(TTSConfig, self.get_section("tts"))

    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration"""
        return cast(LLMConfig, self.get_section("llm"))

    def update_tts_config(self, base_url: str, api_key: str, model: str, max_parallel: int) -> None:
        """Update TTS configuration"""
        self.update_section("tts", {
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
            "max_parallel_requests": max_parallel
        })

    def update_llm_config(self, base_url: str, api_key: str, model_name: str, no_think_mode: bool) -> None:
        """Update LLM configuration"""
        self.update_section("llm", {
            "base_url": base_url,
            "api_key": api_key,
            "model_name": model_name,
            "no_think_mode": no_think_mode
        })

# Global instance
config_manager = ConfigManager()