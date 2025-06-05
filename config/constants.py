from utils.config_manager import config_manager
from utils.file_utils import read_json

# Get configuration from the config manager
def get_current_config():
    """Get current configuration values"""
    tts_config = config_manager.get_tts_config()
    app_config = config_manager.get_section("app")
    
    return {
        "TEMP_DIR": app_config.get("temp_dir", "temp"),
        "BASE_URL": tts_config.get("base_url", "http://localhost:8880/v1"),
        "API_KEY": tts_config.get("api_key", "not-needed"),
        "MODEL": tts_config.get("model", "kokoro"),
        "API_OUTPUT_FORMAT": app_config.get("api_output_format", "wav"),
        "MAX_PARALLEL_REQUESTS_BATCH_SIZE": tts_config.get("max_parallel_requests", 2),
        "TASKS_FILE": app_config.get("tasks_file", "tasks.json")
    }

# Initialize constants
_config = get_current_config()
TEMP_DIR = _config["TEMP_DIR"]
BASE_URL = _config["BASE_URL"]
API_KEY = _config["API_KEY"]
MODEL = _config["MODEL"]
API_OUTPUT_FORMAT = _config["API_OUTPUT_FORMAT"]
MAX_PARALLEL_REQUESTS_BATCH_SIZE = _config["MAX_PARALLEL_REQUESTS_BATCH_SIZE"]
TASKS_FILE = _config["TASKS_FILE"]

# Voice map depends on current model
VOICE_MAP = (
    read_json("static_files/kokoro_voice_map_male_narrator.json")
    if MODEL == "kokoro"
    else read_json("static_files/orpheus_voice_map_male_narrator.json")
)

CHAPTER_LIST_FILE = "chapter_list.txt"
FFMPEG_METADATA_FILE = "ffmpeg_metadata.txt"

def reload_constants():
    """Reload constants from updated configuration"""
    global TEMP_DIR, BASE_URL, API_KEY, MODEL, API_OUTPUT_FORMAT, MAX_PARALLEL_REQUESTS_BATCH_SIZE, TASKS_FILE, VOICE_MAP
    
    _config = get_current_config()
    TEMP_DIR = _config["TEMP_DIR"]
    BASE_URL = _config["BASE_URL"]
    API_KEY = _config["API_KEY"]
    MODEL = _config["MODEL"]
    API_OUTPUT_FORMAT = _config["API_OUTPUT_FORMAT"]
    MAX_PARALLEL_REQUESTS_BATCH_SIZE = _config["MAX_PARALLEL_REQUESTS_BATCH_SIZE"]
    TASKS_FILE = _config["TASKS_FILE"]
    
    # Reload voice map for the new model
    VOICE_MAP = (
        read_json("static_files/kokoro_voice_map_male_narrator.json")
        if MODEL == "kokoro"
        else read_json("static_files/orpheus_voice_map_male_narrator.json")
    )
