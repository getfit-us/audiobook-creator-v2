import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from config.constants import TEMP_DIR, MODEL


@dataclass
class VoiceConfig:
    """Configuration for narrator and dialogue voices"""
    narrator_voice: str
    dialogue_voice: str


# Centralized voice mappings
VOICE_MAPPINGS = {
    "kokoro": {
        "male": VoiceConfig(narrator_voice="am_puck", dialogue_voice="af_alloy"),
        "female": VoiceConfig(narrator_voice="af_heart", dialogue_voice="af_sky")
    },
    "orpheus": {
        "male": VoiceConfig(narrator_voice="leo", dialogue_voice="dan"),
        "female": VoiceConfig(narrator_voice="tara", dialogue_voice="leah")
    }
}


def get_voice_config(model: str, gender: str) -> VoiceConfig:
    """
    Get voice configuration for the given model and gender.
    
    Args:
        model: The TTS model ("kokoro" or "orpheus")
        gender: The narrator gender ("male" or "female")
        
    Returns:
        VoiceConfig: Configuration with narrator and dialogue voices
        
    Raises:
        ValueError: If model or gender is not supported
    """
    if model not in VOICE_MAPPINGS:
        raise ValueError(f"Unsupported model: {model}. Supported models: {list(VOICE_MAPPINGS.keys())}")
    
    if gender not in VOICE_MAPPINGS[model]:
        raise ValueError(f"Unsupported gender: {gender}. Supported genders: {list(VOICE_MAPPINGS[model].keys())}")
    
    return VOICE_MAPPINGS[model][gender]


def select_narrator_voice(model: str, gender: str) -> str:
    """
    Select narrator voice for the given model and gender.
    
    Args:
        model: The TTS model ("kokoro" or "orpheus") 
        gender: The narrator gender ("male" or "female")
        
    Returns:
        str: The narrator voice identifier
    """
    config = get_voice_config(model, gender)
    return config.narrator_voice


def select_voice(gender: str, model: str, type: str, book_title: str) -> Dict[str, str]:
    """
    Returns the narrator and dialogue voices for the given parameters.
    
    Args:
        gender: The narrator gender ("male" or "female")
        model: The TTS model ("kokoro" or "orpheus")
        type: The generation type ("single_voice" or "multi_voice")
        book_title: The book title (used for multi-voice file paths)
        
    Returns:
        Dict containing voice configuration:
        - For single_voice: {"narrator_voice": str, "dialogue_voice": str}
        - For multi_voice: {"narrator_voice": str, "voice_map_path": str, "character_map_path": str}
        
    Raises:
        ValueError: If parameters are invalid
        FileNotFoundError: If required files for multi_voice don't exist
    """
    
    if type == "single_voice":
        config = get_voice_config(model, gender)
        return {
            "narrator_voice": config.narrator_voice,
            "dialogue_voice": config.dialogue_voice
        }
    
    elif type == "multi_voice":
        # For multi-voice, we need to return file paths and narrator voice
        speaker_file_path = os.path.join(
            TEMP_DIR, book_title, "speaker_attributed_book.jsonl"
        )
        character_map_file_path = os.path.join(
            TEMP_DIR, book_title, "character_gender_map.json"
        )
        
        # Validate that required files exist
        if not os.path.exists(speaker_file_path):
            raise FileNotFoundError(f"Speaker attribution file not found: {speaker_file_path}")
        
        if not os.path.exists(character_map_file_path):
            raise FileNotFoundError(f"Character gender map file not found: {character_map_file_path}")
        
        # Determine voice map file based on model and gender
        if model == "kokoro":
            voice_map_filename = f"kokoro_voice_map_{gender}_narrator.json"
        else:  # orpheus
            voice_map_filename = f"orpheus_voice_map_{gender}_narrator.json"
        
        voice_map_path = f"static_files/{voice_map_filename}"
        
        config = get_voice_config(model, gender)
        
        return {
            "narrator_voice": config.narrator_voice,
            "speaker_file_path": speaker_file_path,
            "character_map_path": character_map_file_path,
            "voice_map_path": voice_map_path
        }
    
    else:
        raise ValueError(f"Unsupported type: {type}. Supported types: 'single_voice', 'multi_voice'")


def get_available_voices(model: str) -> Dict[str, Dict[str, str]]:
    """
    Get all available voices for a given model.
    
    Args:
        model: The TTS model ("kokoro" or "orpheus")
        
    Returns:
        Dict with gender as key and voice config as value
    """
    if model not in VOICE_MAPPINGS:
        raise ValueError(f"Unsupported model: {model}")
    
    return {
        gender: {
            "narrator_voice": config.narrator_voice,
            "dialogue_voice": config.dialogue_voice
        }
        for gender, config in VOICE_MAPPINGS[model].items()
    }


def validate_voice_selection(model: str, gender: str, voice_type: str) -> bool:
    """
    Validate if the voice selection parameters are valid.
    
    Args:
        model: The TTS model
        gender: The narrator gender
        voice_type: The voice type ("single_voice" or "multi_voice")
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        if voice_type not in ["single_voice", "multi_voice"]:
            return False
        get_voice_config(model, gender)
        return True
    except ValueError:
        return False
