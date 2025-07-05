"""
Configuration Module

This module contains all configuration constants and settings for the
Chatterbox Web UI application.
"""

import os

# File upload settings
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'flac', 'opus', 'm4a', 'ogg'}
ALLOWED_TEXT_EXTENSIONS = {'txt'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Directory paths
UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = os.path.join('static', 'output')
JSON_FOLDER = os.path.join('static', 'json')

# Default values - optimized for balanced voice consistency
DEFAULT_EXAGGERATION = 0.5  # Restored to original balanced value
DEFAULT_TEMPERATURE = 0.3   # Restored to working value - too low breaks voice matching
DEFAULT_CFG_WEIGHT = 0.7    # Restored to working value
DEFAULT_CHUNK_SIZE = 150   # Increased from 120 for better natural boundaries
DEFAULT_SPEED = 1.0
DEFAULT_PITCH = 0
DEFAULT_SEED = 0

# Voice similarity validation settings (improved for consistency)  
DEFAULT_VOICE_SIMILARITY_THRESHOLD = 0.78  # Restored working threshold
VOICE_SIMILARITY_THRESHOLD_RANGE = (0.0, 1.0)  # 0.0 = disabled, 1.0 = maximum strictness

# Enhanced chunking settings inspired by audiobook processing
AUDIOBOOK_MODE_CHUNK_SIZE = 200  # Larger chunks for audiobook-style processing
RETURN_PAUSE_MULTIPLIER = 1.2    # Adjust automatic pause durations
VOICE_CONSISTENCY_THRESHOLD = 5  # Force sequential processing above this many chunks

# Parameter ranges
EXAGGERATION_RANGE = (0.25, 2.0)
TEMPERATURE_RANGE = (0.05, 5.0)
CFG_WEIGHT_RANGE = (0.0, 1.0)
CHUNK_SIZE_RANGE = (50, 300)
CANDIDATES_RANGE = (1, 10)
ATTEMPTS_RANGE = (1, 10)
WORKERS_RANGE = (1, 8)

# Environment variables with defaults
DEFAULT_MAX_TEXT_LENGTH = 10000
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 5000
DEFAULT_DEBUG = False

# Whisper model configuration
WHISPER_MODELS = {
    'tiny': 'Whisper Tiny (~1 GB VRAM OpenAI / ~0.5 GB faster-whisper)',
    'base': 'Whisper Base (~1.2–2 GB OpenAI / ~0.7–1 GB faster-whisper)',
    'small': 'Whisper Small (~2–3 GB OpenAI / ~1.2–1.7 GB faster-whisper)',
    'medium': 'Whisper Medium (~5–8 GB OpenAI / ~2.5–4.5 GB faster-whisper)',
    'large': 'Whisper Large (~10–13 GB OpenAI / ~4.5–6.5 GB faster-whisper)'
}

# Export formats
EXPORT_FORMATS = ['wav', 'mp3', 'flac', 'ogg']

# Audio processing defaults
DEFAULT_NOISE_REDUCTION = False
DEFAULT_REMOVE_SILENCE = False
DEFAULT_NORMALIZE_AUDIO = False
DEFAULT_USE_AUTO_EDITOR = False

# Auto-editor defaults
DEFAULT_AE_THRESHOLD = 0.06
DEFAULT_AE_MARGIN = 0.2

# FFmpeg normalization defaults
DEFAULT_NORMALIZE_METHOD = 'ebu'
DEFAULT_INTEGRATED_LOUDNESS = -24
DEFAULT_TRUE_PEAK = -2
DEFAULT_LOUDNESS_RANGE = 7

def get_env_var(name, default, var_type=str):
    """Get environment variable with type conversion and default value"""
    value = os.environ.get(name, default)
    if var_type == bool:
        return str(value).lower() in ('true', '1', 'yes', 'on')
    elif var_type == int:
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    elif var_type == float:
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    return value

# Environment-based configuration
MAX_TEXT_LENGTH = get_env_var('MAX_TEXT_LENGTH', DEFAULT_MAX_TEXT_LENGTH, int)
HOST = get_env_var('HOST', DEFAULT_HOST)
PORT = get_env_var('PORT', DEFAULT_PORT, int)
DEBUG = get_env_var('DEBUG', DEFAULT_DEBUG, bool)
SECRET_KEY = os.environ.get('SECRET_KEY', os.urandom(24).hex())

# Validation functions
def validate_parameter_range(value, param_range, param_name):
    """Validate that a parameter is within the allowed range"""
    min_val, max_val = param_range
    if not (min_val <= value <= max_val):
        raise ValueError(f'{param_name} must be between {min_val} and {max_val}')
    return True

def get_allowed_extensions():
    """Get all allowed file extensions"""
    return ALLOWED_AUDIO_EXTENSIONS | ALLOWED_TEXT_EXTENSIONS

def is_audio_file(filename):
    """Check if file is an allowed audio format"""
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_AUDIO_EXTENSIONS

def is_text_file(filename):
    """Check if file is an allowed text format"""
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_TEXT_EXTENSIONS