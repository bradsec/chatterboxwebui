import os

# Disable automatic downloads from Hugging Face Hub
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import nltk
import numpy as np
import re
import uuid
import librosa
from librosa import effects
import noisereduce as nr
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import collections
import contextlib
import sys
import wave
import webrtcvad
import random
from scipy import signal
import warnings
import logging
import gc
from typing import Optional, List, Tuple, Callable
import difflib
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import json

# Configure logging FIRST before any imports that might use logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import modularized audio components
from audio_effects import detect_and_fix_audio_artifacts, analyze_audio_quality
from audio_transforms import adjust_speed_ffmpeg, adjust_pitch_ffmpeg, resample_audio
from audio_processing import normalize_audio_ffmpeg, apply_noise_reduction_ffmpeg, apply_noise_reduction, _apply_noise_reduction_legacy

# Audio format conversion will use ffmpeg directly via subprocess

# Optional dependency imports with fallback handling
WHISPER_AVAILABLE = False
FASTER_WHISPER_AVAILABLE = False
FFMPEG_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
    logger.info("OpenAI Whisper available")
except ImportError:
    logger.warning("OpenAI Whisper not available")
except Exception as e:
    logger.warning(f"OpenAI Whisper failed to load: {str(e)}")

try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_WHISPER_AVAILABLE = True
    logger.info("faster-whisper available")
except ImportError:
    logger.warning("faster-whisper not available")
except Exception as e:
    logger.warning(f"faster-whisper failed to load: {str(e)}")

try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
    logger.info("ffmpeg-python available")
except ImportError:
    logger.warning("ffmpeg-python not available")
except Exception as e:
    logger.warning(f"ffmpeg-python failed to load: {str(e)}")

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.backends.cuda.sdp_kernel.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*past_key_values.*")

def sanitize_filename(filename: str, part_number: int = None, max_length: int = 80) -> str:
    """Generate safe filename for audio parts"""
    if part_number is not None:
        filename = f"{part_number}_{filename}"
    
    # Replace spaces and remove unsafe characters
    filename = filename.replace(" ", "_")
    filename = re.sub(r"[^\w\-_.]", "", filename)
    filename = filename.lower()

    # Truncate if too long
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        filename = name[:max_length - len(ext)] + ext

    return filename


def ensure_directory(dir_name: str) -> bool:
    """Create directory if it doesn't exist, return success status"""
    try:
        os.makedirs(dir_name, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {dir_name}: {str(e)}")
        return False

def read_wave(path: str) -> Tuple[bytes, int]:
    """Read WAV file and return PCM data and sample rate"""
    try:
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            num_channels = wf.getnchannels()
            if num_channels != 1:
                raise ValueError(f"Audio must be mono, got {num_channels} channels")
            
            sample_width = wf.getsampwidth()
            if sample_width != 2:
                raise ValueError(f"Audio must be 16-bit, got {sample_width * 8}-bit")
            
            sample_rate = wf.getframerate()
            if sample_rate not in (8000, 16000, 32000, 48000):
                raise ValueError(f"Unsupported sample rate: {sample_rate}")
            
            pcm_data = wf.readframes(wf.getnframes())
            return pcm_data, sample_rate
    except Exception as e:
        logger.error(f"Error reading WAV file {path}: {str(e)}")
        raise

def write_wave(path: str, audio: np.ndarray, sample_rate: int) -> None:
    """Write audio data to WAV file"""
    try:
        audio = audio.astype(np.int16).tobytes()
        with contextlib.closing(wave.open(path, 'wb')) as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio)
    except Exception as e:
        logger.error(f"Error writing WAV file {path}: {str(e)}")
        raise

class Frame:
    """Represents a single audio frame"""
    def __init__(self, bytes_data: bytes, timestamp: float, duration: float):
        self.bytes = bytes_data
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms: int, audio: bytes, sample_rate: int):
    """Generate audio frames of specified duration"""
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate: int, frame_duration_ms: int, padding_duration_ms: int, 
                 vad, frames) -> List[np.ndarray]:
    """Collect voiced audio segments using VAD"""
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    
    for frame in frames:
        try:
            is_speech = vad.is_speech(frame.bytes, sample_rate)
        except Exception as e:
            logger.warning(f"VAD error: {str(e)}")
            is_speech = True  # Default to treating as speech on error
        
        
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield np.frombuffer(b''.join([f.bytes for f in voiced_frames]), np.int16)
                ring_buffer.clear()
                voiced_frames = []
    
    
    if voiced_frames:
        yield np.frombuffer(b''.join([f.bytes for f in voiced_frames]), np.int16)

def simple_sentence_split(text: str) -> List[str]:
    """Simple sentence splitting fallback when NLTK is not available"""
    sentences = []
    current = ""
    
    i = 0
    while i < len(text):
        char = text[i]
        current += char
        
        # Check for sentence endings
        if char in '.!?':
            # Look ahead to see if this is likely the end of a sentence
            if i == len(text) - 1:  # End of text
                sentences.append(current.strip())
                current = ""
            elif i + 1 < len(text) and text[i + 1].isspace():
                # Followed by whitespace - check if it's really a sentence end
                if i + 2 < len(text) and text[i + 2].isupper():
                    # Followed by uppercase letter - likely sentence end
                    sentences.append(current.strip())
                    current = ""
                elif i + 2 >= len(text):
                    # Near end of text
                    sentences.append(current.strip())
                    current = ""
                # Otherwise, might be an abbreviation, continue
        i += 1
    
    # Add any remaining text
    if current.strip():
        sentences.append(current.strip())
    
    # Clean up sentences - preserve hyphens in words
    clean_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            # Add period if sentence doesn't end with punctuation
            if not sentence.endswith(('.', '!', '?', '...', ';', ':')):
                sentence += '.'
            clean_sentences.append(sentence)
    
    return clean_sentences

# Cache for NLTK availability check
_nltk_available = None

def ensure_nltk_data() -> bool:
    """Ensure NLTK punkt tokenizer data is available"""
    global _nltk_available
    
    # Return cached result if already checked
    if _nltk_available is not None:
        return _nltk_available
    
    try:
        import nltk
        
        # Check for existing tokenizers
        for tokenizer in ['punkt_tab', 'punkt']:
            try:
                nltk.data.find(f'tokenizers/{tokenizer}')
                _nltk_available = True
                logger.info(f"NLTK {tokenizer} tokenizer found")
                return True
            except LookupError:
                continue
        
        # Attempt to download if not found
        logger.info("NLTK punkt tokenizer not found. Attempting download...")
        
        # Find writable NLTK data path
        nltk_data_path = _find_writable_nltk_path()
        
        if nltk_data_path:
            logger.info(f"Downloading NLTK data to: {nltk_data_path}")
            
            # Try downloading punkt_tab first, then punkt
            for tokenizer in ['punkt_tab', 'punkt']:
                try:
                    nltk.download(tokenizer, download_dir=nltk_data_path, quiet=True)
                    nltk.data.find(f'tokenizers/{tokenizer}')
                    logger.info(f"Successfully downloaded and verified {tokenizer}")
                    _nltk_available = True
                    return True
                except Exception as e:
                    logger.warning(f"{tokenizer} download failed: {str(e)}")
                    continue
        
        # If we get here, NLTK is not available
        logger.warning("NLTK tokenizer unavailable. Using fallback sentence splitter.")
        _nltk_available = False
        return False
        
    except Exception as e:
        logger.error(f"Error checking NLTK availability: {str(e)}")
        _nltk_available = False
        return False

def _find_writable_nltk_path() -> Optional[str]:
    """Find a writable path for NLTK data"""
    import nltk
    
    # Try existing NLTK data paths
    for path in nltk.data.path:
        try:
            test_file = os.path.join(path, '.test_write')
            os.makedirs(path, exist_ok=True)
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return path
        except (OSError, PermissionError):
            continue
    
    # Fallback to user home directory
    fallback_path = os.path.expanduser('~/nltk_data')
    try:
        os.makedirs(fallback_path, exist_ok=True)
        return fallback_path
    except Exception:
        return None

def split_text_part(text_part: str, max_chunk_length: int, nltk_available: bool) -> List[str]:
    """Split a text part into appropriately sized chunks"""
    if not text_part.strip():
        return []
    
    # First split into sentences
    if nltk_available:
        try:
            sentences = nltk.sent_tokenize(text_part)
        except Exception as e:
            logger.warning(f"NLTK tokenization failed: {str(e)}")
            sentences = simple_sentence_split(text_part)
    else:
        sentences = simple_sentence_split(text_part)
    
    # If the entire text is shorter than max_chunk_length, return it as one chunk
    if len(text_part) <= max_chunk_length:
        return [text_part.strip()]
    
    # Then pack sentences into chunks
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Test if adding this sentence would exceed the limit
        test_chunk = current_chunk + (" " + sentence if current_chunk else sentence)
        
        if len(test_chunk) <= max_chunk_length:
            current_chunk = test_chunk
        else:
            # Current chunk is full, save it and start a new one
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Single sentence is too long - need to split it further
                if len(sentence) > max_chunk_length:
                    # Try splitting by commas first
                    comma_parts = sentence.split(',')
                    for j, part in enumerate(comma_parts):
                        part = part.strip()
                        if j < len(comma_parts) - 1:
                            part += ","
                        
                        test_chunk = current_chunk + (" " + part if current_chunk else part)
                        
                        if len(test_chunk) <= max_chunk_length:
                            current_chunk = test_chunk
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                                current_chunk = part
                            else:
                                # Part is still too long - split by words
                                if len(part) > max_chunk_length:
                                    words = part.split()
                                    for word in words:
                                        test_chunk = current_chunk + (" " + word if current_chunk else word)
                                        if len(test_chunk) <= max_chunk_length:
                                            current_chunk = test_chunk
                                        else:
                                            if current_chunk:
                                                chunks.append(current_chunk.strip())
                                            current_chunk = word
                                else:
                                    current_chunk = part
                else:
                    # Sentence fits within limit
                    current_chunk = sentence
    
    # Don't forget the last chunk
    if current_chunk and current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Final validation - remove empty chunks and ensure no duplicates
    final_chunks = []
    seen_chunks = set()
    for chunk in chunks:
        chunk = chunk.strip()
        if chunk and chunk not in seen_chunks:
            final_chunks.append(chunk)
            seen_chunks.add(chunk)
    
    return final_chunks

def split_text_into_chunks(text_input: str, max_chunk_length: int = 300) -> Tuple[List[dict], List[Optional[float]]]:
    """
    Split text into chunks suitable for Chatterbox TTS processing.
    Break tags become their own chunks with type 'pause'.
    Text is properly cleaned and separated to avoid TTS model confusion.
    Returns list of chunk dictionaries with 'type' and 'content' keys.
    """
    # Ensure NLTK data is available
    nltk_available = ensure_nltk_data()
    
    # Basic text cleaning - normalize whitespace
    text_input = re.sub(r'\n+', ' ', text_input)
    text_input = re.sub(r'\s+', ' ', text_input)
    text_input = text_input.strip()

    logger.debug(f"Original text: {text_input[:100]}{'...' if len(text_input) > 100 else ''}")

    # Step 1: Extract break tags and their positions, then remove them from text
    break_pattern = r'<break\s+time="([\d.]+)(?:s|ms)?"\s*/>'
    
    # Find all break tags with their positions
    break_matches = []
    for match in re.finditer(break_pattern, text_input, re.IGNORECASE):
        time_value = match.group(1)
        # Determine if it's milliseconds or seconds
        if 'ms' in match.group(0).lower():
            pause_duration = float(time_value) / 1000.0
        else:
            pause_duration = float(time_value)
        
        break_matches.append({
            'start': match.start(),
            'end': match.end(),
            'duration': pause_duration,
            'original': match.group(0)
        })
    
    logger.info(f"Found {len(break_matches)} break tags")
    
    # Step 2: Process text based on whether break tags are present
    chunks = []
    
    if break_matches:
        # Process text with break tags
        current_pos = 0
        
        for break_match in break_matches:
            # Extract text before this break tag
            text_before = text_input[current_pos:break_match['start']].strip()
            
            # Process the text segment if it's not empty
            if text_before:
                processed_chunks = process_text_segment(text_before, max_chunk_length, nltk_available)
                chunks.extend(processed_chunks)
            
            # Add the pause chunk
            chunks.append({
                'type': 'pause',
                'content': break_match['duration']
            })
            logger.info(f"Added pause chunk: {break_match['duration']}s")
            
            current_pos = break_match['end']
        
        # Process any remaining text after the last break tag
        remaining_text = text_input[current_pos:].strip()
        if remaining_text:
            processed_chunks = process_text_segment(remaining_text, max_chunk_length, nltk_available)
            chunks.extend(processed_chunks)
    else:
        # No break tags found, process the entire text normally
        processed_chunks = process_text_segment(text_input, max_chunk_length, nltk_available)
        chunks.extend(processed_chunks)
    
    # Clean up any empty text chunks and validate
    cleaned_chunks = []
    for chunk in chunks:
        if chunk['type'] == 'pause':
            if chunk['content'] > 0:  # Only add valid pause durations
                cleaned_chunks.append(chunk)
        elif chunk['type'] == 'text':
            content = chunk['content'].strip()
            if content:  # Only add non-empty text chunks
                cleaned_chunks.append({
                    'type': 'text',
                    'content': content
                })
    
    logger.info(f"Final chunks: {len(cleaned_chunks)} total")
    for idx, chunk in enumerate(cleaned_chunks):
        if chunk['type'] == 'pause':
            logger.info(f"Chunk {idx}: PAUSE {chunk['content']}s")
        else:
            logger.info(f"Chunk {idx}: TEXT '{chunk['content'][:50]}...'")
    
    return cleaned_chunks, []  # Return empty pause list since pauses are now in chunks

def process_text_segment(text_segment: str, max_chunk_length: int, nltk_available: bool) -> List[dict]:
    """
    Process a text segment (without break tags) into text chunks.
    Handles music notes and regular text splitting.
    """
    chunks = []
    
    if '♪' in text_segment:
        # Handle music notes separately
        song_parts = re.split(r'♪', text_segment)
        for j, song_part in enumerate(song_parts):
            song_part = song_part.strip()
            if not song_part:
                continue
            if j % 2 == 0:
                # Non-song parts - split into sentences and then chunks
                text_chunks = split_text_part(song_part, max_chunk_length, nltk_available)
                for text_chunk in text_chunks:
                    chunks.append({
                        'type': 'text',
                        'content': text_chunk
                    })
            else:
                # Song parts - keep as single units
                chunks.append({
                    'type': 'text',
                    'content': '♪ ' + song_part.strip() + ' ♪'
                })
    else:
        # Regular text splitting - this is where the fix is applied
        text_chunks = split_text_part(text_segment, max_chunk_length, nltk_available)
        for text_chunk in text_chunks:
            if text_chunk.strip():  # Only add non-empty chunks
                chunks.append({
                    'type': 'text',
                    'content': text_chunk.strip()
                })
    
    return chunks


def calculate_natural_pause(current_chunk: str, next_chunk: Optional[str] = None, 
                          sample_rate: int = 24000, custom_pause: Optional[float] = None) -> int:
    """Calculate natural pause duration based on sentence structure and content"""
    try:
        if custom_pause is not None:
            return int(custom_pause * sample_rate)
        
        base_pause = int(0.4 * sample_rate)  # 400ms base pause
        current_chunk = current_chunk.strip()
        
        # Different pause lengths based on punctuation
        if current_chunk.endswith('...'):
            pause_multiplier = 2.2
        elif current_chunk.endswith('!'):
            pause_multiplier = 1.8
        elif current_chunk.endswith('?'):
            pause_multiplier = 1.7
        elif current_chunk.endswith('.'):
            word_count = len(current_chunk.split())
            pause_multiplier = 1.5 if word_count > 15 else 1.2
        elif current_chunk.endswith(','):
            pause_multiplier = 0.8
        elif current_chunk.endswith(';') or current_chunk.endswith(':'):
            pause_multiplier = 1.4
        else:
            # For chunks without punctuation (like mid-sentence chunks), use shorter pause
            pause_multiplier = 0.3
        
        # Analyze next chunk beginning
        if next_chunk:
            next_chunk = next_chunk.strip()
            if next_chunk:
                question_starters = ['what', 'when', 'where', 'why', 'who', 'how', 'which', 'whose']
                emphasis_words = ['but', 'however', 'meanwhile', 'furthermore', 'moreover', 'therefore']
                
                first_word = next_chunk.split()[0].lower()
                
                if first_word in question_starters:
                    pause_multiplier *= 1.3
                elif first_word in emphasis_words:
                    pause_multiplier *= 1.4
                elif next_chunk.startswith('"') or next_chunk.startswith("'"):
                    pause_multiplier *= 1.2
        
        # Check if current chunk ends with a hyphenated word that might continue
        # This helps with words like "text-to-speech" that get split across chunks
        words = current_chunk.split()
        if words and not current_chunk.endswith(('.', '!', '?', '...', ';', ':')):
            last_word = words[-1]
            # If the chunk doesn't end with punctuation and last word ends with hyphen,
            # or if we're in the middle of a sentence, use a much shorter pause
            if last_word.endswith('-') or not any(current_chunk.endswith(p) for p in '.!?;:'):
                pause_multiplier = 0.15  # Very short pause for mid-sentence chunks
        
        # Add natural randomness (±10% for shorter pauses to avoid artifacts)
        randomness = np.random.uniform(0.9, 1.1)
        pause_multiplier *= randomness
        
        # Calculate final pause duration with bounds
        pause_duration = int(base_pause * pause_multiplier)
        min_pause = int(0.05 * sample_rate)  # Minimum 50ms
        max_pause = int(1.5 * sample_rate)   # Maximum 1.5s
        pause_duration = max(min_pause, min(max_pause, pause_duration))
        
        return pause_duration
    except Exception as e:
        logger.error(f"Error calculating natural pause: {str(e)}")
        return int(0.2 * sample_rate)  # Shorter default pause

def generate_natural_silence(duration_samples: int, sample_rate: int = 24000, 
                           silence_type: str = 'natural') -> np.ndarray:
    """Generate more natural silence with subtle ambient characteristics"""
    try:
        if silence_type == 'natural':
            room_tone_level = 0.0001
            room_tone = np.random.normal(0, room_tone_level, duration_samples)
            
            # Add very subtle low-frequency rumble for longer pauses
            if duration_samples > sample_rate * 0.1:
                t = np.linspace(0, duration_samples / sample_rate, duration_samples)
                subtle_rumble = 0.00005 * np.sin(2 * np.pi * 60 * t)
                room_tone += subtle_rumble
            
            return room_tone.astype(np.float32)
        else:
            return np.zeros(duration_samples, dtype=np.float32)
    except Exception as e:
        logger.error(f"Error generating natural silence: {str(e)}")
        return np.zeros(duration_samples, dtype=np.float32)

def initialize_chatterbox_model():
    """Initialize Chatterbox TTS model with device detection"""
    try:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        logger.info(f"Initializing Chatterbox TTS model on device: {device}")
        
        # For MPS devices, handle torch.load properly
        if device == "mps":
            map_location = torch.device(device)
            torch_load_original = torch.load
            def patched_torch_load(*args, **kwargs):
                if 'map_location' not in kwargs:
                    kwargs['map_location'] = map_location
                return torch_load_original(*args, **kwargs)
            torch.load = patched_torch_load
        
        # Suppress warnings during model loading
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ChatterboxTTS.from_pretrained(device=device)
        
        return model, device
    except Exception as e:
        logger.error(f"Error initializing Chatterbox model: {str(e)}")
        raise

# Global model instances (loaded once)
_chatterbox_model = None
_model_device = None

# Global model instances (thread-safe caching)
_whisper_model = None
_whisper_model_name = ""
_whisper_use_faster = False

def get_chatterbox_model():
    """Get or initialize the global Chatterbox model instance"""
    global _chatterbox_model, _model_device
    try:
        if _chatterbox_model is None:
            _chatterbox_model, _model_device = initialize_chatterbox_model()
        return _chatterbox_model, _model_device
    except Exception as e:
        logger.error(f"Error getting Chatterbox model: {str(e)}")
        raise

def set_seed(seed: int) -> None:
    """Set random seed for reproducible generation"""
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
    except Exception as e:
        logger.error(f"Error setting seed: {str(e)}")

# Phase 1: Whisper validation functions
def normalize_for_compare_all_punct(text: str) -> str:
    """Normalize text for comparison by removing all punctuation and extra spaces"""
    try:
        # Remove all punctuation and special characters, keep only alphanumeric and spaces
        text = re.sub(r'[^\w\s]', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    except Exception as e:
        logger.error(f"Error normalizing text: {str(e)}")
        return text.lower()

def download_faster_whisper_model(model_name: str = "medium") -> None:
    """
    Download a faster-whisper model locally for offline use.
    This function temporarily enables downloads to cache the model.
    """
    try:
        # Temporarily allow downloads
        old_offline = os.environ.get('HF_HUB_OFFLINE', '0')
        os.environ['HF_HUB_OFFLINE'] = '0'
        
        logger.info(f"Downloading faster-whisper model: {model_name}")
        model = FasterWhisperModel(model_name, device="cpu", compute_type="float32")
        logger.info(f"Model {model_name} downloaded successfully")
        
        # Clean up
        del model
        
        # Restore offline setting
        os.environ['HF_HUB_OFFLINE'] = old_offline
        
    except Exception as e:
        logger.error(f"Failed to download model {model_name}: {str(e)}")
        # Restore offline setting
        os.environ['HF_HUB_OFFLINE'] = old_offline
        raise

def load_whisper_model(model_name: str = "base", use_faster_whisper: bool = True, device: str = "auto") -> Optional[object]:
    """Load Whisper model with caching and thread safety"""
    global _whisper_model, _whisper_model_name, _whisper_use_faster
    
    try:
        # Auto-detect device if needed
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        # Get current global state
        current_model = _whisper_model
        current_model_name = _whisper_model_name
        current_use_faster = _whisper_use_faster
        
        # Check if we need to reload the model
        if (current_model is None or 
            current_model_name != model_name or 
            current_use_faster != use_faster_whisper):
            
            # Clear existing model
            if current_model is not None:
                _cleanup_model(current_model)
            
            logger.info(f"Loading Whisper model: {model_name} ({'faster-whisper' if use_faster_whisper else 'OpenAI Whisper'})")
            
            if use_faster_whisper and FASTER_WHISPER_AVAILABLE:
                compute_type = "float16" if device == "cuda" else "float32"
                try:
                    # Force local_files_only to prevent Hugging Face downloads
                    _whisper_model = FasterWhisperModel(
                        model_name, 
                        device=device, 
                        compute_type=compute_type,
                        local_files_only=True
                    )
                except Exception as e:
                    logger.error(f"Failed to load local faster-whisper model '{model_name}': {str(e)}")
                    logger.info("To download the model locally, run:")
                    logger.info(f"  python -c \"from connector import download_faster_whisper_model; download_faster_whisper_model('{model_name}')\"")
                    logger.info("Or use a different model name that's already cached locally")
                    return None
            elif WHISPER_AVAILABLE:
                try:
                    _whisper_model = whisper.load_model(model_name, device=device)
                except Exception as e:
                    logger.error(f"Failed to load OpenAI Whisper model '{model_name}': {str(e)}")
                    return None
            else:
                logger.error("No Whisper backend available")
                return None
            
            _whisper_model_name = model_name
            _whisper_use_faster = use_faster_whisper
            logger.info(f"Whisper model loaded successfully on {device}")
            
            return _whisper_model
        else:
            # Return existing model
            return current_model
        
    except Exception as e:
        logger.error(f"Error loading Whisper model: {str(e)}")
        _whisper_model = None
        return None

def _cleanup_model(model) -> None:
    """Safely cleanup a model and free GPU memory"""
    try:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        logger.warning(f"Model cleanup warning: {str(e)}")

def whisper_validate_audio(audio_path: str, target_text: str, 
                          whisper_model_name: str = "base", 
                          use_faster_whisper: bool = True, 
                          validation_threshold: float = 0.85) -> Tuple[bool, float, str]:
    """
    Validate generated audio using Whisper transcription
    Returns: (is_valid, similarity_score, transcribed_text)
    """
    try:
        if not os.path.exists(audio_path):
            return False, 0.0, "Audio file not found"
        
        # Load Whisper model with error handling
        whisper_model = None
        try:
            whisper_model = load_whisper_model(whisper_model_name, use_faster_whisper)
        except Exception as e:
            logger.error(f"Failed to load Whisper model in validation: {str(e)}")
            return True, 1.0, f"Whisper validation failed: {str(e)}"
        
        if whisper_model is None:
            logger.warning("Whisper model not available, skipping validation")
            return True, 1.0, "Whisper validation skipped"
        
        # Check audio file properties first
        try:
            audio_data, sr = librosa.load(audio_path, sr=None)
            duration = len(audio_data) / sr
            
            # Skip validation for very short audio (less than 0.5 seconds)
            if duration < 0.5:
                logger.warning(f"Audio too short ({duration:.2f}s) for reliable Whisper validation")
                return True, 1.0, "Audio too short for validation"
                
        except Exception as e:
            logger.warning(f"Could not analyze audio properties: {str(e)}")
        
        # Transcribe audio with enhanced error handling
        transcribed = ""
        try:
            if use_faster_whisper and FASTER_WHISPER_AVAILABLE:
                segments, info = whisper_model.transcribe(audio_path)
                transcribed = "".join([seg.text for seg in segments]).strip()
            else:
                # For OpenAI Whisper, add more robust error handling
                try:
                    result = whisper_model.transcribe(
                        audio_path,
                        fp16=False,  # Use fp32 for better stability
                        temperature=0.0,  # Use deterministic decoding
                        compression_ratio_threshold=2.4,  # Default threshold
                        logprob_threshold=-1.0,  # Default threshold
                        no_speech_threshold=0.6,  # Default threshold
                        condition_on_previous_text=False  # Don't condition on previous text
                    )
                    transcribed = result['text'].strip()
                except RuntimeError as rt_error:
                    if "Key and Value must have the same sequence length" in str(rt_error):
                        logger.warning(f"Whisper attention error, trying with different settings: {str(rt_error)}")
                        # Retry with different parameters
                        try:
                            result = whisper_model.transcribe(
                                audio_path,
                                fp16=False,
                                temperature=0.2,  # Add some temperature
                                best_of=1,  # Use only one beam
                                beam_size=1,  # Single beam
                                patience=1.0,
                                length_penalty=1.0,
                                suppress_tokens=[-1],
                                initial_prompt=None,
                                condition_on_previous_text=False,
                                compression_ratio_threshold=2.4,
                                logprob_threshold=-1.0,
                                no_speech_threshold=0.6
                            )
                            transcribed = result['text'].strip()
                        except Exception as retry_error:
                            logger.error(f"Whisper retry also failed: {str(retry_error)}")
                            return True, 0.0, f"Whisper transcription failed after retry: {str(retry_error)}"
                    else:
                        raise rt_error
                        
        except Exception as e:
            logger.error(f"Whisper transcription failed: {str(e)}")
            return True, 0.0, f"Transcription failed: {str(e)}"
        
        if not transcribed:
            logger.warning("Whisper returned empty transcription")
            return True, 0.0, "Empty transcription"
        
        # Normalize texts for comparison
        normalized_transcribed = normalize_for_compare_all_punct(transcribed)
        normalized_target = normalize_for_compare_all_punct(target_text)
        
        # Calculate similarity score using SequenceMatcher
        similarity_score = difflib.SequenceMatcher(
            None, normalized_transcribed, normalized_target
        ).ratio()
        
        is_valid = similarity_score >= validation_threshold
        
        logger.info(f"Whisper validation - Score: {similarity_score:.3f}, Valid: {is_valid}")
        logger.debug(f"Target: '{normalized_target[:100]}'")
        logger.debug(f"Transcribed: '{normalized_transcribed[:100]}'")
        
        return is_valid, similarity_score, transcribed
        
    except Exception as e:
        logger.error(f"Error in Whisper validation: {str(e)}")
        return True, 1.0, f"Validation error: {str(e)}"
    
def whisper_validate_audio_with_model(whisper_model, audio_path: str, target_text: str, 
                                    use_faster_whisper: bool = True,
                                    validation_threshold: float = 0.85) -> Tuple[bool, float, str]:
    """
    Validate audio using an already-loaded Whisper model
    This avoids reloading the model for each candidate
    """
    try:
        if not os.path.exists(audio_path):
            return False, 0.0, "Audio file not found"
        
        # Check audio file properties first
        try:
            import librosa
            audio_data, sr = librosa.load(audio_path, sr=None)
            duration = len(audio_data) / sr
            
            # Skip validation for very short audio (less than 0.5 seconds)
            if duration < 0.5:
                logger.warning(f"Audio too short ({duration:.2f}s) for reliable Whisper validation")
                return True, 1.0, "Audio too short for validation"
                
        except Exception as e:
            logger.warning(f"Could not analyze audio properties: {str(e)}")
        
        # Transcribe audio with enhanced error handling
        transcribed = ""
        try:
            if use_faster_whisper and FASTER_WHISPER_AVAILABLE:
                segments, info = whisper_model.transcribe(audio_path)
                transcribed = "".join([seg.text for seg in segments]).strip()
            else:
                # For OpenAI Whisper, add more robust error handling
                try:
                    result = whisper_model.transcribe(
                        audio_path,
                        fp16=False,  # Use fp32 for better stability
                        temperature=0.0,  # Use deterministic decoding
                        compression_ratio_threshold=2.4,
                        logprob_threshold=-1.0,
                        no_speech_threshold=0.6,
                        condition_on_previous_text=False
                    )
                    transcribed = result['text'].strip()
                except RuntimeError as rt_error:
                    if "Key and Value must have the same sequence length" in str(rt_error):
                        logger.warning(f"Whisper attention error, trying different settings")
                        try:
                            result = whisper_model.transcribe(
                                audio_path,
                                fp16=False,
                                temperature=0.2,
                                best_of=1,
                                beam_size=1,
                                condition_on_previous_text=False
                            )
                            transcribed = result['text'].strip()
                        except Exception:
                            logger.warning("Whisper validation failed, treating as valid")
                            return True, 0.0, "Validation failed but treating as valid"
                    else:
                        raise rt_error
                        
        except Exception as e:
            logger.warning(f"Whisper transcription failed: {str(e)}, treating as valid")
            return True, 0.0, f"Transcription failed: {str(e)}"
        
        if not transcribed:
            logger.warning("Whisper returned empty transcription")
            return True, 0.0, "Empty transcription"
        
        # Normalize texts for comparison
        normalized_transcribed = normalize_for_compare_all_punct(transcribed)
        normalized_target = normalize_for_compare_all_punct(target_text)
        
        # Calculate similarity score
        similarity_score = difflib.SequenceMatcher(
            None, normalized_transcribed, normalized_target
        ).ratio()
        
        is_valid = similarity_score >= validation_threshold
        
        logger.debug(f"Whisper validation - Score: {similarity_score:.3f}, Valid: {is_valid}")
        
        return is_valid, similarity_score, transcribed
        
    except Exception as e:
        logger.warning(f"Error in Whisper validation: {str(e)}, treating as valid")
        return True, 1.0, f"Validation error: {str(e)}"

def generate_multiple_candidates(model, text_chunk: str, audio_prompt_path: Optional[str], 
                               exaggeration: float, temperature: float, cfg_weight: float,
                               num_candidates: int = 3, max_attempts: int = 2) -> List[dict]:
    """
    Generate multiple audio candidates for a text chunk
    Returns list of candidate dictionaries with path, duration, and metadata
    """
    candidates = []
    
    try:
        for cand_idx in range(num_candidates):
            for attempt in range(max_attempts):
                try:
                    # Use different seeds for variety
                    candidate_seed = random.randint(1, 2**31-1)
                    set_seed(candidate_seed)
                    
                    # Generate audio
                    if audio_prompt_path and os.path.exists(audio_prompt_path):
                        wav_tensor = model.generate(
                            text_chunk,
                            audio_prompt_path=audio_prompt_path,
                            exaggeration=exaggeration,
                            temperature=temperature,
                            cfg_weight=cfg_weight
                        )
                    else:
                        wav_tensor = model.generate(
                            text_chunk,
                            exaggeration=exaggeration,
                            temperature=temperature,
                            cfg_weight=cfg_weight
                        )
                    
                    # Convert to numpy
                    if isinstance(wav_tensor, torch.Tensor):
                        audio_array = wav_tensor.squeeze().cpu().numpy()
                    else:
                        audio_array = wav_tensor
                    
                    # Ensure proper format
                    if audio_array.dtype != np.float32:
                        audio_array = audio_array.astype(np.float32)
                    
                    # Normalize
                    max_val = np.max(np.abs(audio_array))
                    if max_val > 0:
                        audio_array = audio_array / max_val
                    
                    # Save candidate to temporary file
                    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_path = temp_file.name
                    temp_file.close()
                    
                    # Save audio
                    audio_tensor = torch.tensor(audio_array).unsqueeze(0)
                    ta.save(temp_path, audio_tensor, model.sr)
                    
                    # Calculate duration
                    duration = len(audio_array) / model.sr
                    
                    candidate = {
                        'path': temp_path,
                        'duration': duration,
                        'audio_array': audio_array,
                        'seed': candidate_seed,
                        'candidate_index': cand_idx,
                        'attempt': attempt,
                        'text_chunk': text_chunk
                    }
                    
                    candidates.append(candidate)
                    logger.info(f"Generated candidate {cand_idx + 1}/{num_candidates}, attempt {attempt + 1}")
                    
                    # Clean up GPU memory
                    if isinstance(wav_tensor, torch.Tensor):
                        del wav_tensor
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    break  # Success, no need to retry
                    
                except Exception as e:
                    logger.warning(f"Candidate generation failed (candidate {cand_idx+1}, attempt {attempt+1}): {str(e)}")
                    if attempt == max_attempts - 1:
                        logger.error(f"All attempts failed for candidate {cand_idx+1}")
    
    except Exception as e:
        logger.error(f"Error in candidate generation: {str(e)}")
    
    return candidates


def select_best_candidate(candidates: List[dict], use_whisper_validation: bool = True,
                         whisper_model_name: str = "base", use_faster_whisper: bool = True,
                         validation_threshold: float = 0.85) -> Optional[dict]:
    """
    Select the best candidate from generated options
    Priority: Whisper validation > shortest duration > first candidate
    """
    if not candidates:
        return None
    
    if len(candidates) == 1:
        return candidates[0]
    
    try:
        if use_whisper_validation and (WHISPER_AVAILABLE or FASTER_WHISPER_AVAILABLE):
            logger.info("Using Whisper validation for candidate selection")
            
            # Load Whisper model once for all candidates
            whisper_model = None
            try:
                whisper_model = load_whisper_model(whisper_model_name, use_faster_whisper)
            except Exception as e:
                logger.error(f"Failed to load Whisper model for validation: {str(e)}")
                whisper_model = None
            
            if whisper_model is None:
                logger.warning("Whisper model not available, falling back to duration selection")
            else:
                # Validate all candidates using the same model instance
                validated_candidates = []
                for i, candidate in enumerate(candidates):
                    logger.debug(f"Validating candidate {i+1}/{len(candidates)}")
                    
                    is_valid, score, transcribed = whisper_validate_audio_with_model(
                        whisper_model,
                        candidate['path'], 
                        candidate['text_chunk'],
                        use_faster_whisper,
                        validation_threshold
                    )
                    
                    candidate['whisper_valid'] = is_valid
                    candidate['whisper_score'] = score
                    candidate['whisper_transcribed'] = transcribed
                    
                    if is_valid:
                        validated_candidates.append(candidate)
                
                # Return best validated candidate (highest score)
                if validated_candidates:
                    best_candidate = max(validated_candidates, key=lambda x: x['whisper_score'])
                    logger.info(f"Selected candidate with Whisper score: {best_candidate['whisper_score']:.3f}")
                    return best_candidate
                else:
                    logger.warning("No candidates passed Whisper validation, falling back to duration selection")
        
        # Fallback: select candidate with shortest duration (assumes fewer artifacts)
        best_candidate = min(candidates, key=lambda x: x['duration'])
        logger.info(f"Selected candidate with shortest duration: {best_candidate['duration']:.2f}s")
        return best_candidate
        
    except Exception as e:
        logger.error(f"Error in candidate selection: {str(e)}")
        # Ultimate fallback: return first candidate
        return candidates[0]

def cleanup_candidate_files(candidates: List[dict], keep_selected: Optional[dict] = None) -> None:
    """Clean up temporary candidate files"""
    if not candidates:
        return
        
    for candidate in candidates:
        if candidate != keep_selected and candidate.get('path'):
            try:
                if os.path.exists(candidate['path']):
                    os.unlink(candidate['path'])
            except Exception as e:
                logger.warning(f"Failed to cleanup candidate file {candidate['path']}: {str(e)}")

def convert_audio_format(input_path: str, output_path: str, output_format: str) -> bool:
    """Convert audio to different format using ffmpeg"""
    try:
        if not os.path.exists(input_path):
            logger.error(f"Input file does not exist: {input_path}")
            return False
        
        # Build ffmpeg command based on output format
        cmd = ["ffmpeg", "-i", input_path, "-y"]  # -y to overwrite output files
        
        if output_format.lower() == "mp3":
            cmd.extend(["-codec:a", "libmp3lame", "-b:a", "320k"])
        elif output_format.lower() == "flac":
            cmd.extend(["-codec:a", "flac"])
        else:
            logger.error(f"Unsupported output format: {output_format}")
            return False
        
        cmd.append(output_path)
        
        logger.info(f"Converting {input_path} to {output_format.upper()}: {output_path}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info(f"Successfully converted to {output_format.upper()}")
            return True
        else:
            logger.error(f"FFmpeg conversion failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg conversion timed out")
        return False
    except Exception as e:
        logger.error(f"Error in audio format conversion: {str(e)}")
        return False


def generate_voice(text_input: str, audio_prompt_path: Optional[str] = None, 
                  exaggeration: float = 0.5, temperature: float = 0.8, 
                  cfg_weight: float = 0.5, chunk_size: int = 300, 
                  speed: float = 1.0, pitch: int = 0, reduce_noise: bool = False, 
                  remove_silence: bool = False, seed: int = 0, 
                  progress_callback: Optional[Callable] = None,
                  status_callback: Optional[Callable] = None,
                  # Export formats
                  export_formats: List[str] = None,
                  # Phase 1: New parameters for candidate generation and Whisper validation
                  num_candidates: int = 3, use_whisper_validation: bool = True,
                  whisper_model_name: str = "base", use_faster_whisper: bool = True,
                  use_longest_transcript: bool = True, validation_threshold: float = 0.85, max_attempts: int = 2,
                  # Phase 2: Parallel processing parameters
                  enable_parallel: bool = True, num_workers: int = 4,
                  # Text processing parameters
                  to_lowercase: bool = True, normalize_spacing: bool = True,
                  fix_dot_letters: bool = True, remove_reference_numbers: bool = True,
                  sound_words: str = "",
                  # Phase 3: Audio post-processing parameters
                  use_ffmpeg_normalize: bool = False, normalize_method: str = "ebu",
                  integrated_loudness: int = -24, true_peak: int = -2, loudness_range: int = 7,
                  noise_reduction_method: str = "afftdn",
                  noise_strength: float = 0.85) -> Tuple[Optional[str], int, str]:
    """
    Enhanced TTS generation with multiple candidates, Whisper validation, parallel processing, and audio post-processing
    Returns: Tuple of (output_filename, actual_seed_used, processed_text)
    """
    try:
        logger.info("Starting enhanced Chatterbox TTS generation...")
        
        if status_callback:
            status_callback("🚀 Starting audio generation...")
        
        # Set default export formats
        if export_formats is None:
            export_formats = ['wav']
        elif not export_formats:
            export_formats = ['wav']
        
        # Validate export formats
        valid_formats = ['wav', 'mp3', 'flac']
        export_formats = [fmt.lower() for fmt in export_formats if fmt.lower() in valid_formats]
        if not export_formats:
            export_formats = ['wav']
        
        # Auto-disable features if dependencies are not available
        if use_whisper_validation and not (WHISPER_AVAILABLE or FASTER_WHISPER_AVAILABLE):
            logger.warning("Whisper validation requested but neither OpenAI Whisper nor faster-whisper available. Disabling.")
            use_whisper_validation = False
        
        if use_ffmpeg_normalize and not FFMPEG_AVAILABLE:
            logger.warning("FFMPEG normalization requested but ffmpeg-python not available. Disabling.")
            use_ffmpeg_normalize = False
        
        # Optimize parameters based on available features
        if not use_whisper_validation and num_candidates > 1:
            logger.info(f"Whisper validation disabled, reducing candidates from {num_candidates} to 1 for efficiency")
            num_candidates = 1
            
        # Validate and clamp parameter ranges
        num_candidates = max(1, min(10, num_candidates))
        max_attempts = max(1, min(10, max_attempts))
        num_workers = max(1, min(8, num_workers))
        
        logger.info(f"Phase 1 - Candidates: {num_candidates}, Whisper validation: {use_whisper_validation}")
        logger.info(f"Phase 2 - Parallel: {enable_parallel}, Workers: {num_workers}")
        logger.info(f"Phase 3 - FFMPEG normalize: {use_ffmpeg_normalize}")
        logger.info(f"Basic params - Audio Prompt: {audio_prompt_path}, Exaggeration: {exaggeration}, Temperature: {temperature}")
        logger.debug(f"CFG Weight: {cfg_weight}, Chunk Size: {chunk_size}, Speed: {speed}, Pitch: {pitch}, Seed: {seed}")
        
        # Handle seed generation
        actual_seed = seed
        if seed == 0:
            actual_seed = random.randint(1, 999999)
            logger.debug(f"Generated random seed: {actual_seed}")
        
        set_seed(int(actual_seed))
        
        if status_callback:
            status_callback(f"🎲 Using seed: {actual_seed}")
        
        if status_callback:
            status_callback("🤖 Loading Chatterbox model...")
        
        model, device = get_chatterbox_model()
        
        # Process and validate text input - keep newlines for text processing
        script = text_input.strip()
        
        if status_callback:
            status_callback("📝 Processing text input...")
        
        # Apply text processing options
        if to_lowercase:
            script = script.lower()
            logger.info("Applied lowercase conversion")
        
        if normalize_spacing:
            # Remove extra spaces and normalize whitespace
            script = re.sub(r'\s+', ' ', script).strip()
            logger.info("Applied spacing normalization")
        
        if fix_dot_letters:
            # Convert J.R.R. style to J R R
            script = re.sub(r'([A-Z])\.([A-Z])\.', r'\1 \2 ', script)
            script = re.sub(r'([A-Z])\.([A-Z])\.([A-Z])\.', r'\1 \2 \3 ', script)
            logger.info("Applied dot letters conversion")
        
        if remove_reference_numbers:
            logger.info(f"Before reference number removal: {repr(script[:100])}")
            original_script = script
            
            # Remove numbered list items at the beginning of lines (1. 2. 3. etc.)
            script = re.sub(r'^\s*\d+\.\s*', '', script, flags=re.MULTILINE)
            logger.info(f"After list number removal: {repr(script[:100])}")
            
            # Remove inline reference numbers like ".188", ".3" at end of sentences
            script = re.sub(r'\.\d+(?=\s|$)', '.', script)
            
            # Remove numbered references in parentheses or brackets like (1), [2], etc.
            script = re.sub(r'[\(\[]?\d+[\)\]]?(?=\s|$)', '', script)
            
            # Clean up extra spaces that might result from removals
            script = re.sub(r'\s+', ' ', script).strip()
            
            if script != original_script:
                logger.info("Applied reference number removal (list items and inline references)")
                logger.info(f"Text changed from: {repr(original_script[:100])} to: {repr(script[:100])}")
            else:
                logger.warning("Reference number removal was enabled but no changes were made to the text")
        
        if sound_words:
            logger.info(f"Sound words input received: {repr(sound_words)}")
            # Process sound words replacement
            sound_replacements = []
            for line in sound_words.split('\n'):
                line = line.strip()
                if not line:
                    continue
                if '=>' in line:
                    old, new = line.split('=>', 1)
                    sound_replacements.append((old.strip(), new.strip()))
                    logger.info(f"Sound replacement: '{old.strip()}' => '{new.strip()}'")
                else:
                    sound_replacements.append((line, ''))
                    logger.info(f"Sound removal: '{line}'")
            
            original_script = script
            for old, new in sound_replacements:
                if old:
                    before_count = script.count(old)
                    script = script.replace(old, new)
                    after_count = script.count(old)
                    if before_count > 0:
                        logger.info(f"Replaced {before_count} occurrences of '{old}' with '{new}'")
                    else:
                        logger.warning(f"No occurrences of '{old}' found in text")
            
            if sound_replacements:
                logger.info(f"Applied {len(sound_replacements)} sound word replacements")
                if original_script != script:
                    logger.info(f"Text changed from: {repr(original_script[:100])} to: {repr(script[:100])}")
                else:
                    logger.warning("Sound word processing was enabled but no changes were made to the text")
        else:
            logger.info("No sound words provided")
        
        # Convert newlines to spaces for TTS processing (after all text processing is done)
        processed_text = script  # Keep the processed text with newlines for JSON storage
        script = script.replace("\n", " ").strip()
        
        if not script:
            logger.error("Empty text after processing")
            return None, actual_seed, processed_text if 'processed_text' in locals() else text_input
            
        chunks, _ = split_text_into_chunks(script, chunk_size)
        
        logger.info(f"Text split into {len(chunks)} chunks")
        
        if status_callback:
            status_callback(f"📄 Split text into {len(chunks)} chunks")
        
        if not chunks:
            logger.error("No text chunks to process")
            return None, actual_seed, processed_text
            
        # Disable parallel processing for single chunks
        if len(chunks) <= 1:
            enable_parallel = False
            logger.info("Single chunk detected, disabling parallel processing")
        
        # Separate text chunks from pause chunks for processing
        text_chunks = []
        chunk_map = {}  # Maps text chunk index to original position
        
        for i, chunk_dict in enumerate(chunks):
            if chunk_dict['type'] == 'text':
                chunk_map[len(text_chunks)] = i
                text_chunks.append(chunk_dict)
        
        if not text_chunks:
            logger.error("No text chunks found for processing")
            return None, actual_seed, processed_text
        
        logger.info(f"Processing {len(text_chunks)} text chunks with {len(chunks) - len(text_chunks)} pause chunks")
        
        # Phase 2: Process text chunks (with parallel processing if enabled)
        def process_single_text_chunk(chunk_index: int, chunk_dict: dict) -> Tuple[int, Optional[np.ndarray]]:
            """Process a single text chunk with candidate generation and selection"""
            try:
                chunk_content = chunk_dict['content']
                logger.info(f"Processing text chunk {chunk_index + 1}/{len(text_chunks)}: '{chunk_content[:50]}...'")
                
                # Phase 1: Generate multiple candidates
                candidates = generate_multiple_candidates(
                    model, chunk_content, audio_prompt_path,
                    exaggeration, temperature, cfg_weight,
                    num_candidates, max_attempts
                )
                
                if not candidates:
                    logger.error(f"No candidates generated for chunk {chunk_index + 1}")
                    return chunk_index, None
                
                # Phase 1: Select best candidate using Whisper validation
                best_candidate = select_best_candidate(
                    candidates, use_whisper_validation,
                    whisper_model_name, use_faster_whisper, validation_threshold
                )
                
                if best_candidate is None:
                    logger.error(f"No valid candidate selected for chunk {chunk_index + 1}")
                    cleanup_candidate_files(candidates)
                    return chunk_index, None
                
                # Get the audio array from the best candidate
                final_audio = best_candidate['audio_array']
                
                # Apply artifact cleanup
                cleaned_audio = detect_and_fix_audio_artifacts(final_audio, model.sr, None)
                
                # Clean up temporary candidate files
                cleanup_candidate_files(candidates, best_candidate)
                
                logger.info(f"Chunk {chunk_index + 1} processed successfully (seed: {best_candidate.get('seed', 'unknown')})")
                return chunk_index, cleaned_audio
                
            except Exception as e:
                logger.error(f"Error processing text chunk {chunk_index + 1}: {str(e)}")
                return chunk_index, None
        
        # Process text chunks (parallel or sequential)
        processed_chunks = {}
        total_operations = len(text_chunks) + 2  # +2 for assembly and post-processing phases
        
        # Initial progress update (start at 1% to show activity)
        if progress_callback:
            try:
                progress_callback(1, 100)
            except Exception as e:
                logger.warning(f"Initial progress callback error: {str(e)}")
        
        if enable_parallel and num_workers > 1 and len(text_chunks) > 1:
            logger.info(f"Using parallel processing with {num_workers} workers")
            
            if status_callback:
                status_callback(f"⚡ Processing {len(text_chunks)} chunks in parallel...")
            
            with ThreadPoolExecutor(max_workers=min(num_workers, len(text_chunks))) as executor:
                # Submit all chunks for processing
                futures = {
                    executor.submit(process_single_text_chunk, idx, chunk): idx 
                    for idx, chunk in enumerate(text_chunks)
                }
                
                # Collect results as they complete
                completed = 0
                for future in as_completed(futures):
                    chunk_index, audio_result = future.result()
                    processed_chunks[chunk_index] = audio_result
                    completed += 1
                    
                    logger.info(f"Parallel processing: {completed}/{len(text_chunks)} chunks completed")
                    
                    if status_callback:
                        status_callback(f"✅ Completed chunk {completed}/{len(text_chunks)}")
                    
                    # Call progress callback with current progress (10% to 80%)
                    if progress_callback:
                        try:
                            progress_percent = 10 + int((completed / len(text_chunks)) * 70)
                            progress_callback(progress_percent, 100)
                        except Exception as e:
                            logger.warning(f"Progress callback error: {str(e)}")
        
        else:
            logger.info("Using sequential processing")
            
            if status_callback:
                status_callback(f"🔄 Processing {len(text_chunks)} chunks sequentially...")
            
            for idx, chunk in enumerate(text_chunks):
                if status_callback:
                    status_callback(f"🎯 Processing chunk {idx + 1}/{len(text_chunks)}")
                
                chunk_index, audio_result = process_single_text_chunk(idx, chunk)
                processed_chunks[chunk_index] = audio_result
                
                if progress_callback:
                    try:
                        progress_percent = 10 + int(((idx + 1) / len(text_chunks)) * 70)
                        progress_callback(progress_percent, 100)
                    except Exception as e:
                        logger.warning(f"Progress callback error: {str(e)}")
        
        # Rebuild the final audio with original chunk order (text + pauses)
        if status_callback:
            status_callback("🔧 Assembling final audio...")
        
        if progress_callback:
            try:
                progress_callback(85, 100)  # Assembly phase
            except Exception as e:
                logger.warning(f"Progress callback error: {str(e)}")
        
        audio_pieces = []
        sample_rate = model.sr
        text_chunk_index = 0
        
        for i, chunk_dict in enumerate(chunks):
            if chunk_dict['type'] == 'pause':
                # Handle pause chunks
                pause_duration = int(chunk_dict['content'] * sample_rate)
                natural_pause = generate_natural_silence(pause_duration, sample_rate, 'natural')
                audio_pieces.append(natural_pause)
                logger.debug(f"Added pause: {chunk_dict['content']}s")
                
            elif chunk_dict['type'] == 'text':
                # Handle processed text chunks
                if text_chunk_index in processed_chunks and processed_chunks[text_chunk_index] is not None:
                    audio_pieces.append(processed_chunks[text_chunk_index])
                    
                    # Add natural pauses between text chunks when appropriate
                    if i + 1 < len(chunks) and chunks[i + 1]['type'] != 'pause':
                        next_chunk_content = chunks[i + 1].get('content', '') if chunks[i + 1]['type'] == 'text' else None
                        pause_duration = calculate_natural_pause(chunk_dict['content'], next_chunk_content, sample_rate, None)
                        natural_pause = generate_natural_silence(pause_duration, sample_rate, 'natural')
                        audio_pieces.append(natural_pause)
                        logger.debug(f"Added natural pause: {(pause_duration / sample_rate) * 1000:.0f}ms")
                    
                else:
                    logger.warning(f"Text chunk {text_chunk_index} was not processed successfully, skipping")
                
                text_chunk_index += 1
        
        if not audio_pieces:
            logger.error("No audio was generated successfully")
            return None, actual_seed, processed_text

        # Concatenate all audio pieces
        full_audio = np.concatenate(audio_pieces)
        
        if status_callback:
            status_callback("🔊 Processing and normalizing audio...")
        
        # Normalize the full audio
        max_val = np.max(np.abs(full_audio))
        if max_val > 0:
            full_audio = full_audio / max_val

        # Analyze audio quality
        quality_metrics = analyze_audio_quality(full_audio, sample_rate)

        # Process audio based on settings
        processing_sample_rate = sample_rate
        if remove_silence:
            processing_sample_rate = 16000
            full_audio = resample_audio(full_audio, sample_rate, processing_sample_rate)

        logger.info(f'Processing final audio at sample rate: {processing_sample_rate}')

        # Convert to int16 for processing
        final_audio = np.int16(full_audio * 32767)

        if len(final_audio) > 0:

            if remove_silence:
                logger.info('Removing silence from audio...')
                try:
                    frame_duration_ms = 30
                    padding_duration_ms = 300
                    vad = webrtcvad.Vad(3)
                    frames = frame_generator(frame_duration_ms, final_audio.tobytes(), processing_sample_rate)
                    segments = list(vad_collector(processing_sample_rate, frame_duration_ms, padding_duration_ms, vad, frames))
                    
                    if segments:
                        concat_audio = np.concatenate(segments)
                        final_audio = concat_audio
                    else:
                        logger.warning("Voice activity detection removed all audio")
                except Exception as e:
                    logger.warning(f"Silence removal failed: {str(e)}")
            
            # Convert to float for librosa processing
            final_audio_float = final_audio.astype(np.float32) / 32767

            # Note: Noise reduction is now applied after saving the file (FFmpeg-based)

            # Adjust speed using FFmpeg for better quality
            if speed != 1.0:
                logger.info(f'Adjusting audio speed to: {speed}x using FFmpeg')
                
                if status_callback:
                    status_callback(f"⚡ Adjusting speed to {speed}x...")
                
                try:
                    final_audio_float = adjust_speed_ffmpeg(final_audio_float, processing_sample_rate, speed)
                except Exception as e:
                    logger.warning(f"FFmpeg speed adjustment failed, falling back to librosa: {str(e)}")
                    try:
                        final_audio_float = librosa.effects.time_stretch(final_audio_float, rate=speed)
                    except Exception as e2:
                        logger.warning(f"Librosa speed adjustment also failed: {str(e2)}")

            # Adjust pitch using FFmpeg for better quality
            if pitch != 0:
                logger.info(f'Adjusting audio pitch by: {pitch} semitones using FFmpeg')
                
                if status_callback:
                    status_callback(f"🎵 Adjusting pitch by {pitch} semitones...")
                
                try:
                    final_audio_float = adjust_pitch_ffmpeg(final_audio_float, processing_sample_rate, pitch)
                except Exception as e:
                    logger.warning(f"FFmpeg pitch adjustment failed, falling back to librosa: {str(e)}")
                    try:
                        final_audio_float = librosa.effects.pitch_shift(
                            y=final_audio_float, sr=processing_sample_rate, n_steps=pitch
                        )
                    except Exception as e2:
                        logger.warning(f"Librosa pitch adjustment also failed: {str(e2)}")

            # Resample back to original rate if needed
            if remove_silence and processing_sample_rate != sample_rate:
                final_audio_float = resample_audio(final_audio_float, processing_sample_rate, sample_rate)
                processing_sample_rate = sample_rate

            # Convert to tensor for torchaudio saving
            final_audio_tensor = torch.tensor(final_audio_float).unsqueeze(0)

            # Generate output filename and save
            output_filename = f"{uuid.uuid4().hex}.wav"
            output_path = os.path.join("static", "output", output_filename)
            
            if status_callback:
                status_callback("💾 Saving audio file...")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Use torchaudio to save
            ta.save(output_path, final_audio_tensor, sample_rate)

            # Phase 3: Apply post-processing
            if reduce_noise:
                logger.info("Applying FFmpeg noise reduction...")
                
                if status_callback:
                    status_callback("🔇 Reducing background noise...")
                
                try:
                    if noise_reduction_method == "legacy":
                        # Use legacy noisereduce method - reload, process, save
                        success = _apply_noise_reduction_legacy(output_path)
                    else:
                        # Use FFmpeg-based methods
                        success = apply_noise_reduction_ffmpeg(output_path, method=noise_reduction_method, strength=noise_strength)
                    
                    if not success:
                        logger.warning("Noise reduction failed, file-based processing skipped")
                except Exception as e:
                    logger.warning(f"Noise reduction failed: {str(e)}")
                    
            if use_ffmpeg_normalize:
                logger.info("Applying FFMPEG audio normalization...")
                normalize_audio_ffmpeg(
                    output_path, normalize_method, 
                    integrated_loudness, true_peak, loudness_range
                )
            

            # Export to multiple formats if requested
            output_files = []
            for export_format in export_formats:
                if export_format == 'wav':
                    output_files.append(output_filename)
                else:
                    try:
                        logger.info(f"Converting to {export_format.upper()}...")
                        export_filename = output_filename.replace('.wav', f'.{export_format}')
                        export_path = os.path.join("static", "output", export_filename)
                        
                        if convert_audio_format(output_path, export_path, export_format):
                            output_files.append(export_filename)
                            logger.info(f"Successfully exported {export_format.upper()}: {export_filename}")
                        else:
                            logger.error(f"Failed to export {export_format.upper()}")
                    except Exception as e:
                        logger.error(f"Failed to export {export_format.upper()}: {str(e)}")

            # Remove original WAV if it's not in the export formats
            if 'wav' not in export_formats and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                    logger.info("Removed temporary WAV file")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary WAV file: {str(e)}")

            # Final progress update (100%)
            if progress_callback:
                try:
                    progress_callback(100, 100)
                except Exception as e:
                    logger.warning(f"Final progress callback error: {str(e)}")
            
            if status_callback:
                status_callback(f"🎉 Generation complete! Created {len(output_files)} file(s)")

            # Return the first output file as primary (or list if multiple)
            primary_output = output_files[0] if output_files else output_filename
            logger.info(f"Enhanced audio generation completed: {primary_output} (seed: {actual_seed})")
            logger.info(f"Generated formats: {', '.join(export_formats)}")
            return primary_output, actual_seed, processed_text
        
        logger.error("No final audio generated")
        return None, actual_seed, processed_text if 'processed_text' in locals() else text_input

    except Exception as e:
        logger.error(f"Error in enhanced voice generation: {str(e)}")
        return None, seed if seed != 0 else random.randint(1, 999999), text_input
    finally:
        # Clean up memory
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Memory cleanup error: {str(e)}")