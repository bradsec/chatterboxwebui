import nltk
import numpy as np
import re
import os
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
from scipy.stats import zscore
import warnings
import logging
import gc
from typing import Optional, List, Tuple, Callable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.backends.cuda.sdp_kernel.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*past_key_values.*")

def part_filename(filename: str, part_number: int, max_length: int = 80) -> str:
    """Generate safe filename for audio parts"""
    filename = str(part_number) + "_" + filename.replace(" ", "_")
    filename = re.sub(r"[^\w\-_.]", "", filename)
    filename = filename.lower()

    if len(filename) > max_length:
        filename = filename[:max_length]

    return filename

def resample_audio(audio: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio to target sample rate"""
    try:
        return librosa.resample(audio, orig_sr=original_rate, target_sr=target_rate)
    except Exception as e:
        logger.error(f"Error resampling audio: {str(e)}")
        return audio

def make_dir(dir_name: str) -> str:
    """Create directory if it doesn't exist"""
    try:
        os.makedirs(dir_name, exist_ok=True)
        return dir_name
    except Exception as e:
        logger.error(f"Error creating directory {dir_name}: {str(e)}")
        return dir_name

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
        
        sys.stdout.write('1' if is_speech else '0')
        
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write(f'+({ring_buffer[0][0].timestamp})')
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write(f'-({frame.timestamp + frame.duration})')
                triggered = False
                yield np.frombuffer(b''.join([f.bytes for f in voiced_frames]), np.int16)
                ring_buffer.clear()
                voiced_frames = []
    
    if triggered:
        sys.stdout.write(f'-({frame.timestamp + frame.duration})')
    sys.stdout.write('\n')
    
    if voiced_frames:
        yield np.frombuffer(b''.join([f.bytes for f in voiced_frames]), np.int16)

def simple_sentence_split(text: str) -> List[str]:
    """Simple sentence splitting fallback when NLTK is not available"""
    # Split on sentence endings, but be careful with abbreviations and hyphenated words
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

# Global variable to cache NLTK availability
_nltk_available = None

def ensure_nltk_data():
    """Ensure NLTK punkt tokenizer data is available"""
    global _nltk_available
    
    # Return cached result if already checked
    if _nltk_available is not None:
        return _nltk_available
    
    try:
        # Try to find punkt_tab first (newer NLTK versions)
        try:
            nltk.data.find('tokenizers/punkt_tab')
            _nltk_available = True
            logger.info("NLTK punkt_tab tokenizer found")
            return True
        except LookupError:
            pass
        
        # Try to find punkt (older NLTK versions)
        try:
            nltk.data.find('tokenizers/punkt')
            _nltk_available = True
            logger.info("NLTK punkt tokenizer found")
            return True
        except LookupError:
            pass
        
        # Try downloading - set download_dir to ensure it goes to the right place
        logger.info("NLTK punkt tokenizer not found. Attempting download...")
        
        # Get NLTK data path
        import nltk
        try:
            # Try to use the first writable path
            nltk_data_path = None
            for path in nltk.data.path:
                try:
                    # Test if we can write to this path
                    import tempfile
                    test_file = os.path.join(path, '.test_write')
                    os.makedirs(path, exist_ok=True)
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                    nltk_data_path = path
                    break
                except (OSError, PermissionError):
                    continue
            
            if not nltk_data_path:
                # Fallback to user home directory
                nltk_data_path = os.path.expanduser('~/nltk_data')
                os.makedirs(nltk_data_path, exist_ok=True)
            
            logger.info(f"Downloading NLTK data to: {nltk_data_path}")
            
            # Try downloading punkt_tab first
            try:
                nltk.download('punkt_tab', download_dir=nltk_data_path, quiet=True)
                nltk.data.find('tokenizers/punkt_tab')
                logger.info("Successfully downloaded and verified punkt_tab")
                _nltk_available = True
                return True
            except Exception as e:
                logger.warning(f"punkt_tab download failed: {str(e)}")
            
            # Fallback to punkt
            try:
                nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
                nltk.data.find('tokenizers/punkt')
                logger.info("Successfully downloaded and verified punkt")
                _nltk_available = True
                return True
            except Exception as e:
                logger.warning(f"punkt download failed: {str(e)}")
        
        except Exception as e:
            logger.warning(f"NLTK download setup failed: {str(e)}")
        
        # If we get here, NLTK is not available
        logger.warning("NLTK tokenizer unavailable. Using fallback sentence splitter.")
        _nltk_available = False
        return False
        
    except Exception as e:
        logger.error(f"Error checking NLTK availability: {str(e)}")
        _nltk_available = False
        return False

def split_text_into_chunks(text_input: str, max_chunk_length: int = 300) -> Tuple[List[str], List[Optional[float]]]:
    """
    Split text into chunks suitable for Chatterbox TTS processing.
    Pack multiple sentences into chunks up to max_chunk_length characters, ending on complete sentences.
    Handles custom pause markers like [[1.5]] at the end of chunks only.
    """
    # Ensure NLTK data is available
    nltk_available = ensure_nltk_data()
    
    # Clean and preprocess text - preserve hyphens in compound words
    text_input = re.sub(r'\n+', ' ', text_input)  # Replace multiple newlines with space
    text_input = re.sub(r'\.\s*♪', ' ♪', text_input)  # Handle music note punctuation
    
    # Normalize spaces but preserve hyphens in words like "text-to-speech"
    text_input = re.sub(r'\s+', ' ', text_input)  # Multiple spaces to single space
    text_input = text_input.strip()  # Remove leading/trailing whitespace

    # Extract pause markers and their positions
    pause_pattern = r'\[\[([\d.]+)\]\]'
    pause_markers = {}
    
    def replace_pause_marker(match):
        try:
            pause_duration = float(match.group(1))
            marker_id = f"__PAUSE_MARKER_{len(pause_markers)}__"
            pause_markers[marker_id] = pause_duration
            return marker_id
        except ValueError:
            logger.warning(f"Invalid pause duration: {match.group(1)}")
            return match.group(0)
    
    # Replace pause markers with temporary placeholders
    text_with_markers = re.sub(pause_pattern, replace_pause_marker, text_input, flags=re.IGNORECASE)

    # Handle music notes and sentence splitting
    if '♪' in text_with_markers:
        song_parts = re.split(r'♪', text_with_markers)
        sentences = []
        for i, part in enumerate(song_parts):
            part = part.strip()
            if not part:
                continue
            if i % 2 == 0:
                # Non-song parts - split into sentences
                if nltk_available:
                    try:
                        part_sentences = nltk.sent_tokenize(part)
                    except Exception as e:
                        logger.warning(f"NLTK tokenization failed: {str(e)}")
                        part_sentences = simple_sentence_split(part)
                else:
                    part_sentences = simple_sentence_split(part)
                sentences.extend(part_sentences)
            else:
                # Song parts - keep as single units
                sentences.append('♪ ' + part.strip() + ' ♪')
    else:
        # Regular sentence splitting
        if nltk_available:
            try:
                sentences = nltk.sent_tokenize(text_with_markers)
            except Exception as e:
                logger.warning(f"NLTK tokenization failed: {str(e)}")
                sentences = simple_sentence_split(text_with_markers)
        else:
            sentences = simple_sentence_split(text_with_markers)

    # Pack sentences into chunks
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        test_chunk = current_chunk + (" " + sentence if current_chunk else sentence)
        
        if len(test_chunk) <= max_chunk_length:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Handle sentences that are too long
                if len(sentence) > max_chunk_length:
                    # Split by commas first
                    comma_parts = [part.strip() for part in sentence.split(',')]
                    for j, part in enumerate(comma_parts):
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
                                # Split by words as last resort
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
                    current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Process chunks to extract pause markers
    final_chunks = []
    chunk_pauses = []
    
    for chunk in chunks:
        if not chunk.strip():
            continue
            
        custom_pause = None
        clean_chunk = chunk
        
        for marker_id, pause_duration in pause_markers.items():
            if marker_id in chunk:
                custom_pause = pause_duration
                clean_chunk = chunk.replace(marker_id, '').strip()
                break
        
        if clean_chunk:
            final_chunks.append(clean_chunk)
            chunk_pauses.append(custom_pause)

    return final_chunks, chunk_pauses

def detect_and_fix_audio_artifacts(audio: np.ndarray, sample_rate: int, 
                                  chunk_boundaries: Optional[List[int]] = None) -> np.ndarray:
    """Detect and fix common TTS artifacts like clicks, pops, and glitches"""
    logger.info("Analyzing audio for artifacts...")
    
    try:
        cleaned_audio = audio.copy()
        
        # 1. Detect and remove clicks/pops
        cleaned_audio = remove_clicks_pops(cleaned_audio, sample_rate)
        
        # 2. Fix discontinuities at chunk boundaries
        if chunk_boundaries:
            cleaned_audio = fix_chunk_discontinuities(cleaned_audio, chunk_boundaries, sample_rate)
        
        # 3. Remove DC offset
        cleaned_audio = remove_dc_offset(cleaned_audio)
        
        # 4. Detect and fix glitches in quiet sections
        cleaned_audio = fix_quiet_section_glitches(cleaned_audio, sample_rate)
        
        # 5. Smooth sudden amplitude changes
        cleaned_audio = smooth_amplitude_jumps(cleaned_audio, sample_rate)
        
        logger.info("Audio artifact analysis complete.")
        return cleaned_audio
        
    except Exception as e:
        logger.error(f"Error in artifact detection: {str(e)}")
        return audio

def remove_clicks_pops(audio: np.ndarray, sample_rate: int, threshold_factor: float = 5.0) -> np.ndarray:
    """Remove sudden amplitude spikes (clicks/pops)"""
    try:
        window_size = max(3, int(sample_rate * 0.001))  # 1ms window, minimum 3 samples
        
        # Use a more robust outlier detection
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z_scores = np.abs(zscore(audio, nan_policy='omit'))
        
        # Handle NaN values
        z_scores = np.nan_to_num(z_scores, nan=0.0)
        outliers = z_scores > threshold_factor
        
        if np.any(outliers):
            outlier_count = np.sum(outliers)
            logger.info(f"Found {outlier_count} potential clicks/pops, removing...")
            
            outlier_indices = np.where(outliers)[0]
            for idx in outlier_indices:
                start_idx = max(0, idx - window_size)
                end_idx = min(len(audio), idx + window_size)
                
                before = audio[start_idx:idx]
                after = audio[idx+1:end_idx]
                
                if len(before) > 0 and len(after) > 0:
                    audio[idx] = (before[-1] + after[0]) / 2
                elif len(before) > 0:
                    audio[idx] = before[-1]
                elif len(after) > 0:
                    audio[idx] = after[0]
                else:
                    audio[idx] = 0
        
        return audio
    except Exception as e:
        logger.error(f"Error removing clicks/pops: {str(e)}")
        return audio

def fix_chunk_discontinuities(audio: np.ndarray, chunk_boundaries: List[int], sample_rate: int) -> np.ndarray:
    """Fix amplitude discontinuities where audio chunks were joined"""
    try:
        fade_length = int(sample_rate * 0.01)  # 10ms fade
        
        for boundary in chunk_boundaries:
            if boundary < fade_length or boundary >= len(audio) - fade_length:
                continue
                
            # Check for discontinuity
            before_avg = np.mean(np.abs(audio[boundary-fade_length:boundary]))
            after_avg = np.mean(np.abs(audio[boundary:boundary+fade_length]))
            
            # If there's a significant amplitude jump
            if abs(before_avg - after_avg) > 0.1:
                logger.info(f"Fixing discontinuity at sample {boundary}")
                
                # Apply crossfade
                fade_out = np.linspace(1, 0, fade_length)
                fade_in = np.linspace(0, 1, fade_length)
                
                # Blend the audio around the boundary
                before_section = audio[boundary-fade_length:boundary] * fade_out
                after_section = audio[boundary:boundary+fade_length] * fade_in
                
                # Create smooth transition
                transition = before_section + after_section
                audio[boundary-fade_length:boundary+fade_length] = transition
        
        return audio
    except Exception as e:
        logger.error(f"Error fixing chunk discontinuities: {str(e)}")
        return audio

def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """Remove DC bias from audio"""
    try:
        return audio - np.mean(audio)
    except Exception as e:
        logger.error(f"Error removing DC offset: {str(e)}")
        return audio

def fix_quiet_section_glitches(audio: np.ndarray, sample_rate: int, quiet_threshold: float = 0.01) -> np.ndarray:
    """Detect and fix glitches in sections that should be quiet"""
    try:
        window_size = int(sample_rate * 0.1)  # 100ms windows
        if window_size < 1:
            return audio
            
        audio_abs = np.abs(audio)
        hop_size = max(1, window_size // 4)
        rms_values = []
        positions = []
        
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio_abs[i:i+window_size]
            rms = np.sqrt(np.mean(window**2))
            rms_values.append(rms)
            positions.append(i)
        
        if not rms_values:
            return audio
            
        rms_values = np.array(rms_values)
        positions = np.array(positions)
        quiet_mask = rms_values < quiet_threshold
        
        if np.any(quiet_mask):
            logger.info("Checking quiet sections for glitches...")
            
            for i, (pos, is_quiet) in enumerate(zip(positions, quiet_mask)):
                if is_quiet:
                    window_start = pos
                    window_end = min(pos + window_size, len(audio))
                    window_audio = audio[window_start:window_end]
                    
                    spike_threshold = quiet_threshold * 3
                    spikes = np.abs(window_audio) > spike_threshold
                    
                    if np.any(spikes):
                        logger.info(f"Removing glitches in quiet section at {pos/sample_rate:.2f}s")
                        spike_indices = np.where(spikes)[0]
                        for spike_idx in spike_indices:
                            global_idx = window_start + spike_idx
                            if global_idx < len(audio):
                                audio[global_idx] *= 0.1
        
        return audio
    except Exception as e:
        logger.error(f"Error fixing quiet section glitches: {str(e)}")
        return audio

def smooth_amplitude_jumps(audio: np.ndarray, sample_rate: int, threshold: float = 0.5) -> np.ndarray:
    """Smooth sudden amplitude changes that might sound unnatural"""
    try:
        window_size = max(3, int(sample_rate * 0.01))  # 10ms window, minimum 3 samples
        
        audio_squared = audio ** 2
        kernel = np.ones(window_size) / window_size
        rms_envelope = np.sqrt(np.convolve(audio_squared, kernel, mode='same'))
        
        rms_diff = np.diff(rms_envelope)
        sudden_changes = np.abs(rms_diff) > threshold
        
        if np.any(sudden_changes):
            change_count = np.sum(sudden_changes)
            logger.info(f"Smoothing {change_count} sudden amplitude changes...")
            
            change_indices = np.where(sudden_changes)[0]
            for idx in change_indices:
                smooth_window = max(3, int(sample_rate * 0.005))  # 5ms
                start_idx = max(0, idx - smooth_window)
                end_idx = min(len(audio), idx + smooth_window)
                
                if end_idx - start_idx > 2:
                    section = audio[start_idx:end_idx]
                    # Use a simpler smoothing approach if savgol_filter fails
                    try:
                        window_length = min(len(section)//2*2-1, 5)
                        if window_length >= 3:
                            smoothed = signal.savgol_filter(section, window_length, 1, mode='nearest')
                            audio[start_idx:end_idx] = smoothed
                    except Exception as smoothing_error:
                        logger.warning(f"Savgol filter failed, using simple smoothing: {str(smoothing_error)}")
                        # Simple moving average as fallback
                        kernel_size = min(3, len(section))
                        if kernel_size >= 3:
                            kernel = np.ones(kernel_size) / kernel_size
                            audio[start_idx:end_idx] = np.convolve(section, kernel, mode='same')
        
        return audio
    except Exception as e:
        logger.error(f"Error smoothing amplitude jumps: {str(e)}")
        return audio

def analyze_audio_quality(audio: np.ndarray, sample_rate: int) -> dict:
    """Analyze audio quality and provide metrics"""
    try:
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        
        # Avoid division by zero
        if rms > 1e-10:
            dynamic_range = 20 * np.log10(peak / rms)
        else:
            dynamic_range = 0
        
        clipping_threshold = 0.95
        clipped_samples = np.sum(np.abs(audio) > clipping_threshold)
        clipping_percentage = (clipped_samples / len(audio)) * 100
        
        silence_threshold = 0.01
        silent_samples = np.sum(np.abs(audio) < silence_threshold)
        silence_percentage = (silent_samples / len(audio)) * 100
        
        logger.info(f"Audio Quality Metrics:")
        logger.info(f"  RMS Level: {rms:.4f}")
        logger.info(f"  Peak Level: {peak:.4f}")
        logger.info(f"  Dynamic Range: {dynamic_range:.2f} dB")
        logger.info(f"  Clipping: {clipping_percentage:.2f}%")
        logger.info(f"  Silence: {silence_percentage:.2f}%")
        
        return {
            'rms': rms,
            'peak': peak,
            'dynamic_range': dynamic_range,
            'clipping_percentage': clipping_percentage,
            'silence_percentage': silence_percentage
        }
    except Exception as e:
        logger.error(f"Error analyzing audio quality: {str(e)}")
        return {}

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

# Global model instance (loaded once)
_chatterbox_model = None
_model_device = None

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

def generate_voice(text_input: str, audio_prompt_path: Optional[str] = None, 
                  exaggeration: float = 0.5, temperature: float = 0.8, 
                  cfg_weight: float = 0.5, chunk_size: int = 300, 
                  speed: float = 1.0, pitch: int = 0, reduce_noise: bool = False, 
                  remove_silence: bool = False, seed: int = 0, 
                  progress_callback: Optional[Callable] = None) -> Optional[str]:
    """Generate voice audio from text input"""
    try:
        logger.info("Generating text with Chatterbox TTS...")
        logger.info(f"Audio Prompt: {audio_prompt_path}, Exaggeration: {exaggeration}, Temperature: {temperature}")
        logger.info(f"CFG Weight: {cfg_weight}, Chunk Size: {chunk_size}, Speed: {speed}, Pitch: {pitch}, Seed: {seed}")
        logger.info(f"Reduce Noise: {reduce_noise}, Remove Silence: {remove_silence}")
        
        # Set seed for reproducible generation
        if seed != 0:
            set_seed(int(seed))
            logger.info(f"Set random seed to: {seed}")
        
        model, device = get_chatterbox_model()
        
        script = text_input.replace("\n", " ").strip()
        chunks, chunk_pauses = split_text_into_chunks(script, chunk_size)
        
        if not chunks:
            logger.error("No text chunks to process")
            return None
            
        total_parts = len(chunks)
        parts_processed = 0
        audio_pieces = []
        sample_rate = model.sr
        
        # Track chunk boundaries and pause information for artifact detection
        chunk_boundaries = []
        pause_info = []

        for i, chunk in enumerate(chunks):
            parts_processed += 1
            logger.info(f"Processing part {parts_processed} of {total_parts}: {chunk}")

            try:
                # Generate audio with Chatterbox
                if audio_prompt_path and os.path.exists(audio_prompt_path):
                    wav_tensor = model.generate(
                        chunk,
                        audio_prompt_path=audio_prompt_path,
                        exaggeration=exaggeration,
                        temperature=temperature,
                        cfg_weight=cfg_weight
                    )
                else:
                    wav_tensor = model.generate(
                        chunk,
                        exaggeration=exaggeration,
                        temperature=temperature,
                        cfg_weight=cfg_weight
                    )

                # Convert tensor to numpy array
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

                audio_pieces.append(audio_array)

                # Add pause after this chunk (except for the last one)
                if parts_processed != total_parts:
                    custom_pause = chunk_pauses[i] if i < len(chunk_pauses) else None
                    next_chunk = chunks[i + 1] if i + 1 < len(chunks) else None
                    pause_duration = calculate_natural_pause(chunk, next_chunk, sample_rate, custom_pause)
                    
                    natural_pause = generate_natural_silence(pause_duration, sample_rate, 'natural')
                    audio_pieces.append(natural_pause)
                    
                    # Track pause info for logging
                    pause_ms = (pause_duration / sample_rate) * 1000
                    if custom_pause is not None:
                        pause_info.append(f"Custom pause after '{chunk[:30]}...': {pause_ms:.0f}ms (requested {custom_pause}s)")
                    else:
                        pause_info.append(f"Natural pause after '{chunk[:30]}...': {pause_ms:.0f}ms")
                    
                    # Track chunk boundary for artifact detection
                    current_length = sum(len(piece) for piece in audio_pieces[:-1])
                    chunk_boundaries.append(current_length)

                # Clean up GPU memory
                if isinstance(wav_tensor, torch.Tensor):
                    del wav_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error generating audio for chunk {parts_processed}: {str(e)}")
                continue

            if progress_callback:
                try:
                    progress_callback(parts_processed, total_parts)
                except Exception as e:
                    logger.warning(f"Progress callback error: {str(e)}")

        if not audio_pieces:
            logger.error("No audio was generated successfully")
            return None

        # Print pause information for debugging
        if pause_info:
            logger.info("Pause timing:")
            for info in pause_info[:5]:  # Show first 5 pauses
                logger.info(f"  {info}")
            if len(pause_info) > 5:
                logger.info(f"  ... and {len(pause_info) - 5} more pauses")

        # Concatenate all audio pieces
        full_audio = np.concatenate(audio_pieces)
        
        # Normalize the full audio
        max_val = np.max(np.abs(full_audio))
        if max_val > 0:
            full_audio = full_audio / max_val

        # Apply artifact detection and cleanup
        logger.info("Applying audio artifact detection and cleanup...")
        full_audio = detect_and_fix_audio_artifacts(full_audio, sample_rate, chunk_boundaries)
        
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
            final_audio_int16 = np.frombuffer(final_audio.tobytes(), dtype=np.int16)
            
            if reduce_noise:
                logger.info('Running noise reduction on audio...')
                try:
                    final_audio = nr.reduce_noise(y=final_audio_int16, sr=processing_sample_rate)
                except Exception as e:
                    logger.warning(f"Noise reduction failed: {str(e)}")
                    final_audio = final_audio_int16
            else:
                final_audio = final_audio_int16

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

            # Adjust speed
            if speed != 1.0:
                logger.info(f'Adjusting audio speed to: {speed}')
                try:
                    final_audio_float = librosa.effects.time_stretch(final_audio_float, rate=speed)
                except Exception as e:
                    logger.warning(f"Speed adjustment failed: {str(e)}")

            # Adjust pitch
            if pitch != 0:
                logger.info(f'Adjusting audio pitch by: {pitch}')
                try:
                    final_audio_float = librosa.effects.pitch_shift(
                        y=final_audio_float, sr=processing_sample_rate, n_steps=pitch
                    )
                except Exception as e:
                    logger.warning(f"Pitch adjustment failed: {str(e)}")

            # Resample back to original rate if needed
            if remove_silence and processing_sample_rate != sample_rate:
                final_audio_float = resample_audio(final_audio_float, processing_sample_rate, sample_rate)
                processing_sample_rate = sample_rate

            # Convert to tensor for torchaudio saving
            final_audio_tensor = torch.tensor(final_audio_float).unsqueeze(0)

            # Generate output filename and save
            output_filename = f"{uuid.uuid4().hex}.wav"
            output_path = os.path.join("static", "output", output_filename)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Use torchaudio to save
            ta.save(output_path, final_audio_tensor, sample_rate)

            logger.info(f"Audio generation completed: {output_filename}")
            return output_filename
        
        logger.error("No final audio generated")
        return None

    except Exception as e:
        logger.error(f"Error in voice generation: {str(e)}")
        return None
    finally:
        # Clean up memory
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Memory cleanup error: {str(e)}")