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

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.backends.cuda.sdp_kernel.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*past_key_values.*")


def part_filename(filename, part_number, max_length=80):
    filename = str(part_number) + "_" + filename.replace(" ", "_")
    filename = re.sub(r"[^\w]", "", filename)
    filename = filename.lower()

    if len(filename) > max_length:
        filename = filename[:max_length]

    return filename

def resample_audio(audio, original_rate, target_rate):
    return librosa.resample(audio, orig_sr=original_rate, target_sr=target_rate)

def make_dir(dir_name):
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

def write_wave(path, audio, sample_rate):
    audio = audio.astype(np.int16).tostring()
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)

class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield np.frombuffer(b''.join([f.bytes for f in voiced_frames]), np.int16)
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    if voiced_frames:
        yield np.frombuffer(b''.join([f.bytes for f in voiced_frames]), np.int16)

def split_text_into_chunks(text_input, max_chunk_length=300):
    """
    Split text into chunks suitable for Chatterbox TTS processing.
    Pack multiple sentences into chunks up to 300 characters, ending on complete sentences.
    Handles custom pause markers like [[1.5]] at the end of chunks only.
    """
    # Ensure NLTK data is available
    try:
        import nltk
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            try:
                nltk.download('punkt_tab', quiet=True)
            except:
                nltk.download('punkt', quiet=True)
    
    # Replace multiple newline characters with a single space
    text_input = re.sub(r'\n+', ' ', text_input)
    # Remove a period if it's directly followed by any number of spaces and a musical note
    text_input = re.sub(r'\.\s*♪', ' ♪', text_input)

    # Extract pause markers and their positions
    pause_pattern = r'\[\[([\d.]+)\]\]'
    pause_markers = {}
    
    def replace_pause_marker(match):
        pause_duration = float(match.group(1))
        marker_id = f"__PAUSE_MARKER_{len(pause_markers)}__"
        pause_markers[marker_id] = pause_duration
        return marker_id
    
    # Replace pause markers with temporary placeholders
    text_with_markers = re.sub(pause_pattern, replace_pause_marker, text_input, flags=re.IGNORECASE)

    # First, split into sentences
    if '♪' in text_with_markers:
        # Handle music notes - treat as separate chunks
        song_parts = re.split(r'♪', text_with_markers)
        sentences = []
        for i, part in enumerate(song_parts):
            part = part.strip()
            if not part:
                continue
            if i % 2 == 0:
                # Non-song parts - split into sentences
                try:
                    part_sentences = nltk.sent_tokenize(part)
                except:
                    part_sentences = [s.strip() + '.' for s in part.split('.') if s.strip()]
                sentences.extend(part_sentences)
            else:
                # Song parts - keep as single units
                sentences.append('♪ ' + part.strip() + ' ♪')
    else:
        # Regular sentence splitting
        try:
            sentences = nltk.sent_tokenize(text_with_markers)
        except:
            sentences = [s.strip() for s in text_with_markers.split('.') if s.strip()]
            # Add periods back to sentences that don't already end with punctuation
            sentences = [s + '.' if not s.endswith(('.', '!', '?', '...')) else s for s in sentences]

    # Now pack sentences into chunks up to max_chunk_length
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Check if adding this sentence would exceed the limit
        test_chunk = current_chunk + (" " + sentence if current_chunk else sentence)
        
        if len(test_chunk) <= max_chunk_length:
            # Sentence fits, add it to current chunk
            current_chunk = test_chunk
        else:
            # Sentence doesn't fit
            if current_chunk:
                # Save the current chunk and start a new one
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Even a single sentence is too long, need to split it
                if len(sentence) > max_chunk_length:
                    # Split long sentence by commas first
                    comma_parts = [part.strip() for part in sentence.split(',')]
                    for j, part in enumerate(comma_parts):
                        # Add comma back except for last part
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
                                # Even a comma part is too long, split by words
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
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Process chunks to extract pause markers and associate them with chunks
    final_chunks = []
    chunk_pauses = []
    
    for chunk in chunks:
        if not chunk.strip():
            continue
            
        # Check if this chunk contains a pause marker
        custom_pause = None
        clean_chunk = chunk
        
        for marker_id, pause_duration in pause_markers.items():
            if marker_id in chunk:
                custom_pause = pause_duration
                # Remove the marker from the chunk
                clean_chunk = chunk.replace(marker_id, '').strip()
                break
        
        # Only add non-empty chunks
        if clean_chunk:
            final_chunks.append(clean_chunk)
            chunk_pauses.append(custom_pause)

    return final_chunks, chunk_pauses

def detect_and_fix_audio_artifacts(audio, sample_rate, chunk_boundaries=None):
    """
    Detect and fix common TTS artifacts like clicks, pops, and glitches.
    """
    print("Analyzing audio for artifacts...")
    
    # Make a copy to avoid modifying original
    cleaned_audio = audio.copy()
    
    # 1. Detect and remove clicks/pops (sudden amplitude spikes)
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
    
    print("Audio artifact analysis complete.")
    return cleaned_audio

def remove_clicks_pops(audio, sample_rate, threshold_factor=5.0):
    """Remove sudden amplitude spikes (clicks/pops)"""
    window_size = int(sample_rate * 0.001)  # 1ms window
    if window_size < 3:
        window_size = 3
    
    # Detect outliers using z-score
    z_scores = np.abs(zscore(audio, nan_policy='omit'))
    outliers = z_scores > threshold_factor
    
    if np.any(outliers):
        print(f"Found {np.sum(outliers)} potential clicks/pops, removing...")
        
        # Replace outliers with interpolated values
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

def fix_chunk_discontinuities(audio, chunk_boundaries, sample_rate):
    """Fix amplitude discontinuities where audio chunks were joined"""
    fade_length = int(sample_rate * 0.01)  # 10ms fade
    
    for boundary in chunk_boundaries:
        if boundary < fade_length or boundary >= len(audio) - fade_length:
            continue
            
        # Check for discontinuity
        before_avg = np.mean(np.abs(audio[boundary-fade_length:boundary]))
        after_avg = np.mean(np.abs(audio[boundary:boundary+fade_length]))
        
        # If there's a significant amplitude jump
        if abs(before_avg - after_avg) > 0.1:
            print(f"Fixing discontinuity at sample {boundary}")
            
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

def remove_dc_offset(audio):
    """Remove DC bias from audio"""
    return audio - np.mean(audio)

def fix_quiet_section_glitches(audio, sample_rate, quiet_threshold=0.01):
    """Detect and fix glitches in sections that should be quiet"""
    window_size = int(sample_rate * 0.1)  # 100ms windows
    audio_abs = np.abs(audio)
    
    hop_size = window_size // 4
    rms_values = []
    positions = []
    
    for i in range(0, len(audio) - window_size, hop_size):
        window = audio_abs[i:i+window_size]
        rms = np.sqrt(np.mean(window**2))
        rms_values.append(rms)
        positions.append(i)
    
    rms_values = np.array(rms_values)
    positions = np.array(positions)
    
    quiet_mask = rms_values < quiet_threshold
    
    if np.any(quiet_mask):
        print("Checking quiet sections for glitches...")
        
        for i, (pos, is_quiet) in enumerate(zip(positions, quiet_mask)):
            if is_quiet:
                window_start = pos
                window_end = min(pos + window_size, len(audio))
                window_audio = audio[window_start:window_end]
                
                spike_threshold = quiet_threshold * 3
                spikes = np.abs(window_audio) > spike_threshold
                
                if np.any(spikes):
                    print(f"Removing glitches in quiet section at {pos/sample_rate:.2f}s")
                    spike_indices = np.where(spikes)[0]
                    for spike_idx in spike_indices:
                        global_idx = window_start + spike_idx
                        if global_idx < len(audio):
                            audio[global_idx] *= 0.1
    
    return audio

def smooth_amplitude_jumps(audio, sample_rate, threshold=0.5):
    """Smooth sudden amplitude changes that might sound unnatural"""
    window_size = int(sample_rate * 0.01)  # 10ms window
    if window_size < 3:
        return audio
    
    audio_squared = audio ** 2
    kernel = np.ones(window_size) / window_size
    rms_envelope = np.sqrt(np.convolve(audio_squared, kernel, mode='same'))
    
    rms_diff = np.diff(rms_envelope)
    sudden_changes = np.abs(rms_diff) > threshold
    
    if np.any(sudden_changes):
        print(f"Smoothing {np.sum(sudden_changes)} sudden amplitude changes...")
        
        change_indices = np.where(sudden_changes)[0]
        for idx in change_indices:
            smooth_window = int(sample_rate * 0.005)  # 5ms
            start_idx = max(0, idx - smooth_window)
            end_idx = min(len(audio), idx + smooth_window)
            
            if end_idx - start_idx > 2:
                section = audio[start_idx:end_idx]
                smoothed = signal.savgol_filter(section, 
                                              min(len(section)//2*2-1, 5), 
                                              1, 
                                              mode='nearest')
                audio[start_idx:end_idx] = smoothed
    
    return audio

def analyze_audio_quality(audio, sample_rate):
    """Analyze audio quality and provide metrics"""
    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))
    dynamic_range = 20 * np.log10(peak / (rms + 1e-10))
    
    clipping_threshold = 0.95
    clipped_samples = np.sum(np.abs(audio) > clipping_threshold)
    clipping_percentage = (clipped_samples / len(audio)) * 100
    
    silence_threshold = 0.01
    silent_samples = np.sum(np.abs(audio) < silence_threshold)
    silence_percentage = (silent_samples / len(audio)) * 100
    
    print(f"Audio Quality Metrics:")
    print(f"  RMS Level: {rms:.4f}")
    print(f"  Peak Level: {peak:.4f}")
    print(f"  Dynamic Range: {dynamic_range:.2f} dB")
    print(f"  Clipping: {clipping_percentage:.2f}%")
    print(f"  Silence: {silence_percentage:.2f}%")
    
    return {
        'rms': rms,
        'peak': peak,
        'dynamic_range': dynamic_range,
        'clipping_percentage': clipping_percentage,
        'silence_percentage': silence_percentage
    }

def calculate_natural_pause(current_chunk, next_chunk=None, sample_rate=24000, custom_pause=None):
    """
    Calculate natural pause duration based on sentence structure and content.
    Returns pause duration in samples.
    """
    # If there's a custom pause specified, use it
    if custom_pause is not None:
        return int(custom_pause * sample_rate)
    
    # Increased base pause duration (400ms)
    base_pause = int(0.4 * sample_rate)
    
    # Analyze current chunk ending
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
        if word_count > 15:
            pause_multiplier = 1.5
        else:
            pause_multiplier = 1.2
    elif current_chunk.endswith(','):
        pause_multiplier = 0.8
    elif current_chunk.endswith(';') or current_chunk.endswith(':'):
        pause_multiplier = 1.4
    else:
        pause_multiplier = 0.6
    
    # Analyze next chunk beginning (if available)
    if next_chunk:
        next_chunk = next_chunk.strip()
        question_starters = ['what', 'when', 'where', 'why', 'who', 'how', 'which', 'whose']
        emphasis_words = ['but', 'however', 'meanwhile', 'furthermore', 'moreover', 'therefore']
        
        first_word = next_chunk.split()[0].lower() if next_chunk.split() else ''
        
        if first_word in question_starters:
            pause_multiplier *= 1.3
        elif first_word in emphasis_words:
            pause_multiplier *= 1.4
        elif next_chunk.startswith('"') or next_chunk.startswith("'"):
            pause_multiplier *= 1.2
    
    # Add natural randomness (±15%)
    randomness = np.random.uniform(0.85, 1.15)
    pause_multiplier *= randomness
    
    # Calculate final pause duration
    pause_duration = int(base_pause * pause_multiplier)
    
    # Ensure reasonable bounds (100ms to 1500ms)
    min_pause = int(0.1 * sample_rate)
    max_pause = int(1.5 * sample_rate)
    pause_duration = max(min_pause, min(max_pause, pause_duration))
    
    return pause_duration

def generate_natural_silence(duration_samples, sample_rate=24000, silence_type='natural'):
    """
    Generate more natural silence with subtle ambient characteristics.
    """
    if silence_type == 'natural':
        room_tone_level = 0.0001  # Very quiet
        room_tone = np.random.normal(0, room_tone_level, duration_samples)
        
        # Add very subtle low-frequency rumble (like distant HVAC)
        if duration_samples > sample_rate * 0.1:  # Only for pauses longer than 100ms
            t = np.linspace(0, duration_samples / sample_rate, duration_samples)
            subtle_rumble = 0.00005 * np.sin(2 * np.pi * 60 * t)  # 60Hz rumble
            room_tone += subtle_rumble
        
        return room_tone.astype(np.float32)
    else:
        return np.zeros(duration_samples, dtype=np.float32)

def initialize_chatterbox_model():
    """Initialize Chatterbox TTS model with device detection"""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Initializing Chatterbox TTS model on device: {device}")
    
    # For MPS devices, need to handle torch.load properly
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

# Global model instance (loaded once)
_chatterbox_model = None
_model_device = None

def get_chatterbox_model():
    """Get or initialize the global Chatterbox model instance"""
    global _chatterbox_model, _model_device
    if _chatterbox_model is None:
        _chatterbox_model, _model_device = initialize_chatterbox_model()
    return _chatterbox_model, _model_device

def set_seed(seed: int):
    """Set random seed for reproducible generation"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def generate_voice(text_input, audio_prompt_path=None, exaggeration=0.5, temperature=0.8, cfg_weight=0.5, 
                  chunk_size=300, speed=1.0, pitch=0, reduce_noise=False, remove_silence=False, seed=0, progress_callback=None):
    print("Generating text with Chatterbox TTS...")
    print(f"Audio Prompt: {audio_prompt_path}, Exaggeration: {exaggeration}, Temperature: {temperature}")
    print(f"CFG Weight: {cfg_weight}, Chunk Size: {chunk_size}, Speed: {speed}, Pitch: {pitch}, Seed: {seed}")
    print(f"Reduce Noise: {reduce_noise}, Remove Silence: {remove_silence}")
    
    # Set seed for reproducible generation
    if seed != 0:
        set_seed(int(seed))
        print(f"Set random seed to: {seed}")
    
    model, device = get_chatterbox_model()
    
    script = text_input.replace("\n", " ").strip()
    chunks, chunk_pauses = split_text_into_chunks(script, chunk_size)
    output_filename = None
    total_parts = len(chunks)
    parts_processed = 0
    audio_pieces = []
    
    # Chatterbox native sample rate
    sample_rate = model.sr
    
    # Track chunk boundaries and pause information for artifact detection
    chunk_boundaries = []
    pause_info = []

    for i, chunk in enumerate(chunks):
        parts_processed += 1
        print(f"Processing part {parts_processed} of {total_parts}: {chunk}")

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

            # Ensure audio is in the right format
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)

            # Normalize
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array))

            audio_pieces.append(audio_array)

            # Add pause after this chunk (except for the last one)
            if parts_processed != total_parts:
                # Get custom pause for this chunk (if any)
                custom_pause = chunk_pauses[i] if i < len(chunk_pauses) else None
                
                # Calculate pause duration
                next_chunk = chunks[i + 1] if i + 1 < len(chunks) else None
                pause_duration = calculate_natural_pause(chunk, next_chunk, sample_rate, custom_pause)
                
                # Generate natural silence
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

        except Exception as e:
            print(f"Error generating audio for chunk {parts_processed}: {str(e)}")
            continue

        if progress_callback:
            progress_callback(parts_processed, total_parts)

    if not audio_pieces:
        print("No audio was generated successfully")
        return None

    # Print pause information for debugging
    if pause_info:
        print("Pause timing:")
        for info in pause_info[:5]:  # Show first 5 pauses
            print(f"  {info}")
        if len(pause_info) > 5:
            print(f"  ... and {len(pause_info) - 5} more pauses")

    # Concatenate all audio pieces
    full_audio = np.concatenate(audio_pieces)
    
    # Normalize the full audio
    if np.max(np.abs(full_audio)) > 0:
        full_audio = full_audio / np.max(np.abs(full_audio))

    # Apply artifact detection and cleanup
    print("Applying audio artifact detection and cleanup...")
    full_audio = detect_and_fix_audio_artifacts(full_audio, sample_rate, chunk_boundaries)
    
    # Analyze audio quality
    quality_metrics = analyze_audio_quality(full_audio, sample_rate)

    # For remove_silence, we need to resample to 16kHz for webrtcvad
    processing_sample_rate = sample_rate
    if remove_silence:
        processing_sample_rate = 16000
        full_audio = resample_audio(full_audio, sample_rate, processing_sample_rate)

    print(f'Processing final audio at sample rate: {processing_sample_rate}')

    # Convert to int16 for processing
    final_audio = np.int16(full_audio * 32767)

    if len(final_audio) > 0:
        final_audio_int16 = np.frombuffer(final_audio, dtype=np.int16)
        
        if reduce_noise:
            print(f'Running noise reduction on audio...')
            final_audio = nr.reduce_noise(y=final_audio_int16, sr=processing_sample_rate)
        else:
            final_audio = final_audio_int16

        if remove_silence:
            print(f'Removing silence from audio...')
            frame_duration_ms = 30
            padding_duration_ms = 300
            vad = webrtcvad.Vad(3)
            frames = frame_generator(frame_duration_ms, final_audio, processing_sample_rate)
            segments = vad_collector(processing_sample_rate, frame_duration_ms, padding_duration_ms, vad, frames)
            concataudio = [segment for segment in segments]
            if concataudio:
                final_audio = b"".join(concataudio)
                final_audio = np.frombuffer(final_audio, dtype=np.int16)
            else:
                print("Warning: Voice activity detection removed all audio")
        
        # Convert to float for librosa processing
        final_audio_float = final_audio.astype(np.float32) / 32767

        # Adjust speed for the full audio
        if speed != 1.0:
            print(f'Adjusting audio speed to: {speed}')
            final_audio_float = librosa.effects.time_stretch(final_audio_float, rate=speed)

        # Adjust pitch for the full audio
        if pitch != 0:
            print(f'Adjusting audio pitch by: {pitch}')
            final_audio_float = librosa.effects.pitch_shift(y=final_audio_float, sr=processing_sample_rate, n_steps=pitch)

        # If we resampled for silence removal, resample back to original rate for final output
        if remove_silence and processing_sample_rate != sample_rate:
            final_audio_float = resample_audio(final_audio_float, processing_sample_rate, sample_rate)
            processing_sample_rate = sample_rate

        # Convert to tensor for torchaudio saving
        final_audio_tensor = torch.tensor(final_audio_float).unsqueeze(0)

        output_filename = f"{uuid.uuid4().hex}.wav"
        output_path = os.path.join("static", "output", output_filename)
        
        # Use torchaudio to save (maintaining Chatterbox's native format)
        ta.save(output_path, final_audio_tensor, sample_rate)

    print(f"Audio generation completed: {output_filename}")
    return output_filename