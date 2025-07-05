"""
Audio Processing Module

This module contains higher-level audio processing functions including
normalization, noise reduction, and advanced audio processing workflows.

Functions:
- normalize_audio_ffmpeg: Normalize audio using FFmpeg
- apply_noise_reduction: Apply noise reduction to audio
"""

import os
import subprocess
import logging
import numpy as np
import noisereduce as nr

logger = logging.getLogger(__name__)

# Check if ffmpeg-python is available
try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    logger.warning("ffmpeg-python not available for normalization")


def apply_noise_reduction_ffmpeg(input_path: str, method: str = "afftdn", 
                               strength: float = 0.85) -> bool:
    """Apply noise reduction using FFmpeg filters for better quality"""
    if not os.path.exists(input_path):
        logger.error(f"Input file does not exist: {input_path}")
        return False
    
    temp_output = input_path + ".denoised.wav"
    
    try:
        logger.info(f"Applying FFmpeg noise reduction using {method} filter...")
        
        if method == "afftdn":
            # Adaptive FFT denoiser - best for general noise reduction
            cmd = [
                "ffmpeg", "-i", input_path, "-y",
                "-af", f"afftdn=nr={strength}:nf=-25:tn=1",
                temp_output
            ]
        elif method == "anlmdn":
            # Non-local means denoiser - good for preserving speech quality
            cmd = [
                "ffmpeg", "-i", input_path, "-y", 
                "-af", f"anlmdn=s={strength}:p=0.002:r=0.006",
                temp_output
            ]
        elif method == "arnndn":
            # RNN denoiser - AI-based, very effective but requires model
            cmd = [
                "ffmpeg", "-i", input_path, "-y",
                "-af", f"arnndn=m=./models/rnnoise.rnnn",
                temp_output
            ]
            # Fallback to afftdn if rnndn model not available
        else:
            # Default to afftdn
            cmd = [
                "ffmpeg", "-i", input_path, "-y",
                "-af", f"afftdn=nr={strength}:nf=-25:tn=1",
                temp_output
            ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0 and os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
            # Replace original with processed version
            os.replace(temp_output, input_path)
            logger.info(f"FFmpeg noise reduction completed using {method}")
            return True
        else:
            logger.error(f"FFmpeg noise reduction failed: {result.stderr}")
            # Fallback to legacy method if FFmpeg fails
            return _apply_noise_reduction_legacy(input_path)
            
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg noise reduction timed out")
        return _apply_noise_reduction_legacy(input_path)
    except Exception as e:
        logger.error(f"Error in FFmpeg noise reduction: {str(e)}")
        return _apply_noise_reduction_legacy(input_path)
    finally:
        # Clean up temp file if it exists
        if os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except:
                pass

def _apply_noise_reduction_legacy(input_path: str) -> bool:
    """Fallback to legacy noisereduce library when FFmpeg fails"""
    try:
        import librosa
        import torchaudio as ta
        import torch
        
        logger.info("Falling back to legacy noise reduction...")
        
        # Load audio
        audio_tensor, sr = ta.load(input_path)
        audio_array = audio_tensor.squeeze().numpy()
        
        if len(audio_array) == 0:
            return False
        
        # Apply noisereduce
        reduced_noise = nr.reduce_noise(
            y=audio_array,
            sr=sr,
            stationary=True,
            prop_decrease=0.85
        )
        
        # Save processed audio back
        processed_tensor = torch.tensor(reduced_noise).unsqueeze(0)
        ta.save(input_path, processed_tensor, sr)
        
        logger.info("Legacy noise reduction complete")
        return True
        
    except Exception as e:
        logger.error(f"Legacy noise reduction also failed: {str(e)}")
        return False

def apply_noise_reduction(audio: np.ndarray, sample_rate: int, 
                         stationary: bool = True, prop_decrease: float = 1.0) -> np.ndarray:
    """Apply noise reduction to audio using noisereduce library (legacy method)"""
    try:
        if len(audio) == 0:
            return audio
        
        logger.info("Applying legacy noise reduction...")
        reduced_noise = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            stationary=stationary,
            prop_decrease=prop_decrease
        )
        logger.info("Legacy noise reduction complete")
        return reduced_noise.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Error in legacy noise reduction: {str(e)}")
        return audio


def normalize_audio_ffmpeg(input_path: str, method: str = "ebu", 
                          integrated_loudness: int = -24, true_peak: int = -2, 
                          loudness_range: int = 7) -> bool:
    """Normalize audio using FFmpeg with different methods"""
    if not FFMPEG_AVAILABLE:
        logger.warning("FFmpeg not available for normalization")
        return False
        
    if not os.path.exists(input_path):
        logger.error(f"Input file does not exist: {input_path}")
        return False
    
    temp_output = input_path + ".temp.wav"
    
    try:
        if method == "ebu":
            loudnorm = f"loudnorm=I={integrated_loudness}:TP={true_peak}:LRA={loudness_range}"
            stream = ffmpeg.input(input_path).output(temp_output, af=loudnorm)
        elif method == "peak":
            stream = ffmpeg.input(input_path).output(temp_output, af="dynaudnorm")
        else:
            logger.error(f"Unknown normalization method: {method}")
            return False
        
        # Run ffmpeg
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        
        # Replace original with normalized version
        if os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
            os.replace(temp_output, input_path)
            logger.info(f"Audio normalized using {method} method")
            return True
        else:
            logger.error("FFMPEG normalization failed - output file not created or empty")
            return False
            
    except Exception as e:
        logger.error(f"Error in FFMPEG normalization: {str(e)}")
        # Clean up temp file if it exists
        if os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except:
                pass
        return False


