"""
Audio Transforms Module

This module contains functions for transforming audio properties such as
speed, pitch, and format conversions.

Functions:
- adjust_speed_ffmpeg: Adjust audio speed using FFmpeg
- adjust_pitch_ffmpeg: Adjust audio pitch using FFmpeg
- resample_audio: Resample audio to different sample rates
- convert_audio_format: Convert audio between different formats
"""

import os
import tempfile
import subprocess
import numpy as np
import torch
import torchaudio as ta
import librosa
import logging

logger = logging.getLogger(__name__)


def resample_audio(audio: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio to target sample rate"""
    try:
        return librosa.resample(audio, orig_sr=original_rate, target_sr=target_rate)
    except Exception as e:
        logger.error(f"Error resampling audio: {str(e)}")
        return audio


def adjust_speed_ffmpeg(audio: np.ndarray, sample_rate: int, speed: float) -> np.ndarray:
    """Adjust audio speed using FFmpeg for YouTube-like quality without pitch change"""
    try:
        if speed == 1.0:
            return audio.astype(np.float32)
        
        # Create temporary files
        input_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        output_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        
        # Save input audio to temp file
        audio_tensor = torch.tensor(audio).unsqueeze(0)
        ta.save(input_temp.name, audio_tensor, sample_rate)
        input_temp.close()
        
        # Build FFmpeg command for time stretching with tempo filter
        cmd = [
            "ffmpeg", "-i", input_temp.name, "-y",
            "-filter:a", f"atempo={speed}",
            output_temp.name
        ]
        
        # Run FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0 and os.path.exists(output_temp.name) and os.path.getsize(output_temp.name) > 0:
            # Load processed audio
            processed_audio, _ = ta.load(output_temp.name)
            processed_audio = processed_audio.squeeze().numpy()
            
            # Clean up temp files
            os.unlink(input_temp.name)
            os.unlink(output_temp.name)
            
            logger.info(f"FFmpeg speed adjustment successful: {speed}x")
            return processed_audio.astype(np.float32)
        else:
            error_msg = result.stderr or result.stdout or "Unknown FFmpeg error"
            logger.error(f"FFmpeg speed adjustment failed: {error_msg}")
            raise RuntimeError(f"FFmpeg failed: {error_msg}")
            
    except Exception as e:
        # Clean up temp files on error
        try:
            if 'input_temp' in locals():
                os.unlink(input_temp.name)
            if 'output_temp' in locals():
                os.unlink(output_temp.name)
        except:
            pass
        logger.error(f"Error in FFmpeg speed adjustment: {str(e)}")
        raise


def adjust_pitch_ffmpeg(audio: np.ndarray, sample_rate: int, pitch_semitones: int) -> np.ndarray:
    """Adjust audio pitch using FFmpeg for better quality"""
    try:
        # Create temporary files
        input_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        output_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        
        # Save input audio to temp file
        audio_tensor = torch.tensor(audio).unsqueeze(0)
        ta.save(input_temp.name, audio_tensor, sample_rate)
        input_temp.close()
        
        # Build FFmpeg command for pitch adjustment
        cmd = ["ffmpeg", "-i", input_temp.name, "-y"]
        
        if pitch_semitones != 0:
            # Try different pitch shifting methods in order of preference
            pitch_methods = [
                # Method 1: Try rubberband filter (if available) - best quality
                f"rubberband=pitch={2.0 ** (pitch_semitones / 12.0)}",
                # Method 2: Try afreqshift - good for small changes
                f"afreqshift=shift={pitch_semitones * 50}",
            ]
            
            success = False
            for i, pitch_filter in enumerate(pitch_methods):
                try:
                    test_cmd = cmd + ["-filter:a", pitch_filter, output_temp.name]
                    result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=60)
                    
                    if result.returncode == 0 and os.path.exists(output_temp.name) and os.path.getsize(output_temp.name) > 0:
                        logger.info(f"FFmpeg pitch adjustment successful using method {i+1}: {pitch_semitones} semitones")
                        success = True
                        break
                    else:
                        error_msg = result.stderr or result.stdout or "Unknown error"
                        logger.debug(f"FFmpeg pitch method {i+1} failed: {error_msg}")
                        # Remove failed output file
                        if os.path.exists(output_temp.name):
                            os.unlink(output_temp.name)
                            
                except Exception as method_error:
                    logger.debug(f"FFmpeg pitch method {i+1} error: {str(method_error)}")
                    continue
            
            if not success:
                # All FFmpeg methods failed, fall back to librosa
                logger.warning("FFmpeg pitch adjustment methods failed, falling back to librosa")
                raise RuntimeError("FFmpeg pitch adjustment failed, using librosa fallback")
        else:
            # No pitch change needed, just copy the file
            import shutil
            shutil.copy2(input_temp.name, output_temp.name)
            success = True
        
        if success:
            # Load processed audio
            processed_audio, _ = ta.load(output_temp.name)
            processed_audio = processed_audio.squeeze().numpy()
            
            # Clean up temp files
            os.unlink(input_temp.name)
            os.unlink(output_temp.name)
            
            return processed_audio.astype(np.float32)
        else:
            raise RuntimeError("All pitch adjustment methods failed")
            
    except Exception as e:
        # Clean up temp files on error
        try:
            if 'input_temp' in locals():
                os.unlink(input_temp.name)
            if 'output_temp' in locals():
                os.unlink(output_temp.name)
        except:
            pass
        logger.error(f"Error in FFmpeg pitch adjustment: {str(e)}")
        raise


def convert_audio_format(input_path: str, output_path: str, target_format: str = 'wav', 
                        sample_rate: int = None, bitrate: str = None) -> bool:
    """Convert audio between different formats using FFmpeg"""
    try:
        if not os.path.exists(input_path):
            logger.error(f"Input file does not exist: {input_path}")
            return False
        
        cmd = ["ffmpeg", "-i", input_path, "-y"]
        
        # Add sample rate if specified
        if sample_rate:
            cmd.extend(["-ar", str(sample_rate)])
        
        # Add bitrate for compressed formats
        if bitrate and target_format.lower() in ['mp3', 'aac', 'ogg']:
            cmd.extend(["-b:a", bitrate])
        
        # Add format-specific options
        if target_format.lower() == 'mp3':
            cmd.extend(["-codec:a", "libmp3lame"])
        elif target_format.lower() == 'flac':
            cmd.extend(["-codec:a", "flac"])
        elif target_format.lower() == 'ogg':
            cmd.extend(["-codec:a", "libvorbis"])
        elif target_format.lower() == 'wav':
            cmd.extend(["-codec:a", "pcm_s16le"])
        
        cmd.append(output_path)
        
        # Run FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"Audio format conversion successful: {target_format}")
            return True
        else:
            logger.error(f"Audio format conversion failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error in audio format conversion: {str(e)}")
        return False