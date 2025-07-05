"""
Audio Effects Module

This module contains audio processing functions for detecting and fixing common
TTS artifacts like clicks, pops, glitches, and other audio quality issues.

Functions:
- detect_and_fix_audio_artifacts: Main function to detect and fix audio artifacts
- remove_clicks_pops: Remove sudden amplitude spikes (clicks/pops)
- fix_chunk_discontinuities: Fix amplitude discontinuities at chunk boundaries
- remove_dc_offset: Remove DC bias from audio
- fix_quiet_section_glitches: Fix glitches in quiet audio sections
- smooth_amplitude_jumps: Smooth sudden amplitude changes
- analyze_audio_quality: Analyze audio quality metrics
"""

import numpy as np
import logging
import warnings
from typing import List, Optional
from scipy import signal
from scipy.stats import zscore

logger = logging.getLogger(__name__)


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
        
        # 4. Detect and fix glitches in quiet sections (but skip for very quiet audio)
        rms = np.sqrt(np.mean(cleaned_audio**2))
        if rms > 0.001:  # Only check for glitches if audio isn't extremely quiet
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