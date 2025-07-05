"""
Voice Similarity Analysis Module

This module provides voice consistency validation for generated TTS chunks
to ensure they match the reference audio voice characteristics.

Features:
- Audio feature extraction (MFCC, spectral features, pitch analysis)
- Voice similarity scoring using multiple audio metrics
- Configurable similarity thresholds
- Integration with chunk generation for voice consistency validation
"""

import os
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any
import warnings

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

logger = logging.getLogger(__name__)

try:
    import librosa
    import scipy.stats
    from scipy.spatial.distance import cosine
    LIBROSA_AVAILABLE = True
    logger.info("✅ librosa available for voice similarity analysis")
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("⚠️ librosa not available - voice similarity validation disabled")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    logger.warning("⚠️ soundfile not available - using librosa for audio loading")


class VoiceSimilarityAnalyzer:
    """
    Analyzes voice similarity between reference audio and generated chunks
    using multiple audio features and statistical comparisons.
    Supports multiple character voices and enhanced voice consistency validation.
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.75,
                 feature_weights: Optional[Dict[str, float]] = None,
                 sample_rate: int = 24000,
                 character_voices: Optional[Dict[str, str]] = None):
        """
        Initialize the voice similarity analyzer.
        
        Args:
            similarity_threshold: Minimum similarity score (0-1) to consider a match
            feature_weights: Custom weights for different features
            sample_rate: Audio sample rate for analysis
            character_voices: Dict mapping character names to voice reference paths
        """
        self.similarity_threshold = similarity_threshold
        self.sample_rate = sample_rate
        self.reference_features = None
        self.character_voices = character_voices or {}
        self.character_features = {}  # Cache for character-specific features
        
        # Enhanced feature weights - optimized for ultra-strict voice consistency
        self.feature_weights = feature_weights or {
            'mfcc': 0.55,       # Increased - primary for accent/voice detection
            'spectral': 0.12,   # Reduced slightly to prioritize voice-specific features
            'pitch': 0.13,      # Increased - fundamental frequency is key for voice identity
            'energy': 0.05,     # Energy/loudness characteristics
            'formant': 0.18,    # Increased - critical for accent and voice character detection
            'tempo': 0.07       # Increased - speaking rate consistency important for character voices
        }
        
        logger.info(f"🎯 Voice similarity analyzer initialized (threshold: {similarity_threshold})")
        logger.info(f"🔧 Feature weights: MFCC={self.feature_weights['mfcc']:.2f}, Spectral={self.feature_weights['spectral']:.2f}, Pitch={self.feature_weights['pitch']:.2f}, Formant={self.feature_weights['formant']:.2f}, Tempo={self.feature_weights['tempo']:.2f}")
    
    def extract_voice_features(self, audio_path: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract comprehensive voice features from audio file.
        
        Returns dict with feature arrays or None if extraction fails.
        """
        if not LIBROSA_AVAILABLE:
            logger.warning("⚠️ Cannot extract features - librosa not available")
            return None
        
        try:
            # Load audio file - always use librosa.load with target sample rate to avoid slow resampling
            logger.info(f"📁 Loading audio file: {os.path.basename(audio_path)}")
            logger.info(f"📁 Loading directly at {self.sample_rate}Hz to avoid resampling...")
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            if len(audio) == 0:
                logger.warning(f"⚠️ Empty audio file: {audio_path}")
                return None
            
            audio_duration = len(audio) / self.sample_rate
            logger.info(f"✅ Audio loaded: {len(audio)} samples, {audio_duration:.2f}s duration")
            
            # Use balanced sample duration for reliable analysis
            max_duration = 20.0  # seconds - balanced duration for accuracy and reliability
            if audio_duration > max_duration:
                max_samples = int(max_duration * self.sample_rate)
                audio = audio[:max_samples]
                logger.info(f"🔄 Truncating audio to first {max_duration}s for voice analysis ({audio_duration:.1f}s -> {max_duration}s)")
            else:
                logger.info(f"🔄 Using full audio duration ({audio_duration:.1f}s) for voice analysis")
            
            # Normalize audio
            logger.info("🔧 Normalizing audio...")
            audio = librosa.util.normalize(audio)
            
            features = {}
            
            # 1. MFCC Features (timbre characteristics) - heavily enhanced for strict accent detection
            try:
                logger.info("🔊 Extracting MFCC features... (1/6)")
                # Use more MFCC coefficients with higher resolution for better accent discrimination
                mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=26, n_fft=2048, hop_length=512)
                mfcc_mean = np.mean(mfccs, axis=1)
                mfcc_std = np.std(mfccs, axis=1)
                # Add delta features for dynamic accent characteristics
                mfcc_delta = np.mean(librosa.feature.delta(mfccs), axis=1)
                # Combine mean, std, and delta for maximum accent sensitivity
                features['mfcc'] = np.concatenate([mfcc_mean, mfcc_std, mfcc_delta])
                logger.info("✅ Enhanced MFCC features extracted with delta coefficients (1/6 complete)")
            except Exception as e:
                logger.warning(f"⚠️ MFCC extraction failed: {e}")
                features['mfcc'] = np.zeros(78)  # 26 mean + 26 std + 26 delta
            
            # 2. Spectral Features  
            try:
                logger.info("🎵 Extracting spectral features... (2/6)")
                spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
                spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)
                zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
                
                features['spectral'] = np.array([
                    np.mean(spectral_centroid),
                    np.mean(spectral_rolloff),
                    np.mean(spectral_bandwidth),
                    np.mean(zero_crossing_rate)
                ])
                logger.info("✅ Spectral features extracted (2/6 complete)")
            except Exception as e:
                logger.warning(f"⚠️ Spectral feature extraction failed: {e}")
                features['spectral'] = np.zeros(4)
            
            # 3. Pitch Features (fundamental frequency) - robust extraction
            try:
                logger.info("🎤 Extracting pitch features... (3/6)")
                # Use more conservative pitch extraction to avoid failures
                pitches, magnitudes = librosa.piptrack(
                    y=audio, 
                    sr=self.sample_rate,
                    hop_length=1024,  # Restored conservative hop length
                    fmin=80,          # Narrower voice range
                    fmax=300          # Restored original range
                )
                logger.info("   🔄 Processing pitch data...")
                # Sample balanced frames for reliability
                pitch_values = []
                step = max(1, pitches.shape[1] // 30)  # Balanced sampling
                for t in range(0, pitches.shape[1], step):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:  # Only consider valid pitches
                        pitch_values.append(pitch)
                
                if pitch_values:
                    features['pitch'] = np.array([
                        np.mean(pitch_values),
                        np.std(pitch_values),
                        np.median(pitch_values)
                    ])
                else:
                    features['pitch'] = np.zeros(3)
                logger.info("✅ Pitch features extracted (3/6 complete)")
            except Exception as e:
                logger.warning(f"⚠️ Pitch extraction failed: {e}")
                features['pitch'] = np.zeros(3)
            
            # 4. Energy Features
            try:
                logger.info("⚡ Extracting energy features... (4/6)")
                rms_energy = librosa.feature.rms(y=audio)
                features['energy'] = np.array([
                    np.mean(rms_energy),
                    np.std(rms_energy)
                ])
                logger.info("✅ Energy features extracted (4/6 complete)")
            except Exception as e:
                logger.warning(f"⚠️ Energy extraction failed: {e}")
                features['energy'] = np.zeros(2)
            
            # 5. Formant Features (vowel characteristics) - critical for accent detection
            try:
                logger.info("🗣️ Extracting formant features... (5/6)")
                # Robust formant estimation - fallback to simpler method if enhanced fails
                stft = librosa.stft(audio, hop_length=1024)  # Restored original hop_length
                magnitude = np.abs(stft)
                
                # Find spectral peaks that correspond to formants
                formant_freqs = []
                frame_step = max(1, magnitude.shape[1] // 50)  # Balanced sampling
                for frame in range(0, magnitude.shape[1], frame_step):
                    spectrum = magnitude[:, frame]
                    # Find peaks in the spectrum with relaxed parameters
                    from scipy.signal import find_peaks
                    peaks, _ = find_peaks(spectrum, height=np.max(spectrum) * 0.1)  # Restored original threshold
                    
                    # Convert to frequencies
                    peak_freqs = peaks * self.sample_rate / (2 * len(spectrum))
                    
                    # Take first 3 formants (typically F1, F2, F3)
                    formants = sorted(peak_freqs)[:3]
                    if len(formants) >= 2:  # Need at least F1 and F2
                        # Pad formants to ensure consistent length of 3
                        while len(formants) < 3:
                            formants.append(0.0)
                        formant_freqs.append(formants[:3])
                
                if formant_freqs:
                    # Calculate mean formant frequencies
                    try:
                        formant_array = np.array(formant_freqs)
                        if formant_array.ndim == 2 and formant_array.shape[1] >= 2:
                            features['formant'] = np.array([
                                np.mean(formant_array[:, 0]),  # F1
                                np.mean(formant_array[:, 1]),  # F2
                                np.mean(formant_array[:, 2]) if formant_array.shape[1] > 2 else 0  # F3
                            ])
                        else:
                            features['formant'] = np.zeros(3)
                    except (ValueError, IndexError):
                        features['formant'] = np.zeros(3)
                else:
                    features['formant'] = np.zeros(3)
                logger.info("✅ Formant features extracted (5/6 complete)")
            except Exception as e:
                logger.warning(f"⚠️ Formant extraction failed: {e}")
                features['formant'] = np.zeros(3)
            
            # 6. Tempo Features (speaking rate) - critical for speed consistency
            try:
                logger.info("⏱️ Extracting tempo features... (6/6)")
                # Calculate speaking rate using onset detection
                onsets = librosa.onset.onset_detect(y=audio, sr=self.sample_rate, units='time')
                
                if len(onsets) > 1:
                    # Calculate onset rate (onsets per second)
                    onset_rate = len(onsets) / (len(audio) / self.sample_rate)
                    
                    # Calculate inter-onset intervals
                    onset_intervals = np.diff(onsets)
                    
                    # Calculate tempo-related features
                    features['tempo'] = np.array([
                        onset_rate,                          # Onsets per second
                        np.mean(onset_intervals),           # Average inter-onset interval
                        np.std(onset_intervals),            # Variability in timing
                        np.median(onset_intervals)          # Median inter-onset interval
                    ])
                else:
                    # Fallback for very short audio or no onsets detected
                    features['tempo'] = np.zeros(4)
                logger.info("✅ Tempo features extracted (6/6 complete)")
            except Exception as e:
                logger.warning(f"⚠️ Tempo extraction failed: {e}")
                features['tempo'] = np.zeros(4)
            
            logger.info(f"🎉 All features extracted from {os.path.basename(audio_path)}")
            return features
            
        except Exception as e:
            logger.error(f"❌ Feature extraction failed for {audio_path}: {e}")
            return None
    
    def set_reference_voice(self, reference_audio_path: str, character_name: Optional[str] = None) -> bool:
        """
        Set reference voice features from audio file.
        
        Args:
            reference_audio_path: Path to reference audio file
            character_name: Optional character name for multi-character voice support
            
        Returns True if successful, False otherwise.
        """
        if not os.path.exists(reference_audio_path):
            logger.error(f"❌ Reference audio not found: {reference_audio_path}")
            return False
        
        logger.info(f"🎤 Analyzing reference voice: {os.path.basename(reference_audio_path)}")
        if character_name:
            logger.info(f"👤 Character: {character_name}")
        logger.info("⏳ Extracting voice features... (this may take 15-45 seconds for longer audio files)")
        logger.info("🔄 Starting comprehensive feature extraction process...")
        
        features = self.extract_voice_features(reference_audio_path)
        
        if features is None:
            logger.error("❌ Failed to extract reference voice features")
            return False
        
        if character_name:
            # Store character-specific features
            self.character_features[character_name] = features
            logger.info(f"✅ Character '{character_name}' voice features extracted and cached")
        else:
            # Store as default reference
            self.reference_features = features
            logger.info("✅ Reference voice features extracted and cached successfully")
        
        # Cache the reference path to avoid re-analyzing
        self._last_reference_path = reference_audio_path
        return True
    
    def calculate_similarity(self, candidate_audio_path: str, character_name: Optional[str] = None) -> Optional[float]:
        """
        Calculate similarity score between candidate audio and reference voice.
        
        Args:
            candidate_audio_path: Path to candidate audio file
            character_name: Optional character name for multi-character voice validation
            
        Returns similarity score (0-1) or None if analysis fails.
        Higher scores indicate better similarity.
        """
        # Determine which reference features to use
        reference_features = None
        if character_name and character_name in self.character_features:
            reference_features = self.character_features[character_name]
            logger.debug(f"👤 Using character-specific reference for '{character_name}'")
        elif self.reference_features is not None:
            reference_features = self.reference_features
            logger.debug("🎤 Using default reference voice")
        else:
            logger.warning("⚠️ No reference voice set - cannot calculate similarity")
            return None
        
        if not os.path.exists(candidate_audio_path):
            logger.error(f"❌ Candidate audio not found: {candidate_audio_path}")
            return None
        
        # Extract features from candidate audio
        candidate_features = self.extract_voice_features(candidate_audio_path)
        if candidate_features is None:
            logger.warning(f"⚠️ Failed to extract features from {candidate_audio_path}")
            return None
        
        try:
            # Calculate similarity for each feature type
            feature_similarities = {}
            
            for feature_name in ['mfcc', 'spectral', 'pitch', 'energy', 'formant', 'tempo']:
                ref_feature = reference_features.get(feature_name)
                cand_feature = candidate_features.get(feature_name)
                
                if ref_feature is not None and cand_feature is not None and len(ref_feature) > 0 and len(cand_feature) > 0:
                    # Use cosine similarity for feature comparison
                    # Convert to similarity score (1 - distance)
                    if len(ref_feature) == len(cand_feature):
                        try:
                            # Check for zero vectors to avoid division by zero
                            ref_norm = np.linalg.norm(ref_feature)
                            cand_norm = np.linalg.norm(cand_feature)
                            
                            if ref_norm == 0 or cand_norm == 0:
                                similarity = 0.0
                            else:
                                distance = cosine(ref_feature, cand_feature)
                                similarity = 1 - distance if not np.isnan(distance) else 0.0
                            
                            feature_similarities[feature_name] = max(0.0, min(1.0, similarity))
                        except Exception as e:
                            logger.warning(f"⚠️ Cosine similarity failed for {feature_name}: {e}")
                            feature_similarities[feature_name] = 0.0
                    else:
                        # If different lengths, use correlation
                        try:
                            min_len = min(len(ref_feature), len(cand_feature))
                            corr, _ = scipy.stats.pearsonr(ref_feature[:min_len], cand_feature[:min_len])
                            feature_similarities[feature_name] = max(0.0, min(1.0, abs(corr) if not np.isnan(corr) else 0.0))
                        except Exception as e:
                            logger.warning(f"⚠️ Correlation failed for {feature_name}: {e}")
                            feature_similarities[feature_name] = 0.0
                else:
                    feature_similarities[feature_name] = 0.0
            
            # Calculate weighted overall similarity with stricter accent detection
            overall_similarity = 0.0
            total_weight = 0.0
            
            for feature_name, similarity in feature_similarities.items():
                weight = self.feature_weights.get(feature_name, 0.0)
                overall_similarity += similarity * weight
                total_weight += weight
            
            if total_weight > 0:
                overall_similarity /= total_weight
            
            # Apply penalty for accent-sensitive and tempo features being too different
            # MFCC and formant differences are critical indicators of accent changes
            # Tempo differences indicate speed variation issues
            mfcc_sim = feature_similarities.get('mfcc', 1.0)
            formant_sim = feature_similarities.get('formant', 1.0)
            tempo_sim = feature_similarities.get('tempo', 1.0)
            
            # Apply balanced voice consistency detection - effective but not breaking
            if mfcc_sim < 0.92 or formant_sim < 0.90:
                accent_penalty = 0.25  # Balanced penalty for accent differences
                overall_similarity *= (1.0 - accent_penalty)
                logger.debug(f"🚨 Voice difference detected - applying penalty (MFCC: {mfcc_sim:.3f}, Formant: {formant_sim:.3f})")
            
            # Additional penalty if BOTH MFCC and formant show accent drift
            if mfcc_sim < 0.88 and formant_sim < 0.86:
                severe_accent_penalty = 0.20  # Balanced penalty for severe accent drift
                overall_similarity *= (1.0 - severe_accent_penalty)
                logger.info(f"🚨 SEVERE voice drift detected - accent influence detected (MFCC: {mfcc_sim:.3f}, Formant: {formant_sim:.3f})")
            
            # Enhanced pitch consistency check - critical for voice identity
            pitch_sim = feature_similarities.get('pitch', 1.0)
            if pitch_sim < 0.80:
                pitch_penalty = 0.15  # Balanced penalty for pitch inconsistency
                overall_similarity *= (1.0 - pitch_penalty)
                logger.debug(f"🚨 Pitch difference detected - applying penalty (Pitch: {pitch_sim:.3f})")
            
            # If tempo similarity is low, apply speed penalty
            if tempo_sim < 0.78:
                speed_penalty = 0.12  # Balanced penalty for speed differences
                overall_similarity *= (1.0 - speed_penalty)
                logger.debug(f"🚨 Speed difference detected - applying penalty (Tempo: {tempo_sim:.3f})")
            
            # Provide detailed feedback about voice consistency issues with balanced thresholds
            issues = []
            if mfcc_sim < 0.92:
                if mfcc_sim < 0.85:
                    issues.append(f"major voice timbre mismatch (MFCC: {mfcc_sim:.3f})")
                else:
                    issues.append(f"voice timbre drift (MFCC: {mfcc_sim:.3f})")
            if formant_sim < 0.90:
                if formant_sim < 0.82:
                    issues.append(f"major accent difference - voice change detected (Formant: {formant_sim:.3f})")
                else:
                    issues.append(f"accent drift detected (Formant: {formant_sim:.3f})")
            if pitch_sim < 0.80:
                issues.append(f"pitch inconsistency (Pitch: {pitch_sim:.3f})")
            if tempo_sim < 0.78:
                issues.append(f"speed variation (Tempo: {tempo_sim:.3f})")
            
            if issues:
                logger.info(f"🎯 Voice analysis: {', '.join(issues)} -> Overall: {overall_similarity:.3f}")
            else:
                logger.info(f"🎯 Voice analysis: Good consistency -> Overall: {overall_similarity:.3f}")
            
            return overall_similarity
            
        except Exception as e:
            logger.error(f"❌ Similarity calculation failed: {e}")
            return None
    
    def validate_voice_consistency(self, candidate_audio_path: str) -> Tuple[bool, float]:
        """
        Validate if candidate audio matches reference voice within threshold.
        
        Returns (is_consistent, similarity_score).
        """
        similarity = self.calculate_similarity(candidate_audio_path)
        
        if similarity is None:
            logger.warning("⚠️ Could not calculate similarity - accepting candidate")
            return True, 0.0  # Accept if we can't analyze
        
        is_consistent = similarity >= self.similarity_threshold
        
        if is_consistent:
            logger.debug(f"✅ Voice consistent: {similarity:.3f} >= {self.similarity_threshold}")
        else:
            logger.info(f"❌ Voice inconsistent: {similarity:.3f} < {self.similarity_threshold}")
        
        return is_consistent, similarity


# Global analyzer instance (created when needed)
_voice_analyzer: Optional[VoiceSimilarityAnalyzer] = None

def get_voice_analyzer(similarity_threshold: float = 0.75) -> Optional[VoiceSimilarityAnalyzer]:
    """
    Get global voice similarity analyzer instance.
    
    Returns None if librosa is not available.
    """
    global _voice_analyzer
    
    if not LIBROSA_AVAILABLE:
        return None
    
    if _voice_analyzer is None or _voice_analyzer.similarity_threshold != similarity_threshold:
        _voice_analyzer = VoiceSimilarityAnalyzer(similarity_threshold=similarity_threshold)
    
    return _voice_analyzer

def validate_chunk_voice_consistency(
    candidate_audio_path: str,
    reference_audio_path: str,
    similarity_threshold: float = 0.75
) -> Tuple[bool, float]:
    """
    Convenience function to validate voice consistency for a single chunk.
    
    Args:
        candidate_audio_path: Path to generated audio chunk
        reference_audio_path: Path to reference voice audio
        similarity_threshold: Minimum similarity score to accept
    
    Returns:
        Tuple of (is_consistent, similarity_score)
    """
    analyzer = get_voice_analyzer(similarity_threshold)
    
    if analyzer is None:
        logger.info("🔇 Voice similarity analysis not available - accepting chunk")
        return True, 0.0
    
    # Set reference voice if not already set or if it's different
    if (analyzer.reference_features is None or 
        not hasattr(analyzer, '_last_reference_path') or 
        analyzer._last_reference_path != reference_audio_path):
        
        logger.info("🔍 Analyzing reference voice (first time or voice changed)")
        if not analyzer.set_reference_voice(reference_audio_path):
            logger.warning("⚠️ Could not set reference voice - accepting chunk")
            return True, 0.0
    else:
        logger.info("✅ Using cached reference voice analysis")
    
    return analyzer.validate_voice_consistency(candidate_audio_path)

def is_voice_similarity_available() -> bool:
    """Check if voice similarity analysis is available."""
    return LIBROSA_AVAILABLE