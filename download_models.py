#!/usr/bin/env python3
"""
Download script for Whisper models to ensure offline functionality
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def download_whisper_models():
    """Download commonly used Whisper models for offline use"""
    try:
        # Temporarily allow downloads
        os.environ['HF_HUB_OFFLINE'] = '0'
        os.environ['TRANSFORMERS_OFFLINE'] = '0'
        
        from faster_whisper import WhisperModel as FasterWhisperModel
        
        # Common model sizes
        models = ['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3']
        
        logger.info("Downloading Whisper models for offline use...")
        
        for model_name in models:
            try:
                logger.info(f"Downloading {model_name}...")
                model = FasterWhisperModel(model_name, device="cpu", compute_type="float32")
                logger.info(f"✓ {model_name} downloaded successfully")
                del model
            except Exception as e:
                logger.error(f"✗ Failed to download {model_name}: {str(e)}")
        
        logger.info("Download completed. Models are now available for offline use.")
        
    except ImportError:
        logger.error("faster-whisper not installed. Please install with: pip install faster-whisper")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during download: {str(e)}")
        sys.exit(1)
    finally:
        # Set back to offline mode
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'

if __name__ == "__main__":
    download_whisper_models()