# Chatterbox Web UI

This application is a Python Flask-based web UI designed to facilitate text-to-speech generation using [Resemble AI's Chatterbox TTS](https://github.com/resemble-ai/chatterbox).

## Key Features

- **Zero-shot Voice Cloning**: Upload reference audio to clone any voice
- **Emotion Exaggeration Control**: Unique feature for controlling speech expressiveness
- **Advanced Parameters**: Temperature, CFG weight, and other fine-tuning options
- **Text Chunking**: Automatically splits long text to work within Chatterbox's limits
- **Post-processing**: Speed, pitch, noise reduction, and silence removal
- **Real-time Progress**: Live progress updates during generation
- **Audio Management**: Download, delete, and organize generated audio files

![Screenshot](screenshot.png)

## Sample Audio

https://github.com/user-attachments/assets/ad5d06b3-071c-432f-b73c-d338cca01279

## Installation (example using MiniConda)

```bash
# Environment setup using MiniConda
conda create -n chatterboxwebui python=3.12
conda activate chatterboxwebui

# Check NVidia CUDA version, you may need to change the install command line below
# More information: https://pytorch.org/get-started/previous-versions/
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# Install chatterbox
pip install chatterbox-tts

# chatterboxwebui setup and requirements
git clone https://github.com/bradsec/chatterboxwebui
cd chatterboxwebui

# NLTK (Natural Language Toolkit) is a popular open-source Python library for natural language processing (NLP). It provides a wide range of tools and resources for working with text data, including tasks like tokenization, stemming, parsing, and sentiment analysis.
python -c "import nltk; nltk.download('punkt')"

# Install webui requirements
pip install -r requirements
```

### Running the application
```bash
python server.py
```

### Access the web interface  
Open your browser to `http://127.0.0.1:5000`

## Usage

### Basic Text-to-Speech
1. Enter your text in the text area
2. Adjust parameters as needed (defaults work well for most use cases)
3. Click "Generate" to create audio

### Voice Cloning
1. Upload a reference audio file (WAV, MP3, FLAC, or OPUS)
2. Use clear speech samples, ideally 3-10 seconds long
3. The generated speech will mimic the uploaded voice characteristics

### Parameter Guide

**Exaggeration (0.25-2.0)**
- Controls emotional intensity and expressiveness
- 0.5 (default): Neutral, conversational speech
- Higher values: More dramatic, expressive speech
- Lower values: More subdued, calm speech

**Temperature (0.1-2.0)**
- Controls randomness and creativity
- 0.8 (default): Good balance of quality and variation

**CFG Weight (0.0-1.0)**
- Controls pacing and prompt adherence
- 0.5 (default): Balanced pacing

### Tips for Best Results

**General Use:**
- Default settings work well for most content
- For fast-speaking reference voices, lower CFG weight to ~0.3

**Expressive/Dramatic Speech:**
- Use lower CFG weight (~0.3) and higher exaggeration (~0.7+)
- Higher exaggeration speeds up speech; lower CFG weight compensates

**Voice Cloning:**
- Use clear, high-quality reference audio
- Avoid background noise or music
- 3-10 second samples work best
- Single speaker recordings preferred

## File Structure

- `server.py` - Flask web server and SocketIO handling
- `connector.py` - Chatterbox TTS integration and audio processing
- `templates/index.html` - Main web interface
- `static/js/main.js` - Frontend functionality and SocketIO client
- `static/js/theme.js` - Dark/light theme switching
- `static/css/styles.css` - Styling and responsive design
- `static/output/` - Generated audio files
- `static/json/data.json` - Generation history and metadata

## Technical Details

### Text Processing
- Automatically chunks long text into segments under 300 characters
- Preserves sentence structure and handles special formatting
- Supports music note markers (♪) for song content

### Audio Processing
- Uses Chatterbox's native sample rate and format
- Applies post-processing effects (speed, pitch, noise reduction)
- Concatenates chunks with natural silence between segments
- Maintains high audio quality throughout pipeline

### Device Support
- Automatic device detection (CUDA, MPS for Apple Silicon, CPU)
- Optimized model loading for different hardware configurations
- Memory-efficient processing for long text generation

## Requirements

- Python 3.12+
- PyTorch with appropriate device support (CUDA/MPS/CPU)
- Chatterbox TTS
- Flask and Flask-SocketIO
- Additional dependencies in requirements.txt

## Troubleshooting

**Model Loading Issues:**
- Ensure sufficient disk space for model downloads
- Check internet connection for initial model download
- Verify PyTorch installation matches your hardware

**Audio Generation Errors:**
- Check text length (max 10,000 characters)
- Verify reference audio format if using voice cloning
- Ensure output directory permissions

## License

This project maintains the same MIT license as Chatterbox TTS. See the original [Chatterbox repository](https://github.com/resemble-ai/chatterbox) for details.

## Acknowledgements

- [Resemble AI](https://resemble.ai) for creating Chatterbox TTS
