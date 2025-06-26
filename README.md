# Chatterbox Web UI

A simple, powerful web interface for [Resemble AI's Chatterbox TTS](https://github.com/resemble-ai/chatterbox) with zero-shot voice cloning.

![Screenshot](screenshot.png)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Create environment
conda create -n chatterbox python=3.12
conda activate chatterbox

# Install PyTorch (choose your platform)
# CUDA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Chatterbox
pip install chatterbox-tts

# Install Whisper / Faster Whisper
pip install openai-whisper
pip install faster-whisper

# Install Python FFMPEG wrapper
pip install ffmpeg-python

# Clone and setup
git clone https://github.com/your-username/chatterboxwebui.git
cd chatterboxwebui
pip install -r requirements.txt
```

### 2. Run the App
```bash
python server.py
```
Open your browser to `http://127.0.0.1:5000`

## 🎤 Voice Selection

### Preset Voices
1. Add audio files (`.wav`, `.mp3`, `.flac`, `.ogg`) to the `static/voices/` directory
2. Restart the server
3. Select from the dropdown in the web interface

### Custom Voice Upload
1. Click "Upload Custom Reference Audio"
2. Choose an audio file (10+ seconds recommended)
3. Generated speech will clone this voice

**Note**: You can only use one voice source at a time - either preset or uploaded, not both.

## 📝 Basic Usage

### Simple Text-to-Speech
1. Enter your text (up to 10,000 characters by default)
2. Optionally select a voice or upload reference audio
3. Adjust settings if needed
4. Click "Generate"

### Custom Pauses
Add precise pauses with break tags:
```
Hello there. <break time="2s" /> This was a 2-second pause.
Short pause here. <break time="500ms" /> Continuing...
```

## 📁 File Structure

```
chatterboxwebui/
├── server.py             # Main Flask server
├── connector.py          # TTS processing engine
├── config.py             # Configuration settings
├── static/
│   ├── voices/           # Put preset voice audio files here
│   ├── output/           # Generated audio files
│   ├── uploads/          # Temporary uploaded files
│   └── js/, css/         # Web interface files
└── templates/
    └── index.html        # Main web page
```

## 🔧 Configuration

Set environment variables before running:
```bash
export MAX_TEXT_LENGTH=25000    # Increase text limit
export HOST=0.0.0.0            # Listen on all interfaces
export PORT=8080               # Change port
python server.py
```

## 🛠️ Troubleshooting

### Common Issues

**"NLTK data not found"**
```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

**"No voices in dropdown"**
- Add audio files to `static/voices/` directory
- Restart the server
- Check supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`

**"Generation too slow"**
- Install CUDA version of PyTorch
- Reduce chunk size in settings
- Use shorter text inputs

**"File upload fails"**
- Check file size (max 50MB)
- Ensure file format is supported
- Try converting to WAV format

## 📋 Requirements

- Python 3.10+
- FFMPEG for post audio functions (installed and in system PATH)

## 🏗️ Built With

- **Backend**: Flask, SocketIO, PyTorch
- **Frontend**: Vanilla JavaScript, CSS Grid/Flexbox
- **TTS Engine**: [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) by Resemble AI
- **Audio Processing**: FFmpeg, librosa

## 📜 License

MIT License - Use responsibly and ethically.

## 🙏 Credits

- [Resemble AI](https://resemble.ai) for the Chatterbox TTS model
- [Chatterbox-TTS-Extended](https://github.com/petermg/Chatterbox-TTS-Extended) for advanced features

---

**⚠️ Responsible AI**: This tool can clone voices. Please use ethically and respect others' rights.