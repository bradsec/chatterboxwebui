# Chatterbox Web UI

A Python Flask-based web interface for [Resemble AI's Chatterbox TTS](https://github.com/resemble-ai/chatterbox) with zero-shot voice cloning and advanced speech controls.

## Key Features

- **Break Tag Support**: Add custom pauses using `<break time="1.5s" />` syntax
- **Advanced Parameters**: Temperature, CFG weight, and fine-tuning options
- **Smart Text Chunking**: Automatically handles long text within Chatterbox limits
- **Audio Post-processing**: Speed, pitch, noise reduction, and silence removal
- **Real-time Progress**: Live updates during generation
- **Seed Tracking**: Reproducible results with automatic seed capture
- **Dark/Light Theme**: Persistent theme switching
- **Enhanced Security**: Input validation and secure file handling

Built on [Resemble AI's Chatterbox](https://github.com/resemble-ai/chatterbox):
- **Zero-shot voice cloning** from short samples
- **Emotion exaggeration control** (first open-source TTS with this feature)
- **0.5B Llama backbone** with ultra-stable generation
- **Trained on 0.5M hours** of speech data
- **Built-in Perth watermarking** for responsible AI use

![Screenshot](screenshot.png)

## Sample Audio

### Male
https://github.com/user-attachments/assets/9a4c0ea3-98b5-48f7-a377-2af7bed90b5e

### Female
https://github.com/user-attachments/assets/a264abc0-e6f2-4dc5-b7a0-638c36f0fcee

## Quick Start

### Installation

#### 1. Environment Setup
```bash
# Using Miniconda (recommended)
conda create -n chatterboxwebui python=3.12
conda activate chatterboxwebui
```

#### 2. Install PyTorch
```bash
# CUDA 12.6 (check your CUDA version)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# CPU only
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# macOS Apple Silicon
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
```

#### 3. Install Dependencies
```bash
# Install Chatterbox TTS
pip install chatterbox-tts

# Clone and setup
git clone https://github.com/bradsec/chatterboxwebui.git
cd chatterboxwebui
pip install -r requirements.txt
```

#### 4. Fix NLTK (Important!)
```bash
python -c "
import nltk
import os
nltk_data_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
try:
    nltk.download('punkt_tab', download_dir=nltk_data_dir)
    print('NLTK data downloaded successfully')
except:
    try:
        nltk.download('punkt', download_dir=nltk_data_dir)
        print('NLTK fallback downloaded successfully')
    except Exception as e:
        print(f'Download failed: {e}')
"
```

### Run the Application
```bash
python server.py
```
Open your browser to `http://127.0.0.1:5000`

## Usage Guide

### Basic Text-to-Speech
1. Enter text
2. Adjust parameters as needed
3. Click "Generate"
4. Download or manage files from the audio list

### Voice Cloning
1. Upload reference audio (WAV, MP3, FLAC, OPUS - max 50MB)
2. Use 10 or more seconds of clear, single-speaker audio
3. Generated speech will mimic the uploaded voice

### Break Tags for Custom Pauses
Add precise pauses anywhere in your text:

```
Hello there. <break time="2s" /> This pause was 2 seconds.
First sentence. <break time="500ms" /> Short pause here.
End of paragraph. <break time="1.5s" /> New paragraph starts.
```

**Break Tag Syntax:**
- `<break time="1.5s" />` - 1.5 second pause
- `<break time="800ms" />` - 800 millisecond pause
- Place between sentences/paragraphs for best results
- Break tags are processed separately from text to avoid TTS confusion

### Key Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Exaggeration** | 0.25-2.0 | 0.5 | Emotional intensity (higher = more expressive) |
| **Temperature** | 0.05-5.0 | 0.8 | Randomness/creativity (lower = more consistent) |
| **CFG Weight** | 0.0-1.0 | 0.5 | Pacing control (lower = faster speech) |
| **Chunk Size** | 50-300 | 130 | Text processing size (larger = smoother flow) |

### Text Length Limits
Set via environment variable:
```bash
export MAX_TEXT_LENGTH=25000  # Default: 10000
python server.py
```

**Recommended limits:**
- **5,000 chars**: Quick clips (30 sec - 2 min)
- **10,000 chars**: Default articles (1-5 min)
- **25,000 chars**: Long articles (5-15 min, 8GB+ RAM)
- **50,000+ chars**: Chapters (15+ min, GPU recommended)

### Environment Variables
```bash
HOST=0.0.0.0              # Server host
PORT=5000                 # Server port  
DEBUG=False              # Debug mode
MAX_TEXT_LENGTH=10000    # Text input limit
SECRET_KEY=your_key      # Flask secret (auto-generated)
```

## File Structure
```
chatterboxwebui/
├── server.py              # Flask server & SocketIO
├── connector.py           # Chatterbox integration
├── requirements.txt       # Dependencies
├── templates/index.html   # Web interface
├── static/
│   ├── css/styles.css    # Styling
│   ├── js/main.js        # Frontend logic
│   ├── js/theme.js       # Theme switching
│   ├── output/           # Generated audio (auto-created)
│   ├── uploads/          # Reference audio (auto-created)
│   └── json/data.json    # Generation history (auto-created)
```

## API Usage

```python
from connector import generate_voice

filename, seed_used = generate_voice(
    text_input="Hello, this is a test.",
    audio_prompt_path=None,  # Optional reference audio
    exaggeration=0.5,
    temperature=0.8,
    cfg_weight=0.5,
    chunk_size=130,
    speed=1.0,
    pitch=0,
    reduce_noise=False,
    remove_silence=False,
    seed=0
)
```

## License

MIT License - Same as Chatterbox TTS. See the [original repository](https://github.com/resemble-ai/chatterbox) for details.

## Acknowledgements

- [Resemble AI](https://resemble.ai) for Chatterbox TTS
- Open-source community for supporting libraries

---

**Disclaimer**: Use responsibly. Don't use this technology for harmful purposes.