from flask import Flask, render_template, request, send_from_directory, jsonify
from flask_socketio import SocketIO, emit
import os
import json
import time
import uuid
import logging
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from connector import generate_voice

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24))
socketio = SocketIO(app, max_http_buffer_size=50 * 1024 * 1024, cors_allowed_origins="*")

# Add template filter for formatting numbers
@app.template_filter('format_number')
def format_number(value):
    """Format numbers with thousands separators"""
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return str(value)

# Allowed audio file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'opus', 'm4a', 'ogg'}
UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = os.path.join('static', 'output')
JSON_FOLDER = os.path.join('static', 'json')

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_audio_file(file):
    """Validate uploaded audio file"""
    if not file or file.filename == '':
        return False, 'No file selected'
    
    if not allowed_file(file.filename):
        return False, f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
    
    # Check file size (additional check beyond Flask's MAX_CONTENT_LENGTH)
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Reset to beginning
    
    if size > 50 * 1024 * 1024:  # 50MB
        return False, 'File too large (max 50MB)'
    
    return True, None

def cleanup_reference_audio(filepath):
    """Clean up reference audio file after generation"""
    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Cleaned up reference audio: {filepath}")
    except Exception as e:
        logger.error(f"Error cleaning up reference audio {filepath}: {str(e)}")

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [OUTPUT_FOLDER, JSON_FOLDER, UPLOAD_FOLDER]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({'error': 'File too large (max 50MB)'}), 413

@app.errorhandler(500)
def handle_internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/', methods=['GET'])
def index():
    max_text_length = int(os.environ.get('MAX_TEXT_LENGTH', 10000))
    return render_template('index.html', max_text_length=max_text_length)

@app.route('/static/output/<path:filename>')
def serve_static(filename):
    """Serve generated audio files"""
    # Sanitize filename to prevent directory traversal
    filename = secure_filename(filename)
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """Handle audio file upload for voice reference"""
    try:
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['audio_file']
        
        # Validate file
        is_valid, error_msg = validate_audio_file(file)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        # Generate unique filename to avoid conflicts
        filename = secure_filename(file.filename)
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        # Save file
        file.save(filepath)
        logger.info(f"Uploaded reference audio: {unique_filename}")
        
        return jsonify({
            'success': True, 
            'filename': unique_filename,
            'filepath': filepath
        })
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return jsonify({'error': f'Failed to save file: {str(e)}'}), 500

@app.route('/clear_reference_audio', methods=['POST'])
def clear_reference_audio():
    """Clear/delete a specific reference audio file"""
    try:
        data = request.get_json()
        if not data or 'filename' not in data:
            return jsonify({'error': 'No filename provided'}), 400
        
        filename = secure_filename(data.get('filename'))
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Cleared reference audio: {filename}")
            return jsonify({'success': True, 'message': f'Reference audio {filename} cleared successfully'})
        else:
            return jsonify({'success': True, 'message': 'File was already removed'})
            
    except Exception as e:
        logger.error(f"Error clearing reference audio: {str(e)}")
        return jsonify({'error': f'Failed to clear reference audio: {str(e)}'}), 500

@socketio.on('start_generation')
def handle_start_generation(data):
    """Handle TTS generation request"""
    try:
        # Validate input data
        text_input = data.get('text_input', '').strip()
        if not text_input:
            emit('error', {'error': 'Text is empty.'})
            return

        # Check text length - configurable limit
        max_text_length = int(os.environ.get('MAX_TEXT_LENGTH', 10000))
        if len(text_input) > max_text_length:
            emit('error', {'error': f'Text is too long. Please limit to {max_text_length:,} characters.'})
            return

        # Extract and validate parameters
        audio_prompt_filename = data.get('audio_prompt_path')
        
        try:
            exaggeration = float(data.get('exaggeration', 0.5))
            temperature = float(data.get('temperature', 0.8))
            cfg_weight = float(data.get('cfg_weight', 0.5))
            chunk_size = int(data.get('chunk_size', 300))
            speed = float(data.get('speed', 1.0))
            pitch = int(data.get('pitch', 0))
            seed = int(data.get('seed', 0))
        except (ValueError, TypeError) as e:
            emit('error', {'error': f'Invalid parameter values: {str(e)}'})
            return
        
        # Validate parameter ranges
        if not (0.25 <= exaggeration <= 2.0):
            emit('error', {'error': 'Exaggeration must be between 0.25 and 2.0'})
            return
        if not (0.05 <= temperature <= 5.0):
            emit('error', {'error': 'Temperature must be between 0.05 and 5.0'})
            return
        if not (0.0 <= cfg_weight <= 1.0):
            emit('error', {'error': 'CFG Weight must be between 0.0 and 1.0'})
            return
        if not (50 <= chunk_size <= 300):
            emit('error', {'error': 'Chunk size must be between 50 and 300'})
            return
        
        reduce_noise = bool(data.get('reduce_noise', False))
        remove_silence = bool(data.get('remove_silence', False))

        # Validate audio prompt file if provided
        audio_prompt_path = None
        if audio_prompt_filename:
            audio_prompt_path = os.path.join(UPLOAD_FOLDER, secure_filename(audio_prompt_filename))
            if not os.path.exists(audio_prompt_path):
                emit('error', {'error': 'Reference audio file not found. Please re-upload.'})
                return

        # Generate audio
        start_time = time.time()
        result = generate_voice(
            text_input=text_input,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
            chunk_size=chunk_size,
            speed=speed,
            pitch=pitch,
            reduce_noise=reduce_noise,
            remove_silence=remove_silence,
            seed=seed,
            progress_callback=update_progress
        )
        end_time = time.time()
        duration = round(end_time - start_time, 2)
        
        # Handle the returned tuple (filename, actual_seed)
        if result and len(result) == 2:
            filename, actual_seed = result
        else:
            filename, actual_seed = None, seed
        
        logger.info(f"Generation completed in {duration} seconds")

        if filename:
            # Save generation data to JSON with actual seed used
            write_to_json(text_input, filename, audio_prompt_filename, exaggeration, temperature, 
                         cfg_weight, chunk_size, speed, pitch, reduce_noise, remove_silence, actual_seed, duration)
            
            emit('generation_complete', {'filename': filename, 'generation_time': duration})
        else:
            emit('error', {'error': 'Audio generation failed. Please try again.'})
            
    except Exception as e:
        logger.error(f"Error in generation: {str(e)}")
        emit('error', {'error': f'Generation failed: {str(e)}'})

def update_progress(current, total):
    """Update generation progress"""
    try:
        progress = current / total if total > 0 else 0
        emit('generation_progress', {'progress': progress}, broadcast=False)
    except Exception as e:
        logger.error(f"Error updating progress: {str(e)}")

@app.route('/static/output/<path:filename>', methods=['DELETE'])
def delete_file(filename):
    """Delete a generated audio file"""
    try:
        filename = secure_filename(filename)
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        os.remove(file_path)
        remove_from_json(filename)
        logger.info(f"Deleted file: {filename}")
        
        return jsonify({'message': 'File deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting file {filename}: {str(e)}")
        return jsonify({'error': f'Error deleting file: {str(e)}'}), 500

def write_to_json(text_input, filename, audio_prompt_path, exaggeration, temperature, 
                 cfg_weight, chunk_size, speed, pitch, reduce_noise, remove_silence, seed, duration):
    """Write generation data to JSON file"""
    try:
        json_file = os.path.join(JSON_FOLDER, 'data.json')
        
        # Read existing data
        data = {}
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f"Error reading existing JSON data: {str(e)}")
                data = {}

        # Remove the .wav extension from the filename for the key
        file_id = os.path.splitext(filename)[0]

        # Add new data to the beginning
        new_entry = {
            'textInput': text_input,
            'audioPromptPath': audio_prompt_path,
            'exaggeration': exaggeration,
            'temperature': temperature,
            'cfgWeight': cfg_weight,
            'chunkSize': chunk_size,
            'speed': speed,
            'pitch': pitch,
            'reduceNoise': reduce_noise,
            'removeSilence': remove_silence,
            'seed': seed,
            'actualSeed': seed,  # Store the actual seed used (same as seed for now, but will differ when seed=0)
            'outputFile': filename,
            'generationTime': duration,
            'timestamp': time.time()
        }
        
        data = {file_id: new_entry, **data}

        # Write data back to file
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
    except Exception as e:
        logger.error(f"Error writing to JSON: {str(e)}")

def remove_from_json(filename):
    """Remove entry from JSON file"""
    try:
        json_file = os.path.join(JSON_FOLDER, 'data.json')
        
        if not os.path.exists(json_file):
            return
        
        # Read existing data
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        # Remove the entry
        file_id = os.path.splitext(filename)[0]
        if file_id in data:
            del data[file_id]

        # Write back to file or remove file if empty
        if data:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        else:
            os.remove(json_file)
            
    except Exception as e:
        logger.error(f"Error removing from JSON: {str(e)}")

def cleanup_old_uploads():
    """Clean up old upload files on startup"""
    try:
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
            logger.info("Cleaned up old upload files")
    except Exception as e:
        logger.error(f"Error cleaning up old uploads: {e}")

if __name__ == '__main__':
    # Ensure output directories exist
    ensure_directories()
    
    # Clean up any leftover upload files from previous sessions
    cleanup_old_uploads()
    
    # Get host and port from environment variables for deployment flexibility
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    logger.info(f"Starting Chatterbox Web UI on {host}:{port}")
    socketio.run(app, host=host, port=port, debug=debug)