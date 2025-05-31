from flask import Flask, render_template, request, send_from_directory, jsonify
from flask_socketio import SocketIO, emit
import os
import json
import time
import uuid
from werkzeug.utils import secure_filename
from connector import generate_voice

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
socketio = SocketIO(app, max_http_buffer_size=50 * 1024 * 1024)
generation_task = None

# Allowed audio file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'opus', 'm4a', 'ogg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_reference_audio(filepath):
    """Clean up reference audio file after generation"""
    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            print(f"Cleaned up reference audio: {filepath}")
    except Exception as e:
        print(f"Error cleaning up reference audio {filepath}: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/static/output/<path:filename>')
def serve_static(filename):
    return send_from_directory('static/output', filename)

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """Handle audio file upload for voice reference"""
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Create uploads directory if it doesn't exist
        upload_dir = os.path.join('static', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename to avoid conflicts
        filename = secure_filename(file.filename)
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
        filepath = os.path.join(upload_dir, unique_filename)
        
        try:
            file.save(filepath)
            return jsonify({
                'success': True, 
                'filename': unique_filename,
                'filepath': filepath
            })
        except Exception as e:
            return jsonify({'error': f'Failed to save file: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type. Allowed: wav, mp3, flac, opus, m4a, ogg'}), 400

@app.route('/clear_reference_audio', methods=['POST'])
def clear_reference_audio():
    """Clear/delete a specific reference audio file"""
    data = request.get_json()
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    filepath = os.path.join('static', 'uploads', filename)
    
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'success': True, 'message': f'Reference audio {filename} cleared successfully'})
        else:
            return jsonify({'success': True, 'message': 'File was already removed'})
    except Exception as e:
        return jsonify({'error': f'Failed to clear reference audio: {str(e)}'}), 500

@socketio.on('start_generation')
def handle_start_generation(data):
    text_input = data['text_input']
    audio_prompt_filename = data.get('audio_prompt_path')  # This will be the uploaded filename
    exaggeration = float(data.get('exaggeration', 0.5))
    temperature = float(data.get('temperature', 0.8))
    cfg_weight = float(data.get('cfg_weight', 0.5))
    chunk_size = int(data.get('chunk_size', 300))
    reduce_noise = data.get('reduce_noise', False)
    remove_silence = data.get('remove_silence', False)
    speed = float(data.get('speed', 1.0))
    pitch = int(data.get('pitch', 0))
    seed = int(data.get('seed', 0))
    
    if text_input.strip() == '':
        emit('error', {'error': 'Text is empty.'})
        return

    # Check text length (Chatterbox has a practical limit)
    if len(text_input) > 10000:  # Reasonable limit for chunking
        emit('error', {'error': 'Text is too long. Please limit to 10,000 characters.'})
        return

    # Convert filename to full path if audio prompt is provided
    audio_prompt_path = None
    if audio_prompt_filename:
        audio_prompt_path = os.path.join('static', 'uploads', audio_prompt_filename)
        if not os.path.exists(audio_prompt_path):
            emit('error', {'error': 'Reference audio file not found. Please re-upload.'})
            return

    start_time = time.time()
    filename = generate_voice(
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
    print(f"Generation took {duration} seconds")

    # Note: We don't automatically delete reference audio here anymore
    # It will only be deleted when user clicks "Clear Audio" or "Reset to Defaults"

    if filename:
        # Write data to JSON (store original filename, not path, for display)
        write_to_json(text_input, filename, audio_prompt_filename, exaggeration, temperature, 
                     cfg_weight, chunk_size, speed, pitch, reduce_noise, remove_silence, seed, duration)

        # Emit 'generation_complete' event with the filename
        emit('generation_complete', {'filename': filename, 'generation_time': duration})
    else:
        emit('error', {'error': 'Audio generation failed. Please try again.'})

def update_progress(current, total):
    progress = current / total
    emit('generation_progress', {'progress': progress}, broadcast=True)

@app.route('/static/output/<path:filename>', methods=['DELETE'])
def delete_file(filename):
    try:
        file_path = os.path.join('static/output', filename)
        os.remove(file_path)

        # Remove data from JSON
        remove_from_json(filename)
        return jsonify({'message': 'File deleted successfully'})
    except OSError as e:
        return jsonify({'message': 'Error deleting file: ' + str(e)}), 500

def write_to_json(text_input, filename, audio_prompt_path, exaggeration, temperature, 
                 cfg_weight, chunk_size, speed, pitch, reduce_noise, remove_silence, seed, duration):
    data = {}
    json_file = os.path.join('static/json', 'data.json')
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    
    try:
        # Read existing data
        with open(json_file, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Remove the .wav extension from the filename
    file_id = os.path.splitext(filename)[0]

    # Append new data to the beginning of the JSON list
    data = {file_id: {
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
        'outputFile': filename,
        'generationTime': duration
    }, **data}

    # Write the data back to the file with pretty formatting
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

def remove_from_json(filename):
    data = {}
    json_file = os.path.join('static/json', 'data.json')
    try:
        # Read existing data
        with open(json_file, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Remove the .wav extension from the filename
    file_id = os.path.splitext(filename)[0]

    # Remove the data entry for the given filename
    if file_id in data:
        del data[file_id]

    # Check if data is empty after removal
    if not data:
        if os.path.exists(json_file):
            os.remove(json_file)
    else:
        # If not, write the data back to the file with pretty formatting
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)

def cleanup_old_uploads():
    """Clean up old upload files on startup"""
    upload_dir = os.path.join('static', 'uploads')
    if os.path.exists(upload_dir):
        try:
            for filename in os.listdir(upload_dir):
                filepath = os.path.join(upload_dir, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
            print("Cleaned up old upload files")
        except Exception as e:
            print(f"Error cleaning up old uploads: {e}")

if __name__ == '__main__':
    # Ensure output directories exist
    os.makedirs('static/output', exist_ok=True)
    os.makedirs('static/json', exist_ok=True)
    os.makedirs('static/uploads', exist_ok=True)
    
    # Clean up any leftover upload files from previous sessions
    cleanup_old_uploads()
    
    # Note will allow access from other devices on same network.
    socketio.run(app, host='0.0.0.0', debug=True)