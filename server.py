from flask import Flask, render_template, request, send_from_directory, jsonify
from flask_socketio import SocketIO, emit
import os
import json
import time
import uuid
import logging
import glob
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from connector import generate_voice
import config

# Track active sessions for status updates
active_sessions = {}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StatusConsoleHandler(logging.Handler):
    """Custom logging handler that sends log messages to the status console"""
    
    def __init__(self):
        super().__init__()
        self.setLevel(logging.INFO)
        
    def emit(self, record):
        try:
            global active_sessions
            if not active_sessions:
                return
                
            # Skip certain verbose messages
            skip_patterns = [
                'Status:',  # Already handled by update_status  
                'Progress update:',  # Progress bar messages
                'Session ID',
                'Starting Chatterbox',
                'Initialization complete',
                'Cleaned up',  # File cleanup messages
                'Deleted file:',  # File deletion messages
                'Generation completed in',  # Timing messages
                'Set random seed'  # Seed messages
            ]
            
            message = record.getMessage()
            if any(pattern in message for pattern in skip_patterns):
                return
                
            # Only show certain INFO messages and all WARNING/ERROR messages
            if record.levelname == 'INFO':
                # Only show INFO messages from audio processing modules or important events
                important_info_patterns = [
                    'FFmpeg',
                    'Audio normalized',
                    'Audio processed',
                    'speed adjustment',
                    'pitch adjustment',
                    'Noise reduction',
                    'fallback',
                    'falling back',
                    'Using',
                    'not available',
                    'failed'
                ]
                if not any(pattern in message for pattern in important_info_patterns):
                    return
            
            # Format the log message with appropriate emoji and cleaner text
            level_emojis = {
                'INFO': 'ℹ️',
                'WARNING': '⚠️', 
                'ERROR': '❌',
                'CRITICAL': '🚨'
            }
            
            emoji = level_emojis.get(record.levelname, '📝')
            
            # Clean up the message format
            if record.levelname == 'INFO':
                display_message = f"{emoji} {message}"
            else:
                display_message = f"{emoji} {record.levelname}: {message}"
            
            # Send to status console for all active sessions
            for session_id in active_sessions:
                socketio.emit('generation_status', {'message': display_message}, room=session_id)
            socketio.sleep(0)  # Force immediate delivery
        except Exception:
            # Don't let logging errors break the application
            pass

# Create and add the custom handler to capture warnings/errors from all modules
status_handler = StatusConsoleHandler()
root_logger = logging.getLogger()
root_logger.addHandler(status_handler)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_FILE_SIZE
app.config['SECRET_KEY'] = config.SECRET_KEY
socketio = SocketIO(app, max_http_buffer_size=config.MAX_FILE_SIZE, cors_allowed_origins="*", async_mode='threading')

# Add template filter for formatting numbers
@app.template_filter('format_number')
def format_number(value):
    """Format numbers with thousands separators"""
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return str(value)

# Use configuration constants
UPLOAD_FOLDER = config.UPLOAD_FOLDER
OUTPUT_FOLDER = config.OUTPUT_FOLDER
JSON_FOLDER = config.JSON_FOLDER

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return config.is_audio_file(filename)

def allowed_text_file(filename):
    """Check if uploaded file is a text file"""
    return config.is_text_file(filename)

def validate_audio_file(file):
    """Validate uploaded audio file"""
    if not file or file.filename == '':
        return False, 'No file selected'
    
    if not allowed_file(file.filename):
        return False, f'Invalid file type. Allowed: {", ".join(config.ALLOWED_AUDIO_EXTENSIONS)}'
    
    # Check file size (additional check beyond Flask's MAX_CONTENT_LENGTH)
    try:
        file.seek(0, 2)  # Seek to end
        size = file.tell()
        file.seek(0)  # Reset to beginning
    except (OSError, IOError):
        return False, 'Unable to read file'
    
    if size > config.MAX_FILE_SIZE:
        return False, f'File too large (max {config.MAX_FILE_SIZE // (1024*1024)}MB)'
    
    return True, None

def validate_text_file(file):
    """Validate uploaded text file"""
    if not file or file.filename == '':
        return False, 'No file selected'
    
    if not allowed_text_file(file.filename):
        return False, f'Invalid file type. Allowed: {", ".join(config.ALLOWED_TEXT_EXTENSIONS)}'
    
    # Check file size
    try:
        file.seek(0, 2)  # Seek to end
        size = file.tell()
        file.seek(0)  # Reset to beginning
    except (OSError, IOError):
        return False, 'Unable to read file'
    
    if size > config.MAX_FILE_SIZE:
        return False, f'File too large (max {config.MAX_FILE_SIZE // (1024*1024)}MB)'
    
    return True, None

def cleanup_reference_audio(filepath):
    """Clean up reference audio file after generation"""
    if not filepath:
        return
        
    try:
        if os.path.exists(filepath):
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
    return render_template('index.html', max_text_length=config.MAX_TEXT_LENGTH)

@app.route('/static/output/<path:filename>')
def serve_static(filename):
    """Serve generated audio files"""
    # Sanitize filename to prevent directory traversal
    filename = secure_filename(filename)
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/static/json/<path:filename>')
def serve_json(filename):
    """Serve JSON data files"""
    # Sanitize filename to prevent directory traversal
    filename = secure_filename(filename)
    json_path = os.path.join(JSON_FOLDER, filename)
    
    # If file doesn't exist, return empty JSON
    if not os.path.exists(json_path):
        return jsonify({})
    
    return send_from_directory(JSON_FOLDER, filename)

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

@app.route('/get_voice_files', methods=['GET'])
def get_voice_files():
    """Get list of available voice files from static/voices directory"""
    try:
        voices_folder = os.path.join(app.static_folder, 'voices')
        
        # Create voices directory if it doesn't exist
        if not os.path.exists(voices_folder):
            os.makedirs(voices_folder)
            return jsonify({'voices': []})
        
        # Get list of audio files in voices directory
        voice_files = []
        supported_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')
        
        for filename in os.listdir(voices_folder):
            if filename.lower().endswith(supported_extensions):
                file_path = os.path.join(voices_folder, filename)
                if os.path.isfile(file_path):
                    # Get file info
                    file_stats = os.stat(file_path)
                    voice_files.append({
                        'filename': filename,
                        'display_name': os.path.splitext(filename)[0].replace('_', ' ').title(),
                        'size': file_stats.st_size,
                        'modified': file_stats.st_mtime
                    })
        
        # Sort by display name
        voice_files.sort(key=lambda x: x['display_name'])
        
        return jsonify({'voices': voice_files})
        
    except Exception as e:
        logger.error(f"Error getting voice files: {str(e)}")
        return jsonify({'error': f'Failed to get voice files: {str(e)}'}), 500

@app.route('/upload_text_files', methods=['POST'])
def upload_text_files():
    """Handle multiple text file uploads for batch processing"""
    try:
        if 'text_files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('text_files')
        if not files or len(files) == 0:
            return jsonify({'error': 'No files selected'}), 400
        
        uploaded_files = []
        
        for file in files:
            # Validate each file
            is_valid, error_msg = validate_text_file(file)
            if not is_valid:
                return jsonify({'error': f'File {file.filename}: {error_msg}'}), 400
            
            # Generate unique filename to avoid conflicts
            filename = secure_filename(file.filename)
            name, ext = os.path.splitext(filename)
            unique_filename = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
            filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
            
            # Save file
            file.save(filepath)
            
            # Read file content to get text length for validation
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if len(content) > config.MAX_TEXT_LENGTH:
                        os.remove(filepath)  # Clean up the file
                        return jsonify({'error': f'File {file.filename}: Text too long (max {config.MAX_TEXT_LENGTH:,} characters)'}), 400
                    if not content:
                        os.remove(filepath)  # Clean up the file
                        return jsonify({'error': f'File {file.filename}: File is empty'}), 400
            except UnicodeDecodeError:
                os.remove(filepath)  # Clean up the file
                return jsonify({'error': f'File {file.filename}: Invalid text encoding (must be UTF-8)'}), 400
            
            uploaded_files.append({
                'original_name': file.filename,
                'stored_name': unique_filename,
                'filepath': filepath
            })
            
            logger.info(f"Uploaded text file: {unique_filename}")
        
        return jsonify({
            'success': True,
            'files': uploaded_files,
            'count': len(uploaded_files)
        })
        
    except Exception as e:
        logger.error(f"Error uploading text files: {str(e)}")
        return jsonify({'error': f'Failed to upload text files: {str(e)}'}), 500

@app.route('/delete_all_audio', methods=['POST'])
def delete_all_audio():
    """Delete all generated audio files and clear JSON data"""
    try:
        json_file = os.path.join(JSON_FOLDER, 'data.json')
        deleted_count = 0
        failed_count = 0
        
        # First, get list of files from JSON if it exists
        files_to_delete = []
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                files_to_delete = [item.get('outputFile', '') for item in data.values() if item.get('outputFile')]
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f"Error reading JSON data: {str(e)}")
        
        # Also get any files that might be in the output folder but not in JSON
        try:
            output_files = glob.glob(os.path.join(OUTPUT_FOLDER, '*.wav'))
            for file_path in output_files:
                filename = os.path.basename(file_path)
                if filename not in files_to_delete:
                    files_to_delete.append(filename)
        except Exception as e:
            logger.warning(f"Error scanning output folder: {str(e)}")
        
        # Delete each file
        for filename in files_to_delete:
            if filename:  # Skip empty filenames
                filepath = os.path.join(OUTPUT_FOLDER, filename)
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        deleted_count += 1
                        logger.info(f"Deleted audio file: {filename}")
                    else:
                        logger.warning(f"File not found for deletion: {filename}")
                except Exception as e:
                    logger.error(f"Error deleting file {filename}: {str(e)}")
                    failed_count += 1
        
        # Clear or remove the JSON file
        try:
            if os.path.exists(json_file):
                os.remove(json_file)
                logger.info("Cleared JSON data file")
            else:
                logger.info("JSON data file did not exist")
        except Exception as e:
            logger.error(f"Error clearing JSON data: {str(e)}")
            failed_count += 1
        
        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'failed_count': failed_count,
            'message': f'Deleted {deleted_count} files, {failed_count} failures'
        })
        
    except Exception as e:
        logger.error(f"Error in delete_all_audio: {str(e)}")
        return jsonify({'error': f'Failed to delete files: {str(e)}'}), 500

def handle_multiple_text_files(text_files_content, original_data, session_id):
    """Handle generation for multiple text files"""
    try:
        logger.info(f"handle_multiple_text_files called with session_id: {session_id}")
        total_files = len(text_files_content)
        all_results = []
        failed_files = []
        
        status_callback = create_status_callback(session_id)
        status_callback(f"🎯 Starting batch processing for {total_files} file(s)...")
        
        for i, text_file_info in enumerate(text_files_content, 1):
            filename = text_file_info['filename']
            content = text_file_info['content']
            
            try:
                status_callback(f"📝 Processing file {i}/{total_files}: {filename}")
                
                # Create a copy of the original data for this file
                file_data = original_data.copy()
                file_data['text_input'] = content
                file_data['text_files_paths'] = None  # Clear to avoid recursion
                
                # Note: Individual file processing will be handled by generate_voice's default naming
                
                # Call the original generation logic for this single file
                result = process_single_text_generation(file_data, session_id)
                
                if result:
                    result['source_file'] = filename
                    result['file_number'] = i
                    all_results.append(result)
                    status_callback(f"✅ Completed file {i}/{total_files}: {filename}")
                else:
                    failed_files.append(filename)
                    status_callback(f"❌ Failed file {i}/{total_files}: {filename}")
                    
            except Exception as e:
                logger.error(f"Error processing file {filename}: {str(e)}")
                failed_files.append(filename)
                status_callback(f"❌ Error processing file {i}/{total_files}: {filename} - {str(e)}")
        
        # Clean up uploaded text files
        for text_file_info in text_files_content:
            try:
                if os.path.exists(text_file_info['path']):
                    os.remove(text_file_info['path'])
            except Exception as e:
                logger.warning(f"Could not clean up text file {text_file_info['path']}: {str(e)}")
        
        # Send batch completion results
        if all_results:
            status_callback(f"🎉 Batch processing completed! {len(all_results)}/{total_files} files processed successfully.")
            socketio.emit('batch_generation_complete', {
                'results': all_results,
                'total_files': total_files,
                'successful_files': len(all_results),
                'failed_files': failed_files
            }, room=session_id)
            socketio.sleep(0)
        else:
            socketio.emit('error', {'error': 'All text files failed to process.'}, room=session_id)
            socketio.sleep(0)  # Force immediate delivery
            
    except Exception as e:
        logger.error(f"Error in batch text file processing: {str(e)}")
        socketio.emit('error', {'error': f'Batch processing failed: {str(e)}'}, room=session_id)
        socketio.sleep(0)  # Force immediate delivery

def process_single_text_generation(data, session_id):
    """Process a single text generation (extracted from handle_start_generation)"""
    try:
        text_input = data.get('text_input', '').strip()
        
        # Extract and validate basic parameters
        audio_prompt_filename = data.get('audio_prompt_path')
        
        # Convert filename to full path if provided
        audio_prompt_path = None
        if audio_prompt_filename:
            # First check if it's an uploaded file in uploads folder
            upload_path = os.path.join(UPLOAD_FOLDER, audio_prompt_filename)
            voices_path = os.path.join(app.static_folder, 'voices', audio_prompt_filename)
            
            if os.path.exists(upload_path):
                # Use uploaded file
                audio_prompt_path = upload_path
                status_callback = create_status_callback(session_id)
                status_callback(f"🎤 Using uploaded reference audio: {audio_prompt_filename}")
            elif os.path.exists(voices_path):
                # Use preset voice from static/voices directory
                audio_prompt_path = voices_path
                status_callback = create_status_callback(session_id)
                status_callback(f"🎭 Using preset voice: {audio_prompt_filename}")
            else:
                # File not found in either location
                status_callback = create_status_callback(session_id)
                status_callback(f"⚠️ Reference audio file not found: {audio_prompt_filename}")
                audio_prompt_path = None
        
        # Basic TTS parameters with None checking
        exaggeration = float(data.get('exaggeration') or config.DEFAULT_EXAGGERATION)
        temperature = float(data.get('temperature') or config.DEFAULT_TEMPERATURE)
        cfg_weight = float(data.get('cfg_weight') or config.DEFAULT_CFG_WEIGHT)
        chunk_size = int(data.get('chunk_size') or config.DEFAULT_CHUNK_SIZE)
        speed = float(data.get('speed') or config.DEFAULT_SPEED)
        pitch = int(data.get('pitch') or config.DEFAULT_PITCH)
        seed = int(data.get('seed') or config.DEFAULT_SEED)
        
        # Export formats
        export_formats = data.get('export_formats', ['wav'])
        if not export_formats or not isinstance(export_formats, list):
            export_formats = ['wav']
        
        # Advanced generation controls
        num_candidates = int(data.get('num_candidates') or 3)
        max_attempts = int(data.get('max_attempts') or 3)
        
        # Whisper validation controls  
        use_whisper_validation = not bool(data.get('bypass_whisper', False))
        whisper_model_name = data.get('whisper_model', 'medium')
        use_faster_whisper = bool(data.get('use_faster_whisper', True))
        use_longest_transcript = bool(data.get('use_longest_transcript', True))
        validation_threshold = float(data.get('validation_threshold') or 0.85)
        
        # Parallel processing controls
        enable_parallel = bool(data.get('enable_parallel', True))
        parallel_workers = int(data.get('parallel_workers') or 4)
        
        # Text processing controls
        to_lowercase = bool(data.get('to_lowercase', True))
        normalize_spacing = bool(data.get('normalize_spacing', True))
        fix_dot_letters = bool(data.get('fix_dot_letters', True))
        remove_reference_numbers = bool(data.get('remove_reference_numbers', True))
        
        # Debug logging
        logger.info(f"Text processing parameters received: to_lowercase={to_lowercase}, normalize_spacing={normalize_spacing}, fix_dot_letters={fix_dot_letters}, remove_reference_numbers={remove_reference_numbers}")
        
        # Sound word replacements
        sound_words = data.get('sound_words', '')
        logger.info(f"Sound words received from frontend: {repr(sound_words)}")
        
        # Audio processing controls
        normalize_audio = bool(data.get('normalize_audio', False))
        normalize_method = data.get('normalize_method', 'ebu')
        normalize_level = int(data.get('normalize_level') or -24)
        normalize_tp = int(data.get('normalize_tp') or -2)
        normalize_lra = int(data.get('normalize_lra') or 7)
        reduce_noise = bool(data.get('reduce_noise', False))
        noise_reduction_method = data.get('noise_reduction_method', 'spectral_gating')
        noise_strength = float(data.get('noise_strength') or 0.85)
        remove_silence = bool(data.get('remove_silence', False))
        
        # Generate audio with advanced controls
        progress_callback = create_progress_callback(session_id)
        status_callback = create_status_callback(session_id)
        
        status_callback("🎯 Starting voice generation process...")
        start_time = time.time()
        
        # Call the generation function
        result = generate_voice(
            text_input=text_input,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
            chunk_size=chunk_size,
            speed=speed,
            pitch=pitch,
            seed=seed,
            reduce_noise=reduce_noise,
            remove_silence=remove_silence,
            progress_callback=progress_callback,
            status_callback=status_callback,
            export_formats=export_formats,
            num_candidates=num_candidates,
            max_attempts=max_attempts,
            use_whisper_validation=use_whisper_validation,
            whisper_model_name=whisper_model_name,
            use_faster_whisper=use_faster_whisper,
            use_longest_transcript=use_longest_transcript,
            validation_threshold=validation_threshold,
            enable_parallel=enable_parallel,
            num_workers=parallel_workers,
            to_lowercase=to_lowercase,
            normalize_spacing=normalize_spacing,
            fix_dot_letters=fix_dot_letters,
            remove_reference_numbers=remove_reference_numbers,
            sound_words=sound_words,
            use_ffmpeg_normalize=normalize_audio,
            normalize_method=normalize_method,
            integrated_loudness=normalize_level,
            true_peak=normalize_tp,
            loudness_range=normalize_lra,
            noise_reduction_method=noise_reduction_method,
            noise_strength=noise_strength
        )
        
        end_time = time.time()
        duration = round(end_time - start_time, 2)
        
        # Handle the returned tuple (filename_or_filenames, actual_seed, processed_text)
        if result and len(result) >= 2:
            if len(result) == 3:
                output_files, actual_seed, processed_text_result = result
            else:
                # Backward compatibility for old 2-tuple format
                output_files, actual_seed = result
                processed_text_result = text_input
        else:
            return None
        
        logger.debug(f"Generation completed in {duration} seconds")
        
        if output_files:
            # Determine primary filename for JSON saving
            if isinstance(output_files, list):
                filename = output_files[0] if output_files else None
            else:
                filename = output_files
            
            if filename:
                # Save generation data to JSON with actual seed used and advanced parameters
                write_to_json(
                    processed_text_result, filename, audio_prompt_filename, exaggeration, temperature, 
                    cfg_weight, chunk_size, speed, pitch, reduce_noise, remove_silence, actual_seed, duration,
                    # Additional parameters for complete record
                    export_formats=export_formats, num_candidates=num_candidates, use_whisper_validation=use_whisper_validation,
                    whisper_model_name=whisper_model_name, use_faster_whisper=use_faster_whisper, use_longest_transcript=use_longest_transcript,
                    validation_threshold=validation_threshold, enable_parallel=enable_parallel, parallel_workers=parallel_workers,
                    to_lowercase=to_lowercase, normalize_spacing=normalize_spacing, fix_dot_letters=fix_dot_letters, remove_reference_numbers=remove_reference_numbers,
                    sound_words=sound_words, normalize_audio=normalize_audio, normalize_method=normalize_method,
                    noise_reduction_method=noise_reduction_method, noise_strength=noise_strength
                )
                
                # Return structured result
                return {
                    'output_files': output_files,
                    'actual_seed': actual_seed,
                    'duration': duration,
                    'primary_filename': filename
                }
        
        return None
        
    except Exception as e:
        logger.error(f"Error in single text generation: {str(e)}")
        return None

@socketio.on('start_generation')
def handle_start_generation(data):
    """Handle TTS generation request with advanced controls"""
    try:
        # Get session ID for status updates
        from flask import request as flask_request
        session_id = flask_request.sid
        
        # Add session to active sessions
        global active_sessions
        active_sessions[session_id] = True
        
        # Create callbacks for this session
        status_callback = create_status_callback(session_id)
        
        # Test status message
        status_callback("🔄 Generation request received...")
        # Validate input data - handle both single text and multiple text files
        text_input = data.get('text_input', '').strip()
        text_files_paths = data.get('text_files_paths')
        
        # Check if we have text files to process
        if text_files_paths and isinstance(text_files_paths, list) and len(text_files_paths) > 0:
            # Process multiple text files
            status_callback(f"📚 Processing {len(text_files_paths)} text file(s)...")
            
            # Validate text files exist and read their content
            text_files_content = []
            for file_path in text_files_paths:
                try:
                    if not os.path.exists(file_path):
                        socketio.emit('error', {'error': f'Text file not found: {file_path}'}, room=session_id)
                        return
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if not content:
                            socketio.emit('error', {'error': f'Text file is empty: {os.path.basename(file_path)}'}, room=session_id)
                            return
                        if len(content) > config.MAX_TEXT_LENGTH:
                            socketio.emit('error', {'error': f'Text in file {os.path.basename(file_path)} is too long. Please limit to {config.MAX_TEXT_LENGTH:,} characters.'}, room=session_id)
                            return
                        
                        text_files_content.append({
                            'path': file_path,
                            'filename': os.path.basename(file_path),
                            'content': content
                        })
                except Exception as e:
                    socketio.emit('error', {'error': f'Error reading text file {os.path.basename(file_path)}: {str(e)}'}, room=session_id)
                    return
            
            # Process each text file separately
            return handle_multiple_text_files(text_files_content, data, session_id)
            
        elif text_input:
            # Process single text input from textarea
            if len(text_input) > config.MAX_TEXT_LENGTH:
                socketio.emit('error', {'error': f'Text is too long. Please limit to {config.MAX_TEXT_LENGTH:,} characters.'}, room=session_id)
                return
        else:
            # No text input or text files provided
            socketio.emit('error', {'error': 'No text provided. Please enter text in the input box or upload text files.'}, room=session_id)
            return

        # Process single text input using the refactored function
        result = process_single_text_generation(data, session_id)
        
        if result:
            # Extract result information
            output_files = result.get('output_files')
            actual_seed = result.get('actual_seed')
            duration = result.get('duration')
            
            # Determine primary filename for response
            if isinstance(output_files, list):
                filename = output_files[0] if output_files else None
            else:
                filename = output_files
            
            if filename:
                socketio.emit('generation_complete', {'filename': filename, 'generation_time': duration}, room=session_id)
                socketio.sleep(0)
            else:
                socketio.emit('error', {'error': 'Audio generation failed. Please try again.'}, room=session_id)
                socketio.sleep(0)
        else:
            socketio.emit('error', {'error': 'Audio generation failed. Please try again.'}, room=session_id)
            socketio.sleep(0)
            
    except Exception as e:
        logger.error(f"Error in generation: {str(e)}")
        socketio.emit('error', {'error': f'Generation failed: {str(e)}'}, room=session_id)
        socketio.sleep(0)
    finally:
        # Clean up session from active sessions
        if session_id in active_sessions:
            del active_sessions[session_id]

def create_progress_callback(session_id):
    """Create a progress callback function for a specific session"""
    def update_progress(current, total):
        try:
            progress = current / total if total > 0 else 0
            # Ensure progress is between 0 and 1
            progress = max(0.0, min(1.0, progress))
            socketio.emit('generation_progress', {'progress': progress}, room=session_id)
            # Force immediate delivery of the message
            socketio.sleep(0)
        except Exception as e:
            logger.error(f"Error updating progress: {str(e)}")
    return update_progress

def create_status_callback(session_id):
    """Create a status callback function for a specific session"""
    def update_status(message):
        try:
            logger.info(f"Status: {message}")
            socketio.emit('generation_status', {'message': message}, room=session_id)
            # Force immediate delivery of the message
            socketio.sleep(0)
        except Exception as e:
            logger.error(f"Error sending status update: {str(e)}")
    return update_status

# Legacy functions for backward compatibility
def update_progress(current, total):
    """Legacy function - use create_progress_callback instead"""
    pass

def update_status(message):
    """Legacy function - use create_status_callback instead"""
    logger.info(f"Status: {message}")

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
        logger.debug(f"Deleted file: {filename}")
        
        return jsonify({'message': 'File deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting file {filename}: {str(e)}")
        return jsonify({'error': f'Error deleting file: {str(e)}'}), 500

def write_to_json(text_input, filename, audio_prompt_path, exaggeration, temperature, 
                 cfg_weight, chunk_size, speed, pitch, reduce_noise, remove_silence, seed, duration,
                 **kwargs):
    """Write generation data to JSON file with support for additional parameters"""
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
            'actualSeed': seed,  # Store the actual seed used
            'outputFile': filename,
            'generationTime': duration,
            'timestamp': time.time(),
            # Advanced parameters
            'exportFormats': kwargs.get('export_formats', ['wav']),
            'numCandidates': kwargs.get('num_candidates', 3),
            'useWhisperValidation': kwargs.get('use_whisper_validation', True),
            'whisperModelName': kwargs.get('whisper_model_name', 'medium'),
            'useFasterWhisper': kwargs.get('use_faster_whisper', True),
            'useLongestTranscript': kwargs.get('use_longest_transcript', True),
            'validationThreshold': kwargs.get('validation_threshold', 0.85),
            'enableParallel': kwargs.get('enable_parallel', True),
            'numWorkers': kwargs.get('num_workers', 4),
            # Text processing parameters
            'toLowercase': kwargs.get('to_lowercase', True),
            'normalizeSpacing': kwargs.get('normalize_spacing', True),
            'fixDotLetters': kwargs.get('fix_dot_letters', True),
            'removeReferenceNumbers': kwargs.get('remove_reference_numbers', True),
            'soundWords': kwargs.get('sound_words', ''),
            # Audio post-processing parameters
            'useFfmpegNormalize': kwargs.get('use_ffmpeg_normalize', False),
            'normalizeMethod': kwargs.get('normalize_method', 'ebu'),
            'noiseReductionMethod': kwargs.get('noise_reduction_method', 'afftdn'),
            'noiseStrength': kwargs.get('noise_strength', 0.85),
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
            logger.debug("Cleaned up old upload files")
    except Exception as e:
        logger.error(f"Error cleaning up old uploads: {e}")

if __name__ == '__main__':
    # Only run initialization once (not in reloader subprocess)
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        # Ensure output directories exist
        ensure_directories()
        
        # Clean up any leftover upload files from previous sessions
        cleanup_old_uploads()
        
        logger.info("Initialization complete")
    
    # Use configuration for server settings
    logger.info(f"Starting Chatterbox Web UI on {config.HOST}:{config.PORT}")
    socketio.run(app, host=config.HOST, port=config.PORT, debug=config.DEBUG)