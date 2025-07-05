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
from connector import generate_voice, generate_multivoice_audio
import config
import re
from typing import List
import cancellation_state

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
                    'failed',
                    'voice similarity',
                    'Voice similarity',
                    'voice consistent',
                    'voice inconsistent',
                    'Voice consistent',
                    'Voice inconsistent',
                    'Candidate',
                    'Rejected candidate',
                    'librosa',
                    'Extracting',
                    'features',
                    'MFCC',
                    'spectral',
                    'pitch',
                    'energy',
                    'tempo',
                    'formant',
                    'Accent difference',
                    'Speed difference',
                    'SEVERE accent drift',
                    'American influence',
                    'voice timbre',
                    'accent drift'
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
    """Get list of available voice files and voice library profiles"""
    try:
        # Legacy voice files from static/voices directory
        voices_folder = os.path.join(app.static_folder, 'voices')
        
        # Enhanced voice library from voice_library directory
        voice_library_folder = os.path.join('voice_library')
        
        # Create directories if they don't exist
        os.makedirs(voices_folder, exist_ok=True)
        os.makedirs(voice_library_folder, exist_ok=True)
        
        voice_files = []
        voice_profiles = []
        supported_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')
        
        # Get legacy voice files
        for filename in os.listdir(voices_folder):
            if filename.lower().endswith(supported_extensions):
                file_path = os.path.join(voices_folder, filename)
                if os.path.isfile(file_path):
                    file_stats = os.stat(file_path)
                    voice_files.append({
                        'filename': filename,
                        'display_name': os.path.splitext(filename)[0].replace('_', ' ').title(),
                        'size': file_stats.st_size,
                        'modified': file_stats.st_mtime,
                        'type': 'legacy'
                    })
        
        # Get enhanced voice library profiles
        for item in os.listdir(voice_library_folder):
            profile_path = os.path.join(voice_library_folder, item)
            if os.path.isdir(profile_path):
                config_file = os.path.join(profile_path, 'config.json')
                if os.path.exists(config_file):
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                        
                        # Find audio file in profile directory
                        audio_file = None
                        for audio_filename in os.listdir(profile_path):
                            if audio_filename.lower().endswith(supported_extensions):
                                audio_file = audio_filename
                                break
                        
                        if audio_file:
                            audio_path = os.path.join(profile_path, audio_file)
                            file_stats = os.stat(audio_path)
                            
                            voice_profiles.append({
                                'profile_name': item,
                                'display_name': config.get('display_name', item),
                                'description': config.get('description', ''),
                                'audio_file': audio_file,
                                'audio_path': os.path.join('voice_library', item, audio_file),
                                'size': file_stats.st_size,
                                'modified': file_stats.st_mtime,
                                'settings': {
                                    'exaggeration': config.get('exaggeration', 0.5),
                                    'temperature': config.get('temperature', 0.3),
                                    'cfg_weight': config.get('cfg_weight', 0.7)
                                },
                                'type': 'profile'
                            })
                    except Exception as e:
                        logger.warning(f"Error loading voice profile {item}: {str(e)}")
                        continue
        
        # Sort both lists
        voice_files.sort(key=lambda x: x['display_name'])
        voice_profiles.sort(key=lambda x: x['display_name'])
        
        return jsonify({
            'legacy_voices': voice_files,
            'voice_profiles': voice_profiles,
            'total_legacy': len(voice_files),
            'total_profiles': len(voice_profiles)
        })
        
    except Exception as e:
        logger.error(f"Error getting voice files: {str(e)}")
        return jsonify({'error': f'Failed to get voice files: {str(e)}'}), 500


@app.route('/voice_profiles', methods=['POST'])
def create_voice_profile():
    """Create a new voice profile in the voice library"""
    try:
        data = request.get_json()
        
        # Validate required fields
        profile_name = data.get('profile_name', '').strip()
        display_name = data.get('display_name', '').strip()
        description = data.get('description', '').strip()
        audio_filename = data.get('audio_filename', '').strip()
        
        if not profile_name or not display_name or not audio_filename:
            return jsonify({'error': 'Profile name, display name, and audio file are required'}), 400
        
        # Sanitize profile name for filesystem
        safe_profile_name = "".join(c for c in profile_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_profile_name = safe_profile_name.replace(' ', '_').lower()
        
        if not safe_profile_name:
            return jsonify({'error': 'Invalid profile name'}), 400
        
        # Create profile directory
        voice_library_folder = os.path.join('voice_library')
        profile_dir = os.path.join(voice_library_folder, safe_profile_name)
        
        if os.path.exists(profile_dir):
            return jsonify({'error': 'Voice profile already exists'}), 400
        
        os.makedirs(profile_dir, exist_ok=True)
        
        # Move audio file from uploads to profile directory
        upload_path = os.path.join(UPLOAD_FOLDER, audio_filename)
        if not os.path.exists(upload_path):
            return jsonify({'error': 'Audio file not found in uploads'}), 400
        
        # Determine audio file extension
        _, ext = os.path.splitext(audio_filename)
        profile_audio_filename = f"voice{ext}"
        profile_audio_path = os.path.join(profile_dir, profile_audio_filename)
        
        # Move the file
        import shutil
        shutil.move(upload_path, profile_audio_path)
        
        # Create profile configuration
        profile_config = {
            'profile_name': safe_profile_name,
            'display_name': display_name,
            'description': description,
            'audio_file': profile_audio_filename,
            'created_at': time.time(),
            'exaggeration': float(data.get('exaggeration', 0.5)),
            'temperature': float(data.get('temperature', 0.3)),
            'cfg_weight': float(data.get('cfg_weight', 0.7)),
            'version': '1.0'
        }
        
        # Save configuration
        config_path = os.path.join(profile_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(profile_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created voice profile: {safe_profile_name}")
        
        return jsonify({
            'success': True,
            'profile_name': safe_profile_name,
            'display_name': display_name,
            'message': f'Voice profile "{display_name}" created successfully'
        })
        
    except Exception as e:
        logger.error(f"Error creating voice profile: {str(e)}")
        return jsonify({'error': f'Failed to create voice profile: {str(e)}'}), 500


@app.route('/voice_profiles/<profile_name>', methods=['DELETE'])
def delete_voice_profile(profile_name):
    """Delete a voice profile from the voice library"""
    try:
        # Sanitize profile name
        safe_profile_name = secure_filename(profile_name)
        profile_dir = os.path.join('voice_library', safe_profile_name)
        
        if not os.path.exists(profile_dir):
            return jsonify({'error': 'Voice profile not found'}), 404
        
        # Remove entire profile directory
        import shutil
        shutil.rmtree(profile_dir)
        
        logger.info(f"Deleted voice profile: {safe_profile_name}")
        
        return jsonify({
            'success': True,
            'message': f'Voice profile "{profile_name}" deleted successfully'
        })
        
    except Exception as e:
        logger.error(f"Error deleting voice profile {profile_name}: {str(e)}")
        return jsonify({'error': f'Failed to delete voice profile: {str(e)}'}), 500


@app.route('/voice_profiles/<profile_name>/settings', methods=['GET'])
def get_voice_profile_settings(profile_name):
    """Get settings for a specific voice profile"""
    try:
        safe_profile_name = secure_filename(profile_name)
        config_path = os.path.join('voice_library', safe_profile_name, 'config.json')
        
        if not os.path.exists(config_path):
            return jsonify({'error': 'Voice profile not found'}), 404
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return jsonify({
            'success': True,
            'settings': {
                'exaggeration': config.get('exaggeration', 0.5),
                'temperature': config.get('temperature', 0.3),
                'cfg_weight': config.get('cfg_weight', 0.7),
                'display_name': config.get('display_name', profile_name),
                'description': config.get('description', ''),
                'audio_path': os.path.join('voice_library', safe_profile_name, config.get('audio_file', 'voice.wav'))
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting voice profile settings {profile_name}: {str(e)}")
        return jsonify({'error': f'Failed to get voice profile settings: {str(e)}'}), 500


@app.route('/parse_multivoice_text', methods=['POST'])
def parse_multivoice_text():
    """Parse text with multi-voice character markers and return character assignments"""
    try:
        data = request.get_json()
        text_content = data.get('text_content', '').strip()
        
        if not text_content:
            return jsonify({'error': 'No text content provided'}), 400
        
        # Parse multi-voice text using audiobook-inspired parsing
        characters = parse_multi_voice_characters(text_content)
        
        if not characters:
            return jsonify({
                'success': True,
                'is_multivoice': False,
                'characters': [],
                'message': 'No character markers found - text will use single voice'
            })
        
        return jsonify({
            'success': True,
            'is_multivoice': True,
            'characters': characters,
            'character_count': len(characters),
            'message': f'Found {len(characters)} characters for multi-voice processing'
        })
        
    except Exception as e:
        logger.error(f"Error parsing multi-voice text: {str(e)}")
        return jsonify({'error': f'Failed to parse multi-voice text: {str(e)}'}), 500


@app.route('/create_multivoice_project', methods=['POST'])
def create_multivoice_project():
    """Create a multi-voice audiobook project with character voice assignments"""
    try:
        data = request.get_json()
        
        # Validate required fields
        project_name = data.get('project_name', '').strip()
        text_content = data.get('text_content', '').strip()
        voice_assignments = data.get('voice_assignments', {})
        
        if not project_name or not text_content or not voice_assignments:
            return jsonify({'error': 'Project name, text content, and voice assignments are required'}), 400
        
        # Create project directory
        projects_dir = os.path.join('audiobook_projects')
        os.makedirs(projects_dir, exist_ok=True)
        
        safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_project_name = safe_project_name.replace(' ', '_').lower()
        
        project_dir = os.path.join(projects_dir, safe_project_name)
        
        if os.path.exists(project_dir):
            return jsonify({'error': 'Project already exists'}), 400
        
        os.makedirs(project_dir, exist_ok=True)
        
        # Parse characters from text
        characters = parse_multi_voice_characters(text_content)
        
        # Validate voice assignments
        for character in characters:
            if character not in voice_assignments:
                return jsonify({'error': f'Voice assignment missing for character: {character}'}), 400
        
        # Create project metadata
        project_metadata = {
            'project_name': safe_project_name,
            'display_name': project_name,
            'project_type': 'multivoice',
            'created_at': time.time(),
            'characters': characters,
            'voice_assignments': voice_assignments,
            'text_length': len(text_content),
            'word_count': len(text_content.split()),
            'status': 'created',
            'version': '1.0'
        }
        
        # Save project metadata
        metadata_path = os.path.join(project_dir, 'project_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(project_metadata, f, indent=2, ensure_ascii=False)
        
        # Save text content
        text_path = os.path.join(project_dir, 'source_text.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        logger.info(f"Created multi-voice project: {safe_project_name}")
        
        return jsonify({
            'success': True,
            'project_name': safe_project_name,
            'project_dir': project_dir,
            'characters': characters,
            'message': f'Multi-voice project "{project_name}" created successfully'
        })
        
    except Exception as e:
        logger.error(f"Error creating multi-voice project: {str(e)}")
        return jsonify({'error': f'Failed to create multi-voice project: {str(e)}'}), 500


def get_available_voice_files():
    """Get available voice files from static/voices directory"""
    voices_folder = os.path.join(app.static_folder, 'voices')
    os.makedirs(voices_folder, exist_ok=True)
    
    voice_files = {}
    supported_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')
    
    for filename in os.listdir(voices_folder):
        if filename.lower().endswith(supported_extensions):
            file_path = os.path.join(voices_folder, filename)
            if os.path.isfile(file_path):
                name_without_ext = os.path.splitext(filename)[0].lower()
                voice_files[name_without_ext] = filename
    
    return voice_files

def auto_match_character_voices(characters, available_voices):
    """Auto-match character names to available voice files (returns filenames, not full paths)"""
    matched_voices = {}
    
    for character in characters:
        character_lower = character.lower()
        
        # Direct match
        if character_lower in available_voices:
            voice_file = available_voices[character_lower]
            matched_voices[character] = voice_file
            continue
        
        # Fuzzy matching - check if character name is contained in voice filename
        best_match = None
        for voice_name, voice_file in available_voices.items():
            if character_lower in voice_name or voice_name in character_lower:
                best_match = voice_file
                break
        
        if best_match:
            matched_voices[character] = best_match
        else:
            # No match found - will use default voice or first available
            matched_voices[character] = None
    
    return matched_voices

def parse_multi_voice_characters(text: str) -> List[str]:
    """Parse text and extract character names from multi-voice markers"""
    import re
    
    # Find character markers like [CHARACTER_NAME] or [Narrator]
    character_pattern = r'\[([^\]]+)\]'
    matches = re.findall(character_pattern, text)
    
    # Clean and deduplicate character names
    characters = []
    seen = set()
    
    for match in matches:
        character = match.strip()
        if character and character not in seen:
            characters.append(character)
            seen.add(character)
    
    return characters

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
                    raw_content = f.read()
                    content = normalize_text_content(raw_content)
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
    """Enhanced batch processing for multiple text files with audiobook-inspired improvements"""
    try:
        logger.info(f"Enhanced batch processing called with session_id: {session_id}")
        total_files = len(text_files_content)
        all_results = []
        failed_files = []
        
        status_callback = create_status_callback(session_id)
        status_callback(f"📚 Starting enhanced batch processing for {total_files} file(s)...")
        
        # Enhanced batch processing with memory management
        from connector import background_memory_management
        background_memory_management(total_files)
        
        # Calculate total text length for optimal processing
        total_text_length = sum(len(f['content']) for f in text_files_content)
        logger.info(f"Batch processing: {total_files} files, {total_text_length:,} total characters")
        
        # Progress tracking for real-time updates
        for i, text_file_info in enumerate(text_files_content, 1):
            filename = text_file_info['filename']
            content = text_file_info['content']
            
            try:
                status_callback(f"📖 Processing file {i}/{total_files}: {filename} ({len(content):,} chars)")
                
                # Enhanced file processing with error recovery
                file_data = original_data.copy()
                file_data['text_input'] = content
                file_data['text_files_paths'] = None  # Clear to avoid recursion
                
                # Memory management between files for large batches
                if i > 1 and i % 3 == 0:  # Every 3 files
                    background_memory_management(1)
                    status_callback(f"🧹 Memory management (file {i}/{total_files})")
                
                # Process single file with enhanced error recovery
                result = process_single_text_generation_enhanced(file_data, session_id, filename)
                
                if result:
                    result['source_file'] = filename
                    result['file_number'] = i
                    result['text_length'] = len(content)
                    all_results.append(result)
                    status_callback(f"✅ Completed file {i}/{total_files}: {filename}")
                    
                    # Emit individual file completion for real-time UI update
                    socketio.emit('file_generation_complete', {
                        'filename': result.get('primary_filename', 'unknown'),
                        'source_file': filename,
                        'file_number': i,
                        'total_files': total_files,
                        'generation_time': result.get('duration', 0)
                    }, room=session_id)
                    socketio.sleep(0)
                else:
                    failed_files.append(filename)
                    status_callback(f"❌ Failed file {i}/{total_files}: {filename}")
                    
            except Exception as e:
                logger.error(f"Error processing file {filename}: {str(e)}")
                failed_files.append(filename)
                status_callback(f"❌ Error processing file {i}/{total_files}: {filename} - {str(e)}")
                
                # Enhanced error recovery - continue with next file
                logger.info(f"Continuing batch processing despite error in {filename}")
        
        # Clean up uploaded text files
        cleanup_batch_files(text_files_content)
        
        # Enhanced batch completion with detailed statistics
        if all_results:
            total_duration = sum(r.get('duration', 0) for r in all_results)
            avg_duration = total_duration / len(all_results) if all_results else 0
            
            status_callback(f"🎉 Batch completed! {len(all_results)}/{total_files} files processed (avg: {avg_duration:.1f}s/file)")
            
            socketio.emit('batch_generation_complete', {
                'results': all_results,
                'total_files': total_files,
                'successful_files': len(all_results),
                'failed_files': failed_files,
                'total_duration': total_duration,
                'average_duration': avg_duration,
                'total_text_length': total_text_length
            }, room=session_id)
            socketio.sleep(0)
        else:
            socketio.emit('error', {'error': 'All text files failed to process.'}, room=session_id)
            socketio.sleep(0)
            
    except Exception as e:
        logger.error(f"Error in enhanced batch processing: {str(e)}")
        socketio.emit('error', {'error': f'Enhanced batch processing failed: {str(e)}'}, room=session_id)
        socketio.sleep(0)


def process_single_text_generation_enhanced(data, session_id, source_filename=None):
    """Enhanced single text generation with better error recovery"""
    try:
        # Call original function with enhanced error handling
        result = process_single_text_generation(data, session_id, source_filename)
        
        if result and source_filename:
            # Add enhanced metadata
            result['processing_mode'] = 'enhanced_batch'
            result['source_filename'] = source_filename
            
        return result
        
    except Exception as e:
        logger.error(f"Enhanced processing failed for {source_filename}: {str(e)}")
        return None


def cleanup_batch_files(text_files_content):
    """Enhanced cleanup for batch processing"""
    cleanup_count = 0
    for text_file_info in text_files_content:
        try:
            if os.path.exists(text_file_info['path']):
                os.remove(text_file_info['path'])
                cleanup_count += 1
        except Exception as e:
            logger.warning(f"Could not clean up text file {text_file_info['path']}: {str(e)}")
    
    logger.info(f"Cleaned up {cleanup_count}/{len(text_files_content)} batch files")

def process_single_text_generation(data, session_id, source_filename=None):
    """Process a single text generation (extracted from handle_start_generation)"""
    try:
        text_input = data.get('text_input', '').strip()
        
        # Extract and validate basic parameters
        audio_prompt_filename = data.get('audio_prompt_path')
        voice_profile = data.get('voice_profile')
        
        # Convert filename to full path if provided with enhanced validation
        audio_prompt_path = None
        voice_profile_settings = None
        if audio_prompt_filename:
            # Sanitize filename to prevent path traversal
            safe_filename = secure_filename(audio_prompt_filename)
            if safe_filename != audio_prompt_filename:
                logger.warning(f"Audio prompt filename sanitized: {audio_prompt_filename} -> {safe_filename}")
                audio_prompt_filename = safe_filename
            
            # First check if it's an uploaded file in uploads folder
            upload_path = os.path.join(UPLOAD_FOLDER, audio_prompt_filename)
            voices_path = os.path.join(app.static_folder, 'voices', audio_prompt_filename)
            
            status_callback = create_status_callback(session_id)
            
            if os.path.exists(upload_path):
                # Use uploaded file
                audio_prompt_path = upload_path
                file_size = os.path.getsize(upload_path)
                status_callback(f"🎤 Using uploaded reference audio: {audio_prompt_filename} ({file_size:,} bytes)")
                logger.info(f"Reference audio found (uploaded): {upload_path}")
            elif os.path.exists(voices_path):
                # Use preset voice from static/voices directory
                audio_prompt_path = voices_path
                file_size = os.path.getsize(voices_path)
                status_callback(f"🎭 Using preset voice: {audio_prompt_filename} ({file_size:,} bytes)")
                logger.info(f"Reference audio found (preset): {voices_path}")
            else:
                # File not found in either location - this is critical for voice consistency
                error_msg = f"❌ CRITICAL: Reference audio file not found: {audio_prompt_filename}"
                status_callback(error_msg)
                logger.error(f"Reference audio not found in either location:")
                logger.error(f"  Upload path: {upload_path}")
                logger.error(f"  Voices path: {voices_path}")
                logger.error("This will cause voice to fall back to default!")
                audio_prompt_path = None
        else:
            logger.info("No reference audio filename provided - checking for voice profile")
        
        # Handle voice profile if specified and no audio file was provided
        if voice_profile and not audio_prompt_path:
            status_callback = create_status_callback(session_id)
            voice_profile_path = os.path.join('voice_library', voice_profile)
            config_path = os.path.join(voice_profile_path, 'config.json')
            
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        voice_profile_settings = json.load(f)
                    
                    # Get the audio file from the voice profile
                    audio_file = voice_profile_settings.get('audio_file')
                    if audio_file:
                        profile_audio_path = os.path.join(voice_profile_path, audio_file)
                        if os.path.exists(profile_audio_path):
                            audio_prompt_path = profile_audio_path
                            file_size = os.path.getsize(profile_audio_path)
                            display_name = voice_profile_settings.get('display_name', voice_profile)
                            status_callback(f"🎭 Using voice profile: {display_name} ({file_size:,} bytes)")
                            logger.info(f"Voice profile audio loaded: {profile_audio_path}")
                        else:
                            error_msg = f"❌ Voice profile audio file not found: {profile_audio_path}"
                            status_callback(error_msg)
                            logger.error(error_msg)
                    else:
                        error_msg = f"❌ Voice profile missing audio_file setting: {voice_profile}"
                        status_callback(error_msg)
                        logger.error(error_msg)
                        
                except Exception as e:
                    error_msg = f"❌ Error loading voice profile {voice_profile}: {str(e)}"
                    status_callback(error_msg)
                    logger.error(error_msg)
            else:
                error_msg = f"❌ Voice profile not found: {voice_profile}"
                status_callback(error_msg)
                logger.error(error_msg)
        
        if not audio_prompt_path and not voice_profile:
            logger.info("No reference audio or voice profile - will use model default voice")
        
        # Basic TTS parameters with None checking
        # Use voice profile settings as defaults if available
        profile_exaggeration = voice_profile_settings.get('exaggeration', config.DEFAULT_EXAGGERATION) if voice_profile_settings else config.DEFAULT_EXAGGERATION
        profile_temperature = voice_profile_settings.get('temperature', config.DEFAULT_TEMPERATURE) if voice_profile_settings else config.DEFAULT_TEMPERATURE
        profile_cfg_weight = voice_profile_settings.get('cfg_weight', config.DEFAULT_CFG_WEIGHT) if voice_profile_settings else config.DEFAULT_CFG_WEIGHT
        
        exaggeration = float(data.get('exaggeration') or profile_exaggeration)
        temperature = float(data.get('temperature') or profile_temperature)
        cfg_weight = float(data.get('cfg_weight') or profile_cfg_weight)
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
        
        # Voice similarity validation (enabled by default for consistency)
        voice_similarity_threshold = float(data.get('voice_similarity_threshold', config.DEFAULT_VOICE_SIMILARITY_THRESHOLD))
        
        # Check if generation was cancelled before starting
        if is_generation_cancelled(session_id):
            logger.info(f"Generation cancelled before starting for session {session_id}")
            return None
        
        # Generate audio with advanced controls
        progress_callback = create_progress_callback(session_id)
        status_callback = create_status_callback(session_id)
        
        status_callback("🎯 Starting voice generation process...")
        start_time = time.time()
        
        # Check for multi-voice character markers
        import re
        status_callback(f"🔍 Checking text for multivoice markers: {text_input[:100]}...")
        character_markers = re.findall(r'\[([^\]]+)\]', text_input)
        status_callback(f"🔍 Found {len(character_markers)} character markers: {character_markers}")
        
        if character_markers:
            # Multi-voice generation detected
            characters = list(set(character_markers))  # Remove duplicates
            status_callback(f"🎭 Multi-voice text detected with {len(characters)} characters: {', '.join(characters)}")
            
            # Check if this is from a multi-voice project
            multivoice_project = data.get('multivoice_project')
            character_voices = data.get('character_voices', {})
            
            # Auto-match characters to available voices if no explicit assignments
            if not multivoice_project and not character_voices:
                status_callback("🔍 Auto-matching characters to available voices...")
                available_voices = get_available_voice_files()
                character_voices = auto_match_character_voices(characters, available_voices)
                
                # Log matching results
                matched_count = sum(1 for voice in character_voices.values() if voice is not None)
                status_callback(f"✅ Auto-matched {matched_count}/{len(characters)} characters to voices")
                for char, voice_file in character_voices.items():
                    if voice_file:
                        status_callback(f"  • {char} → {voice_file}")
                    else:
                        status_callback(f"  • {char} → (using default voice)")
            
            if multivoice_project or character_voices:
                status_callback(f"🎬 Processing {len(set(character_markers))} characters with assigned voices...")
                
                # Load character voices from project if needed
                if multivoice_project and not character_voices:
                    project_path = os.path.join("audiobook_projects", multivoice_project)
                    metadata_path = os.path.join(project_path, "project_metadata.json")
                    
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r') as f:
                                project_data = json.load(f)
                            character_voices = project_data.get('character_voices', {})
                            status_callback(f"📁 Loaded character voices from project: {multivoice_project}")
                        except Exception as e:
                            logger.error(f"Error loading project voices: {e}")
                            status_callback(f"⚠️ Error loading project voices, using default")
                
                # Convert relative paths to full paths for character voices
                full_character_voices = {}
                for char, voice_file in character_voices.items():
                    if voice_file:
                        # Check uploads folder first, then voices folder
                        upload_path = os.path.join(UPLOAD_FOLDER, voice_file)
                        voices_path = os.path.join(app.static_folder, 'voices', voice_file)
                        
                        if os.path.exists(upload_path):
                            full_character_voices[char] = upload_path
                        elif os.path.exists(voices_path):
                            full_character_voices[char] = voices_path
                        else:
                            logger.warning(f"Voice file not found for {char}: {voice_file}")
                            full_character_voices[char] = audio_prompt_path  # Fallback to default
                
                # Use multi-voice generation
                # Ensure we have a default voice for unmatched characters
                default_voice_for_multivoice = audio_prompt_path
                if not default_voice_for_multivoice:
                    # If no default voice, try to use the first available voice from static/voices
                    available_voices = get_available_voice_files()
                    if available_voices:
                        first_voice = list(available_voices.values())[0]
                        default_voice_for_multivoice = os.path.join(app.static_folder, 'voices', first_voice)
                        status_callback(f"🔄 Using {first_voice} as default voice for unmatched characters")
                
                result = generate_multivoice_audio(
                    text_input=text_input,
                    character_voices=full_character_voices,
                    default_voice=default_voice_for_multivoice,
                    source_filename=source_filename,
                    session_id=session_id,
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
                    noise_strength=noise_strength,
                    strict_voice_consistency=True,
                    voice_similarity_threshold=voice_similarity_threshold
                )
            else:
                # Character markers found but no voice assignments - fall back to single voice
                status_callback("⚠️ Character markers found but no voice assignments, using single voice")
                result = generate_voice(
                    text_input=text_input,
                    audio_prompt_path=audio_prompt_path,
                    source_filename=source_filename,
                    session_id=session_id,
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
                    noise_strength=noise_strength,
                    strict_voice_consistency=True,
                    voice_similarity_threshold=voice_similarity_threshold
                )
        else:
            # Single voice generation
            status_callback("🎙️ Using single voice generation (no character markers found)")
            result = generate_voice(
                text_input=text_input,
                audio_prompt_path=audio_prompt_path,
                source_filename=source_filename,
                session_id=session_id,
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
                noise_strength=noise_strength,
                voice_similarity_threshold=voice_similarity_threshold
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
                # Determine display name for audioPromptPath
                display_audio_prompt = audio_prompt_filename
                if voice_profile_settings:
                    display_audio_prompt = voice_profile_settings.get('display_name', voice_profile)
                elif character_markers and len(character_markers) > 0:
                    # Multivoice generation detected
                    display_audio_prompt = 'Multivoice'
                elif not audio_prompt_filename:
                    display_audio_prompt = 'Default Voice'
                
                write_to_json(
                    processed_text_result, filename, display_audio_prompt, exaggeration, temperature, 
                    cfg_weight, chunk_size, speed, pitch, reduce_noise, remove_silence, actual_seed, duration,
                    # Additional parameters for complete record
                    export_formats=export_formats, num_candidates=num_candidates, use_whisper_validation=use_whisper_validation,
                    whisper_model_name=whisper_model_name, use_faster_whisper=use_faster_whisper, use_longest_transcript=use_longest_transcript,
                    validation_threshold=validation_threshold, enable_parallel=enable_parallel, num_workers=parallel_workers,
                    to_lowercase=to_lowercase, normalize_spacing=normalize_spacing, fix_dot_letters=fix_dot_letters, remove_reference_numbers=remove_reference_numbers,
                    sound_words=sound_words, normalize_audio=normalize_audio, normalize_method=normalize_method,
                    noise_reduction_method=noise_reduction_method, noise_strength=noise_strength,
                    # Additional missing parameters
                    max_attempts=max_attempts, normalize_level=normalize_level, normalize_tp=normalize_tp, normalize_lra=normalize_lra,
                    # Voice profile information
                    voice_profile=voice_profile, voice_profile_settings=voice_profile_settings
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
        raw_text_input = data.get('text_input', '')
        text_input = normalize_text_content(raw_text_input)
        if raw_text_input != text_input:
            logger.info("Applied text normalization to direct input")
            logger.debug(f"Original length: {len(raw_text_input)}, normalized length: {len(text_input)}")
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
                        raw_content = f.read()
                        content = normalize_text_content(raw_content)
                        if raw_content != content:
                            logger.info(f"Applied text normalization to file: {os.path.basename(file_path)}")
                            logger.debug(f"File {os.path.basename(file_path)} - Original length: {len(raw_content)}, normalized length: {len(content)}")
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
        # Update data object with normalized text for consistent processing
        data_copy = data.copy()
        data_copy['text_input'] = text_input
        result = process_single_text_generation(data_copy, session_id)
        
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
        # Clean up session from active sessions and cancellation flags
        if session_id in active_sessions:
            del active_sessions[session_id]
        cancellation_state.clear_cancellation_flag(session_id)

@socketio.on('cancel_generation')
def handle_cancel_generation():
    """Handle generation cancellation request"""
    try:
        from flask import request as flask_request
        session_id = flask_request.sid
        
        logger.info(f"Cancellation requested for session: {session_id}")
        
        # Set cancellation flag to interrupt generation process
        global active_sessions
        cancellation_state.set_cancellation_flag(session_id)
        
        # Remove session from active sessions to stop status updates
        if session_id in active_sessions:
            del active_sessions[session_id]
            logger.info(f"Removed session {session_id} from active sessions")
        
        # Send cancellation confirmation
        socketio.emit('generation_cancelled', {'message': 'Generation cancelled successfully'}, room=session_id)
        socketio.sleep(0)
        
    except Exception as e:
        logger.error(f"Error handling cancellation: {str(e)}")
        socketio.emit('error', {'error': f'Cancellation failed: {str(e)}'}, room=session_id)
        socketio.sleep(0)

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
            # Check if generation was cancelled
            if cancellation_state.is_generation_cancelled(session_id):
                logger.info(f"Status update skipped - generation cancelled for session {session_id}")
                return
                
            logger.info(f"Status: {message}")
            socketio.emit('generation_status', {'message': message}, room=session_id)
            # Force immediate delivery of the message
            socketio.sleep(0)
        except Exception as e:
            logger.error(f"Error sending status update: {str(e)}")
    return update_status

def is_generation_cancelled(session_id):
    """Check if generation has been cancelled for a session"""
    return cancellation_state.is_generation_cancelled(session_id)

# Legacy functions removed - replaced by create_progress_callback and create_status_callback

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
            # Voice profile information
            'voiceProfile': kwargs.get('voice_profile', ''),
            'voiceProfileSettings': kwargs.get('voice_profile_settings', {}),
            # Additional parameters for complete generation record
            'maxAttempts': kwargs.get('max_attempts', 3),
            'normalizeLevel': kwargs.get('normalize_level', -24),
            'normalizeTp': kwargs.get('normalize_tp', -2),
            'normalizeLra': kwargs.get('normalize_lra', 7),
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

def normalize_text_content(text: str) -> str:
    """
    Normalize text content for consistent processing across all input sources.
    This ensures that text from files and direct input are processed identically.
    """
    if not text:
        return ""
    
    # Remove BOM if present
    if text.startswith('\ufeff'):
        text = text[1:]
    
    # Convert various line endings to standard newlines
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove excessive whitespace while preserving intentional line breaks
    # Replace multiple newlines with double newlines (paragraph breaks)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Replace tabs with spaces
    text = text.replace('\t', ' ')
    
    # Remove excessive spaces (but preserve single spaces and line breaks)
    text = re.sub(r'[ ]+', ' ', text)  # Multiple spaces -> single space
    
    # Remove leading/trailing whitespace from each line
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)
    
    # Remove leading/trailing whitespace from entire text
    text = text.strip()
    
    return text

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