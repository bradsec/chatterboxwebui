/**
 * Form validation utilities for Chatterbox Web UI
 */

// Get max text length from the server (passed via template)
const getMaxTextLength = () => {
  // Try to get from data attribute set by server
  const bodyMaxLength = document.body.dataset.maxTextLength;
  if (bodyMaxLength) {
    return parseInt(bodyMaxLength);
  }
  
  // Try to get from meta tag
  const metaTag = document.querySelector('meta[name="max-text-length"]');
  if (metaTag) {
    return parseInt(metaTag.content);
  }
  
  // Fallback to default
  return 10000;
};

const MAX_TEXT_LENGTH = getMaxTextLength();

function validateText(text) {
  if (!text || text.trim() === '') {
    return { valid: false, error: 'Text is empty.' };
  }
  if (text.length > MAX_TEXT_LENGTH) {
    return { valid: false, error: `Text is too long. Please limit to ${MAX_TEXT_LENGTH.toLocaleString()} characters.` };
  }
  return { valid: true };
}

function validateFile(file) {
  if (!file) return { valid: true }; // File is optional
  
  const allowedTypes = ['audio/wav', 'audio/mp3', 'audio/flac', 'audio/opus', 'audio/m4a', 'audio/ogg'];
  const allowedExtensions = ['.wav', '.mp3', '.flac', '.opus', '.m4a', '.ogg'];
  
  const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
  
  if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
    return { valid: false, error: 'Invalid file type. Allowed: WAV, MP3, FLAC, OPUS, M4A, OGG' };
  }
  
  if (file.size > 50 * 1024 * 1024) { // 50MB
    return { valid: false, error: 'File too large. Maximum size is 50MB.' };
  }
  
  return { valid: true };
}

function updateCharacterCount() {
  const textInput = document.getElementById('text-input');
  const text = textInput.value;
  const charCount = text.length;
  
  // Create or update character counter
  let counter = document.getElementById('char-counter');
  if (!counter) {
    counter = document.createElement('div');
    counter.id = 'char-counter';
    counter.style.fontSize = '12px';
    counter.style.opacity = '0.7';
    counter.style.textAlign = 'right';
    counter.style.marginTop = '5px';
    textInput.parentNode.insertBefore(counter, textInput.nextSibling);
  }
  
  counter.textContent = `${charCount.toLocaleString()}/${MAX_TEXT_LENGTH.toLocaleString()} characters`;
  counter.style.color = charCount > MAX_TEXT_LENGTH ? '#ff4444' : 'var(--font-color)';
}

function validateFormData(data) {
  const errors = [];
  
  // Validate text
  const textValidation = validateText(data.textInput);
  if (!textValidation.valid) {
    errors.push(textValidation.error);
  }
  
  // Validate core TTS parameters
  if (data.exaggeration < 0.25 || data.exaggeration > 2.0) {
    errors.push('Exaggeration must be between 0.25 and 2.0');
  }
  if (data.temperature < 0.05 || data.temperature > 5.0) {
    errors.push('Temperature must be between 0.05 and 5.0');
  }
  if (data.cfgWeight < 0.0 || data.cfgWeight > 1.0) {
    errors.push('CFG Weight must be between 0.0 and 1.0');
  }
  if (data.chunkSize < 50 || data.chunkSize > 300) {
    errors.push('Chunk size must be between 50 and 300');
  }
  
  // Validate speed and pitch
  if (data.speed < 0.1 || data.speed > 2.0) {
    errors.push('Speed must be between 0.1 and 2.0');
  }
  if (data.pitch < -12 || data.pitch > 12) {
    errors.push('Pitch must be between -12 and 12 semitones');
  }
  
  // Validate seed
  if (data.seed < 0 || data.seed > 999999) {
    errors.push('Seed must be between 0 and 999999');
  }
  
  // Validate generation parameters
  if (data.numCandidates < 1 || data.numCandidates > 10) {
    errors.push('Number of candidates must be between 1 and 10');
  }
  if (data.maxAttempts < 1 || data.maxAttempts > 10) {
    errors.push('Max attempts must be between 1 and 10');
  }
  if (data.parallelWorkers < 1 || data.parallelWorkers > 8) {
    errors.push('Parallel workers must be between 1 and 8');
  }
  if (data.validationThreshold < 0.5 || data.validationThreshold > 1.0) {
    errors.push('Validation threshold must be between 0.5 and 1.0');
  }
  
  
  // Validate normalization parameters
  if (data.normalizeLevel < -70 || data.normalizeLevel > -5) {
    errors.push('Normalize level must be between -70 and -5 dB');
  }
  if (data.normalizeTp < -9 || data.normalizeTp > 0) {
    errors.push('True peak must be between -9 and 0 dB');
  }
  if (data.normalizeLra < 1 || data.normalizeLra > 50) {
    errors.push('Loudness range must be between 1 and 50');
  }
  
  // Validate export format
  const validFormats = ['wav', 'mp3', 'flac'];
  if (data.exportFormats && data.exportFormats.length > 0) {
    for (const format of data.exportFormats) {
      if (!validFormats.includes(format.toLowerCase())) {
        errors.push(`Invalid export format: ${format}. Valid formats: ${validFormats.join(', ')}`);
      }
    }
  }
  
  // Validate whisper model
  const validWhisperModels = ['tiny', 'base', 'small', 'medium', 'large'];
  if (data.whisperModel && !validWhisperModels.includes(data.whisperModel)) {
    errors.push(`Invalid Whisper model: ${data.whisperModel}. Valid models: ${validWhisperModels.join(', ')}`);
  }
  
  // Check for NaN values in all numeric fields
  const numericFields = [
    'exaggeration', 'temperature', 'cfgWeight', 'chunkSize', 'speed', 'pitch', 'seed',
    'numCandidates', 'maxAttempts', 'parallelWorkers', 'validationThreshold',
    'normalizeLevel', 'normalizeTp', 'normalizeLra'
  ];
  for (const field of numericFields) {
    if (data[field] !== undefined && isNaN(data[field])) {
      errors.push(`Invalid numeric value for ${field}`);
    }
  }
  
  return {
    valid: errors.length === 0,
    errors: errors
  };
}

function getFormData() {
  const textInput = document.getElementById('text-input');
  const fileInput = document.getElementById('audio-prompt');
  
  // Get form values with safe parsing and validation (voice consistency optimized defaults)
  const exaggeration = safeParseFloat(safeGetElementValue('exaggeration'), 0.5);
  const temperature = safeParseFloat(safeGetElementValue('temperature'), 0.3);
  const cfgWeight = safeParseFloat(safeGetElementValue('cfg-weight'), 0.7);
  const chunkSize = safeParseInt(safeGetElementValue('chunk-size'), 130);
  const reduceNoise = safeGetElementChecked('reduce-noise', false);
  const removeSilence = safeGetElementChecked('remove-silence', false);
  const speed = safeParseFloat(safeGetElementValue('speed'), 1.0);
  const pitch = safeParseInt(safeGetElementValue('pitch'), 0);
  
  // Handle seed value with safe parsing
  let seed = 0;
  const seedValue = safeGetElementValue('seed', '0');
  if (seedValue === 'custom') {
    const customSeed = safeGetElementValue('custom-seed-input');
    if (customSeed) {
      seed = safeParseInt(customSeed, 0);
      if (seed < 0 || seed > 999999) {
        throw new Error('Custom seed must be a number between 0 and 999999');
      }
    }
  } else {
    seed = safeParseInt(seedValue, 0);
  }
  
  // Get advanced control values with safe parsing
  const textFilesInput = document.getElementById('text-files');
  // Get selected radio button value for export format
  const exportFormatRadio = document.querySelector('input[name="export-format"]:checked');
  const exportFormat = exportFormatRadio ? exportFormatRadio.value : 'wav';
  const exportFormats = [exportFormat];
  
  const numCandidates = safeParseInt(safeGetElementValue('num-candidates'), 3);
  const maxAttempts = safeParseInt(safeGetElementValue('max-attempts'), 3);
  const whisperModel = safeGetElementValue('whisper-model', 'medium');
  const useFasterWhisper = safeGetElementChecked('use-faster-whisper', true);
  const bypassWhisper = safeGetElementChecked('bypass-whisper', false);
  const useLongestTranscript = safeGetElementChecked('use-longest-transcript', true);
  const enableParallel = safeGetElementChecked('enable-parallel', true);
  const parallelWorkers = safeParseInt(safeGetElementValue('parallel-workers'), 4);
  const validationThreshold = safeParseFloat(safeGetElementValue('validation-threshold'), 0.85);
  
  const toLowercase = safeGetElementChecked('to-lowercase', true);
  const normalizeSpacing = safeGetElementChecked('normalize-spacing', true);
  const fixDotLetters = safeGetElementChecked('fix-dot-letters', true);
  const removeReferenceNumbers = safeGetElementChecked('remove-reference-numbers', true);
  
  const keepOriginal = safeGetElementChecked('keep-original', false);
  
  const normalizeAudio = safeGetElementChecked('normalize-audio', false);
  const normalizeMethod = safeGetElementValue('normalize-method', 'ebu');
  const normalizeLevel = safeParseInt(safeGetElementValue('normalize-level'), -24);
  const normalizeTp = safeParseInt(safeGetElementValue('normalize-tp'), -2);
  const normalizeLra = safeParseInt(safeGetElementValue('normalize-lra'), 7);
  
  const soundWords = safeGetElementValue('sound-words', '');
  
  return {
    textInput: textInput.value,
    audioPromptFile: fileInput.files[0],
    exaggeration,
    temperature,
    cfgWeight,
    chunkSize,
    reduceNoise,
    removeSilence,
    speed,
    pitch,
    seed,
    seedValue,
    exportFormats,
    numCandidates,
    maxAttempts,
    whisperModel,
    useFasterWhisper,
    bypassWhisper,
    useLongestTranscript,
    enableParallel,
    parallelWorkers,
    validationThreshold,
    toLowercase,
    normalizeSpacing,
    fixDotLetters,
    removeReferenceNumbers,
    useAutoEditor,
    keepOriginal,
    normalizeAudio,
    normalizeMethod,
    normalizeLevel,
    normalizeTp,
    normalizeLra,
    soundWords
  };
}

// Safe parsing functions with error handling
function safeParseFloat(value, defaultValue = 0) {
  if (value === null || value === undefined || value === '') {
    return defaultValue;
  }
  const parsed = parseFloat(value);
  return isNaN(parsed) ? defaultValue : parsed;
}

function safeParseInt(value, defaultValue = 0) {
  if (value === null || value === undefined || value === '') {
    return defaultValue;
  }
  const parsed = parseInt(value);
  return isNaN(parsed) ? defaultValue : parsed;
}

function safeParseBoolean(value, defaultValue = false) {
  if (value === null || value === undefined) {
    return defaultValue;
  }
  if (typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'string') {
    return value.toLowerCase() === 'true' || value === '1';
  }
  return Boolean(value);
}

function safeGetElementValue(elementId, defaultValue = '') {
  try {
    const element = document.getElementById(elementId);
    if (!element) {
      return defaultValue;
    }
    return element.value || defaultValue;
  } catch (error) {
    return defaultValue;
  }
}

function safeGetElementChecked(elementId, defaultValue = false) {
  try {
    const element = document.getElementById(elementId);
    if (!element) {
      return defaultValue;
    }
    return element.checked || defaultValue;
  } catch (error) {
    return defaultValue;
  }
}

// Export functions for use in other modules
window.FormValidation = {
  MAX_TEXT_LENGTH,
  validateText,
  validateFile,
  updateCharacterCount,
  validateFormData,
  getFormData,
  safeParseFloat,
  safeParseInt,
  safeParseBoolean,
  safeGetElementValue,
  safeGetElementChecked
};