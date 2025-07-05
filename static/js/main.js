document.addEventListener('DOMContentLoaded', function() {
  const socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port);

  // DOM element references
  const generateButton = document.getElementById('generate-button');
  const cancelButton = document.getElementById('cancel-button');
  const resetButton = document.getElementById('reset-button');
  const deleteAllButton = document.getElementById('delete-all-button');
  const audioList = document.getElementById('audio-list');
  const progressContainer = document.querySelector('.progress-container');
  const progressComplete = document.querySelector('.progress-complete');
  const fileInput = document.getElementById('audio-prompt');
  const fileNameDisplay = document.getElementById('file-name-display');
  const clearAudioButton = document.getElementById('clear-audio-button');
  
  // File upload label event handler
  const fileUploadLabel = document.querySelector('label[for="audio-prompt"]');
  if (fileUploadLabel && fileInput) {
    fileUploadLabel.addEventListener('click', function(event) {
      if (fileInput.disabled) {
        event.preventDefault();
      }
    });
  }
  const seedSelect = document.getElementById('seed');
  const customSeedContainer = document.getElementById('custom-seed-container');
  const customSeedInput = document.getElementById('custom-seed-input');
  const textInput = document.getElementById('text-input');
  // Note: preset-voice-select removed in favor of Voice Library system

  // Track current uploaded file
  let currentUploadedFile = null;
  
  // Voice Manager for integrating Voice Library with TTS generation
  window.VoiceManager = {
    selectedVoice: null,
    selectedVoiceType: null,
    selectedDisplayName: null,
    
    setSelectedVoice: function(voiceValue, displayName, voiceType) {
      this.selectedVoice = voiceValue;
      this.selectedVoiceType = voiceType;
      this.selectedDisplayName = displayName;
      
      // Clear any uploaded file since we're using a voice library selection
      if (voiceType === 'legacy' || voiceType === 'profile') {
        clearReferenceAudio();
      }
      
      console.log(`VoiceManager: Set voice to ${displayName} (${voiceType}: ${voiceValue})`);
    },
    
    clearSelectedVoice: function() {
      this.selectedVoice = null;
      this.selectedVoiceType = null;
      this.selectedDisplayName = null;
      console.log('VoiceManager: Cleared voice selection');
    },
    
    getCurrentVoice: function() {
      return {
        voice: this.selectedVoice,
        type: this.selectedVoiceType,
        displayName: this.selectedDisplayName
      };
    },
    
    hasSelectedVoice: function() {
      return this.selectedVoice !== null;
    }
  };

  // Use centralized message functions from DOMUtils
  function showFlashMessage(message, type = 'success') {
    DOMUtils.showFlashMessage(message, type);
  }

  function showError(message) {
    DOMUtils.showError(message);
  }


  // Handle clear audio button
  clearAudioButton.addEventListener('click', DOMUtils.debounce(clearReferenceAudio, 300));

  // Handle text file upload
  const textFilesInput = document.getElementById('text-files');
  const textFilesDisplay = document.getElementById('text-files-display');
  const clearTextFilesButton = document.getElementById('clear-text-files-button');

  if (textFilesInput && textFilesDisplay && clearTextFilesButton) {
    textFilesInput.addEventListener('change', function(event) {
      const files = event.target.files;
      if (files && files.length > 0) {
        const fileNames = Array.from(files).map(file => file.name);
        textFilesDisplay.textContent = `${files.length} file(s): ${fileNames.join(', ')}`;
        clearTextFilesButton.classList.remove('hide');
        
        // Disable text input when files are selected
        textInput.disabled = true;
        textInput.style.opacity = '0.5';
        textInput.style.cursor = 'not-allowed';
        textInput.placeholder = 'Text input disabled - using uploaded text files';
      } else {
        textFilesDisplay.textContent = 'No files selected';
        clearTextFilesButton.classList.add('hide');
        
        // Re-enable text input when no files are selected
        textInput.disabled = false;
        textInput.style.opacity = '1';
        textInput.style.cursor = 'text';
        textInput.placeholder = 'Enter your text here (max ' + MAX_TEXT_LENGTH.toLocaleString() + ' characters)...';
      }
    });

    clearTextFilesButton.addEventListener('click', function() {
      textFilesInput.value = '';
      textFilesDisplay.textContent = 'No files selected';
      clearTextFilesButton.classList.add('hide');
      
      // Re-enable text input when files are cleared
      textInput.disabled = false;
      textInput.style.opacity = '1';
      textInput.style.cursor = 'text';
      textInput.placeholder = 'Enter your text here (max ' + MAX_TEXT_LENGTH.toLocaleString() + ' characters)...';
      
      showFlashMessage('Text files cleared', 'info');
    });
  }

  function clearReferenceAudio() {
    const wasFileSelected = fileInput.files[0] || currentUploadedFile;
    
    fileInput.value = '';
    fileNameDisplay.textContent = 'No file selected';
    fileNameDisplay.style.color = 'var(--font-color)';
    clearAudioButton.classList.add('hide');
    
    // If we have an uploaded file on the server, delete it
    if (currentUploadedFile) {
      fetch('/clear_reference_audio', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ filename: currentUploadedFile })
      })
      .then(response => response.json())
      .then(data => {
        if (data.success) {
            if (wasFileSelected) {
            showFlashMessage('Reference audio cleared', 'info');
          }
        }
      })
      .catch(error => {
        showError('Error clearing reference audio');
      });
    } else if (wasFileSelected) {
      showFlashMessage('Reference audio cleared', 'info');
    }
    
    currentUploadedFile = null;
    
    // Also clear preset voice selection and re-enable both options
    // Preset voice functionality removed - using Voice Library system
    fileInput.disabled = false;
    fileInput.style.opacity = '1';
  }

  // Handle seed selection
  seedSelect.addEventListener('change', function(event) {
    if (event.target.value === 'custom') {
      customSeedContainer.classList.remove('hide');
      customSeedInput.focus();
    } else {
      customSeedContainer.classList.add('hide');
    }
  });

  // Voice selection handling
  // Load preset voices on page load
  loadPresetVoices();

  // Preset voice functionality removed - using Voice Library system

  // Handle file upload - disable preset when file is selected
  fileInput.addEventListener('change', function(event) {
    console.log('File input change event triggered', event);
    const file = event.target.files[0];
    console.log('Selected file:', file);
    if (file) {
      // Voice Library system handles voice selection
      
      // Continue with existing file validation logic
      const validation = FormValidation.validateFile(file);
      if (!validation.valid) {
        showError(validation.error);
        fileInput.value = '';
        fileNameDisplay.textContent = 'No file selected';
        clearAudioButton.classList.add('hide');
        // Voice Library system handles voice selection
        return;
      }
      
      fileNameDisplay.textContent = file.name;
      fileNameDisplay.style.color = 'var(--font-color)';
      clearAudioButton.classList.remove('hide');
      currentUploadedFile = null; // Reset since user selected a new file
    } else {
      // Voice Library system handles voice selection
      fileNameDisplay.textContent = 'No file selected';
      clearAudioButton.classList.add('hide');
    }
  });

  // Character counter for text input
  textInput.addEventListener('input', DOMUtils.debounce(FormValidation.updateCharacterCount, 100));

  // Slider handling is now done by DOMUtils.initializeSliders()

  // Reset button functionality
  resetButton.addEventListener('click', function() {
    if (confirm('Reset all settings to defaults? This will clear all form values including the reference audio file.')) {
      resetToDefaults();
    }
  });

  // Cancel button functionality
  if (cancelButton) {
    cancelButton.addEventListener('click', function() {
      if (confirm('Are you sure you want to cancel the current generation?')) {
        socket.emit('cancel_generation');
        DOMUtils.addStatusMessage('🛑 Cancellation requested...');
      }
    });
  }

  // Delete all button functionality - improved
  if (deleteAllButton) {
    deleteAllButton.addEventListener('click', function() {
      const audioCards = document.querySelectorAll('.audio-item');
      if (audioCards.length === 0) {
        showFlashMessage('No audio files to delete', 'info');
        return;
      }
      
      if (confirm(`Are you sure you want to delete all ${audioCards.length} audio files? This action cannot be undone.`)) {
        deleteAllAudioFiles();
      }
    });
  }

  function resetToDefaults() {
    // Clear reference audio first
    clearReferenceAudio();
    
    // Reset text input to sample text
    textInput.value = sampleText;
    FormValidation.updateCharacterCount();
    
    // Reset sliders to defaults (optimized for voice consistency)
    document.getElementById('exaggeration').value = '0.50';
    document.getElementById('temperature').value = '0.80';
    document.getElementById('cfg-weight').value = '0.50';
    document.getElementById('chunk-size').value = '300';
    document.getElementById('speed').value = '1.00';
    document.getElementById('pitch').value = '0';
    
    // Reset new sliders
    if (document.getElementById('num-candidates')) document.getElementById('num-candidates').value = '3';
    if (document.getElementById('max-attempts')) document.getElementById('max-attempts').value = '3';
    if (document.getElementById('parallel-workers')) document.getElementById('parallel-workers').value = '3';
    if (document.getElementById('validation-threshold')) document.getElementById('validation-threshold').value = '0.85';
    if (document.getElementById('ae-threshold')) document.getElementById('ae-threshold').value = '0.06';
    if (document.getElementById('ae-margin')) document.getElementById('ae-margin').value = '0.2';
    if (document.getElementById('normalize-level')) document.getElementById('normalize-level').value = '-24';
    if (document.getElementById('normalize-tp')) document.getElementById('normalize-tp').value = '-2';
    if (document.getElementById('normalize-lra')) document.getElementById('normalize-lra').value = '7';
    if (document.getElementById('noise-reduction-method')) document.getElementById('noise-reduction-method').value = 'afftdn';
    if (document.getElementById('noise-strength')) document.getElementById('noise-strength').value = '0.85';
    if (document.getElementById('voice-similarity-threshold')) document.getElementById('voice-similarity-threshold').value = '0.5';
    
    // Update slider displays
    document.getElementById('exaggeration-value').textContent = '0.50';
    document.getElementById('temperature-value').textContent = '0.80';
    document.getElementById('speed-value').textContent = '1.00';
    document.getElementById('pitch-value').textContent = '0';
    document.getElementById('cfg-weight-value').textContent = '0.50';
    document.getElementById('chunk-size-value').textContent = '300';
    
    // Update new slider displays
    if (document.getElementById('num-candidates-value')) document.getElementById('num-candidates-value').textContent = '3';
    if (document.getElementById('max-attempts-value')) document.getElementById('max-attempts-value').textContent = '3';
    if (document.getElementById('parallel-workers-value')) document.getElementById('parallel-workers-value').textContent = '3';
    if (document.getElementById('validation-threshold-value')) document.getElementById('validation-threshold-value').textContent = '0.85';
    if (document.getElementById('normalize-level-value')) document.getElementById('normalize-level-value').textContent = '-24';
    if (document.getElementById('normalize-tp-value')) document.getElementById('normalize-tp-value').textContent = '-2';
    if (document.getElementById('normalize-lra-value')) document.getElementById('normalize-lra-value').textContent = '7';
    if (document.getElementById('noise-strength-value')) document.getElementById('noise-strength-value').textContent = '0.85';
    if (document.getElementById('voice-similarity-threshold-value')) document.getElementById('voice-similarity-threshold-value').textContent = '0.5';
    
    // Reset dropdowns to defaults
    document.getElementById('seed').value = '0';
    
    // Reset new dropdowns
    if (document.getElementById('whisper-model')) document.getElementById('whisper-model').value = 'medium';
    if (document.getElementById('normalize-method')) document.getElementById('normalize-method').value = 'ebu';
    // Reset export format radio buttons
    const exportFormatRadios = document.querySelectorAll('input[name="export-format"]');
    exportFormatRadios.forEach(radio => {
      radio.checked = radio.value === 'wav';
    });
    
    // Hide custom seed input if visible
    customSeedContainer.classList.add('hide');
    customSeedInput.value = '';
    
    // Reset checkboxes
    document.getElementById('reduce-noise').checked = false;
    document.getElementById('remove-silence').checked = false;
    
    // Reset new checkboxes to their defaults (matching server-side defaults)
    if (document.getElementById('use-faster-whisper')) document.getElementById('use-faster-whisper').checked = true;
    if (document.getElementById('bypass-whisper')) document.getElementById('bypass-whisper').checked = true;
    if (document.getElementById('use-longest-transcript')) document.getElementById('use-longest-transcript').checked = true;
    if (document.getElementById('enable-parallel')) document.getElementById('enable-parallel').checked = true;
    if (document.getElementById('to-lowercase')) document.getElementById('to-lowercase').checked = false;
    if (document.getElementById('normalize-spacing')) document.getElementById('normalize-spacing').checked = false;
    if (document.getElementById('fix-dot-letters')) document.getElementById('fix-dot-letters').checked = false;
    if (document.getElementById('remove-reference-numbers')) document.getElementById('remove-reference-numbers').checked = false;
    if (document.getElementById('normalize-audio')) document.getElementById('normalize-audio').checked = false;
    
    // Clear text files
    if (textFilesInput) {
      textFilesInput.value = '';
      textFilesDisplay.textContent = 'No files selected';
      clearTextFilesButton.classList.add('hide');
      
      // Re-enable text input when files are cleared
      textInput.disabled = false;
      textInput.style.opacity = '1';
      textInput.style.cursor = 'text';
      textInput.placeholder = 'Enter your text here (max ' + MAX_TEXT_LENGTH.toLocaleString() + ' characters)...';
    }
    
    // Clear sound words
    if (document.getElementById('sound-words')) document.getElementById('sound-words').value = '';
    
    // Clear localStorage
    clearStoredSettings();
    
    showFlashMessage('All settings reset to defaults', 'success');
  }

  function clearStoredSettings() {
    const settingsKeys = [
      'textInput', 'exaggeration', 'temperature', 'cfgWeight', 'chunkSize',
      'removeSilence', 'speed', 'pitch', 'seed', 'customSeed'
    ];
    
    settingsKeys.forEach(key => localStorage.removeItem(key));
  }

  // Delete all audio files function - improved with better server communication
  function deleteAllAudioFiles() {
    const audioCards = document.querySelectorAll('.audio-item');
    const totalFiles = audioCards.length;

    if (totalFiles === 0) {
      showFlashMessage('No audio files to delete', 'info');
      return;
    }

    // Disable the delete all button to prevent multiple clicks
    if (deleteAllButton) {
      deleteAllButton.disabled = true;
      deleteAllButton.textContent = 'Deleting...';
    }

    // Show progress in flash message
    showFlashMessage(`🗑️ Deleting ${totalFiles} files...`, 'info');

    // Clear progress complete message if it was showing
    if (progressComplete.classList.contains('hide') === false) {
      progressComplete.textContent = "";
    }

    // Fade out all cards immediately for visual feedback
    audioCards.forEach((card) => {
      card.style.opacity = '0.3';
      card.style.transition = 'opacity 0.3s ease';
    });

    // Use the new server endpoint for bulk deletion
    fetch('/delete_all_audio', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      }
    })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      if (data.success) {
        completeDeleteAll(data.deleted_count, data.failed_count, totalFiles);
      } else {
        throw new Error(data.error || 'Delete all operation failed');
      }
    })
    .catch(error => {
        showFlashMessage(`❌ Failed to delete files: ${error.message}`, 'error');
      
      // Restore card opacity on error
      audioCards.forEach((card) => {
        card.style.opacity = '1';
      });
      
      // Re-enable delete all button
      if (deleteAllButton) {
        deleteAllButton.disabled = false;
        deleteAllButton.textContent = 'Delete All';
      }
    });
  }

  function completeDeleteAll(deletedCount, failedCount, totalFiles) {
    // Re-enable delete all button
    if (deleteAllButton) {
      deleteAllButton.disabled = false;
      deleteAllButton.textContent = 'Delete All';
    }

    // Show result message
    if (failedCount === 0) {
      showFlashMessage(`✅ Successfully deleted all ${deletedCount} audio files`, 'success');
    } else if (deletedCount === 0) {
      showFlashMessage(`❌ Failed to delete any files (${failedCount} errors)`, 'error');
    } else {
      showFlashMessage(`⚠️ Deleted ${deletedCount} files, ${failedCount} failed`, 'warning');
    }

    // Reload the audio list to reflect changes
    setTimeout(() => {
      loadAudioList();
    }, 1000);
  }

  // Form submission with enhanced validation
  document.getElementById('generator-form').addEventListener('submit', function(event) {
    event.preventDefault();

    if (isGenerating) {
      return;
    }

    const textInputValue = textInput.value;
    const audioPromptFile = fileInput.files[0];

    // Validate inputs
    const textValidation = FormValidation.validateText(textInputValue);
    if (!textValidation.valid) {
      showError(textValidation.error);
      textInput.focus();
      return;
    }

    const fileValidation = FormValidation.validateFile(audioPromptFile);
    if (!fileValidation.valid) {
      showError(fileValidation.error);
      return;
    }

    // Get form values with validation and null checking (voice consistency optimized defaults)
    const exaggeration = parseFloat(document.getElementById('exaggeration')?.value || '0.5');
    const temperature = parseFloat(document.getElementById('temperature')?.value || '0.8');
    const cfgWeight = parseFloat(document.getElementById('cfg-weight')?.value || '0.5');
    const chunkSize = parseInt(document.getElementById('chunk-size')?.value || '300');
    const reduceNoise = document.getElementById('reduce-noise')?.checked || false;
    const removeSilence = document.getElementById('remove-silence')?.checked || false;
    const speed = parseFloat(document.getElementById('speed')?.value || '1.0');
    const pitch = parseInt(document.getElementById('pitch')?.value || '0');
    
    // Get new advanced control values
    const textFiles = textFilesInput ? textFilesInput.files : null;
    // Get selected radio button value for export format
    const exportFormatRadio = document.querySelector('input[name="export-format"]:checked');
    const exportFormat = exportFormatRadio ? exportFormatRadio.value : 'wav';
    const exportFormats = [exportFormat];
    const numCandidates = document.getElementById('num-candidates') ? parseInt(document.getElementById('num-candidates').value) : 3;
    const maxAttempts = document.getElementById('max-attempts') ? parseInt(document.getElementById('max-attempts').value) : 3;
    const whisperModel = document.getElementById('whisper-model') ? document.getElementById('whisper-model').value : 'medium';
    const useFasterWhisper = document.getElementById('use-faster-whisper') ? document.getElementById('use-faster-whisper').checked : true;
    const bypassWhisper = document.getElementById('bypass-whisper') ? document.getElementById('bypass-whisper').checked : true;
    const useLongestTranscript = document.getElementById('use-longest-transcript') ? document.getElementById('use-longest-transcript').checked : true;
    const enableParallel = document.getElementById('enable-parallel') ? document.getElementById('enable-parallel').checked : false;
    const parallelWorkers = document.getElementById('parallel-workers') ? parseInt(document.getElementById('parallel-workers').value) : 4;
    const validationThreshold = parseFloat(document.getElementById('validation-threshold')?.value || '0.85');
    
    const toLowercase = document.getElementById('to-lowercase') ? document.getElementById('to-lowercase').checked : true;
    const normalizeSpacing = document.getElementById('normalize-spacing') ? document.getElementById('normalize-spacing').checked : true;
    const fixDotLetters = document.getElementById('fix-dot-letters') ? document.getElementById('fix-dot-letters').checked : true;
    const removeReferenceNumbers = document.getElementById('remove-reference-numbers') ? document.getElementById('remove-reference-numbers').checked : true;
    
    const normalizeAudio = document.getElementById('normalize-audio')?.checked || false;
    const normalizeMethod = document.getElementById('normalize-method')?.value || 'ebu';
    const normalizeLevel = parseInt(document.getElementById('normalize-level')?.value || '-24');
    const normalizeTp = parseInt(document.getElementById('normalize-tp')?.value || '-2');
    const normalizeLra = parseInt(document.getElementById('normalize-lra')?.value || '7');
    
    const noiseReductionMethod = 'anlmdn'; // Speech-optimized method
    const noiseStrength = parseFloat(document.getElementById('noise-strength')?.value || '0.85');
    
    const soundWords = document.getElementById('sound-words')?.value || '';
    
    // Voice similarity validation
    const voiceSimilarityThreshold = parseFloat(document.getElementById('voice-similarity-threshold')?.value || '0.5');

    // Check for NaN values
    if (isNaN(exaggeration) || isNaN(temperature) || isNaN(cfgWeight) || isNaN(chunkSize) || 
        isNaN(speed) || isNaN(pitch) || 
        isNaN(normalizeLevel) || isNaN(normalizeTp) || isNaN(normalizeLra) || isNaN(noiseStrength) ||
        isNaN(voiceSimilarityThreshold)) {
      showError('Invalid parameter values detected. Please refresh the page and try again.');
      return;
    }

    // Validate ranges
    if (exaggeration < 0.25 || exaggeration > 2.0) {
      showError('Exaggeration must be between 0.25 and 2.0');
      return;
    }
    if (temperature < 0.05 || temperature > 5.0) {
      showError('Temperature must be between 0.05 and 5.0');
      return;
    }
    if (cfgWeight < 0.0 || cfgWeight > 1.0) {
      showError('CFG Weight must be between 0.0 and 1.0');
      return;
    }
    if (chunkSize < 50 || chunkSize > 300) {
      showError('Chunk size must be between 50 and 300');
      return;
    }
    if (textInputValue.length > MAX_TEXT_LENGTH) {
      showError(`Text is too long. Please limit to ${MAX_TEXT_LENGTH.toLocaleString()} characters.`);
      return;
    }

    // Handle seed value
    let seed = 0;
    const seedValue = seedSelect.value;
    if (seedValue === 'custom') {
      const customSeed = customSeedInput.value;
      if (customSeed) {
        seed = parseInt(customSeed);
        if (isNaN(seed) || seed < 0 || seed > 999999) {
          showError('Custom seed must be a number between 0 and 999999');
          return;
        }
      }
    } else {
      seed = parseInt(seedValue);
    }

    // Save form state to localStorage
    saveFormState({
      textInput: textInputValue,
      exaggeration,
      temperature,
      cfgWeight,
      chunkSize,
      removeSilence,
      speed,
      pitch,
      seed: seedValue,
      customSeed: seedValue === 'custom' ? seed : ''
    });

    // Function to start generation with text files support
    function startGeneration(audioPromptFilename = null, textFilesPaths = null, voiceProfile = null) {
      window.WebSocketHandlers.setIsGenerating(true);
      
      const generationData = {
        text_input: textInputValue,
        text_files_paths: textFilesPaths,
        audio_prompt_path: audioPromptFilename,
        voice_profile: voiceProfile,
        export_formats: exportFormats,
        num_candidates: numCandidates,
        max_attempts: maxAttempts,
        whisper_model: whisperModel,
        use_faster_whisper: useFasterWhisper,
        bypass_whisper: bypassWhisper,
        use_longest_transcript: useLongestTranscript,
        enable_parallel: enableParallel,
        parallel_workers: parallelWorkers,
        validation_threshold: validationThreshold,
        to_lowercase: toLowercase,
        normalize_spacing: normalizeSpacing,
        fix_dot_letters: fixDotLetters,
        remove_reference_numbers: removeReferenceNumbers,
        normalize_audio: normalizeAudio,
        normalize_method: normalizeMethod,
        normalize_level: normalizeLevel,
        normalize_tp: normalizeTp,
        normalize_lra: normalizeLra,
        noise_reduction_method: noiseReductionMethod,
        noise_strength: noiseStrength,
        sound_words: soundWords,
        voice_similarity_threshold: voiceSimilarityThreshold,
        exaggeration,
        temperature,
        cfg_weight: cfgWeight,
        chunk_size: chunkSize,
        reduce_noise: reduceNoise,
        remove_silence: removeSilence,
        speed,
        pitch,
        seed
      };

      socket.emit('start_generation', generationData);

      generateButton.disabled = true;
      generateButton.textContent = 'Generating...';
      generateButton.classList.add('generating');
      
      // Show cancel button during generation
      if (cancelButton) {
        cancelButton.classList.remove('hide');
      }
      
      progressComplete.textContent = "";
      progressComplete.style.color = 'var(--primary-btn-bg-color)';
      
      // Clear status console and show initial message
      DOMUtils.clearStatusConsole();
      DOMUtils.addStatusMessage("🎙️ Initializing audio generation...");
      
      // Show the progress container (now with console instead of spinner)
      progressContainer.classList.remove('hide');
      progressComplete.classList.remove('hide');
    }

    // Function to upload text files if present
    async function uploadTextFiles() {
      if (!textFiles || textFiles.length === 0) {
        return null;
      }

      const formData = new FormData();
      for (let i = 0; i < textFiles.length; i++) {
        formData.append('text_files', textFiles[i]);
      }

      DOMUtils.addStatusMessage(`📁 Uploading ${textFiles.length} text file(s)...`);

      try {
        const response = await fetch('/upload_text_files', {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        if (data.success) {
          showFlashMessage(`📁 ${data.count} text file(s) uploaded successfully`, 'success');
          return data.files.map(f => f.filepath);
        } else {
          throw new Error(data.error || 'Text files upload failed');
        }
      } catch (error) {
        showFlashMessage(`Text files upload failed: ${error.message}`, 'error');
        throw error;
      }
    }

    // Function to upload audio file if present
    async function uploadAudioFile() {
      if (!audioPromptFile) {
        return null;
      }

      const formData = new FormData();
      formData.append('audio_file', audioPromptFile);

      DOMUtils.addStatusMessage("📁 Uploading reference audio...");

      try {
        const response = await fetch('/upload_audio', {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        if (data.success) {
          currentUploadedFile = data.filename;
          showFlashMessage(`📁 Reference audio uploaded: ${audioPromptFile.name}`, 'success');
          return data.filename;
        } else {
          throw new Error(data.error || 'Audio upload failed');
        }
      } catch (error) {
        showFlashMessage(`Audio upload failed: ${error.message}`, 'error');
        throw error;
      }
    }

    // Main upload and generation flow
    async function handleUploadsAndGeneration() {
      try {
        // Show initial status
        DOMUtils.clearStatusConsole();
        progressContainer.classList.remove('hide');
        progressComplete.classList.remove('hide');

        // Upload files in parallel if both are present, or sequentially
        let audioFilename = null;
        let textFilesPaths = null;

        // Check if we need to process text from textarea or text files
        if (textFiles && textFiles.length > 0) {
          // Text files are selected - use them (text input is disabled when files are selected)
          textFilesPaths = await uploadTextFiles();
        } else if (!textInputValue.trim()) {
          // No text in input box and no text files
          throw new Error('Please provide text to generate speech from (either in the text box or upload text files).');
        }

        // Upload audio file if present
        if (audioPromptFile) {
          audioFilename = await uploadAudioFile();
        }

        // Determine which voice source to use
        let finalAudioFilename = null;
        let selectedVoiceProfile = null;
        
        if (audioFilename) {
          // Use uploaded custom voice
          finalAudioFilename = audioFilename;
        } else {
          // Check simple voice dropdown first
          const voiceSelect = document.getElementById('voice-select');
          if (voiceSelect && voiceSelect.value) {
            finalAudioFilename = voiceSelect.value;
            console.log(`Using voice dropdown selection: ${voiceSelect.value}`);
          } else {
            // No voice selected - use default model voice (leave finalAudioFilename as null)
            console.log(`No voice selected - using default model voice`);
            finalAudioFilename = null;
          }
        }
        
        // Start generation with selected voice
        startGeneration(finalAudioFilename, textFilesPaths, selectedVoiceProfile);

      } catch (error) {
        // Handle any upload errors
        window.WebSocketHandlers.setIsGenerating(false);
        window.WebSocketHandlers.updateGenerateButton(false);
        progressContainer.classList.add('hide');
        progressComplete.classList.add('hide');
        showError(`Upload failed: ${error.message}`);
      }
    }

    // Start the upload and generation process
    handleUploadsAndGeneration();
  });

  function saveFormState(state) {
    Object.entries(state).forEach(([key, value]) => {
      localStorage.setItem(key, value);
    });
  }

  // Setup WebSocket handlers
  window.WebSocketHandlers.setupWebSocketHandlers(socket);

  let sampleText = "Artificial intelligence is transforming the way we interact with machines, making communication more natural and intuitive. Text-to-speech technology, powered by AI, allows computers to convert written words into spoken language with increasing fluency and realism. This capability is essential for accessibility, virtual assistants, and interactive voice response systems.";

  // Load form state from localStorage
  function loadFormState() {
    textInput.value = localStorage.getItem('textInput') || sampleText;
    document.getElementById('exaggeration').value = localStorage.getItem('exaggeration') || '0.50';
    document.getElementById('temperature').value = localStorage.getItem('temperature') || '0.80';
    document.getElementById('cfg-weight').value = localStorage.getItem('cfgWeight') || '0.50';
    document.getElementById('chunk-size').value = localStorage.getItem('chunkSize') || '300';
    document.getElementById('speed').value = localStorage.getItem('speed') || '1.0';
    document.getElementById('pitch').value = localStorage.getItem('pitch') || '0';
    document.getElementById('remove-silence').checked = localStorage.getItem('removeSilence') === 'true';
    
    // Initialize sliders and update displays
    const sliders = DOMUtils.initializeSliders();
    sliders.forEach(({ slider, display }) => {
      if (slider && display) {
        if (slider.id === 'chunk-size') {
          display.textContent = parseInt(slider.value);
        } else {
          display.textContent = parseFloat(slider.value).toFixed(2);
        }
      }
    });
    
    // Load seed settings
    const savedSeed = localStorage.getItem('seed') || '0';
    seedSelect.value = savedSeed;
    if (savedSeed === 'custom') {
      customSeedContainer.classList.remove('hide');
      customSeedInput.value = localStorage.getItem('customSeed') || '';
    }

    // Update character count
    FormValidation.updateCharacterCount();
  }

  // Load the audio list and voices on page load
  loadAudioList();
  loadPresetVoices();
  
  // Export loadAudioList function for use by WebSocket handlers
  window.AudioPlayer = {
    loadAudioList: loadAudioList,
    setIsNewCard: (value) => { isNewCard = value; }
  };

  const audioItemTemplate = document.getElementById('audio-item-template');
  let isNewCard = false;

  function loadAudioList(callback) {
    // Clear the existing audio list
    audioList.innerHTML = '';

    // Update delete all button visibility
    DOMUtils.updateDeleteAllButton();

    // Fetch the JSON data
    fetch('static/json/data.json?t=' + Date.now()) // Add cache buster
      .then(response => {
        if (!response.ok) {
          throw new Error('JSON data file not found');
        }
        return response.json();
      })
      .then(data => {
        for (const key in data) {
          if (data.hasOwnProperty(key)) {
            const item = data[key];
            createAudioCard(item, key);
          }
        }
        // Update delete all button after loading cards
        DOMUtils.updateDeleteAllButton();
      })
      .catch(error => {
        if (error.message !== 'JSON data file not found') {
          showError('Error loading audio list');
        }
        // Update delete all button even on error
        DOMUtils.updateDeleteAllButton();
      })
      .finally(() => {
        if (callback) callback();
      });
  }

  function updateDeleteAllButton() {
    if (deleteAllButton) {
      const audioCards = document.querySelectorAll('.audio-item');
      if (audioCards.length > 0) {
        deleteAllButton.style.display = 'inline-block';
        deleteAllButton.textContent = `Delete All (${audioCards.length})`;
      } else {
        deleteAllButton.style.display = 'none';
      }
    }
  }

  function createAudioCard(item, key) {
    if (!audioItemTemplate) {
      return;
    }

    const filename = item.outputFile;
    const textInputValue = item.textInput;
    const genTime = item.generationTime;
    const audioPromptPath = item.audioPromptPath || 'Default Voice';
    const exaggeration = item.exaggeration;
    const temperature = item.temperature;
    const cfgWeight = item.cfgWeight;
    const chunkSize = item.chunkSize || 300;
    const speed = item.speed;
    const pitch = item.pitch;
    const removeSilence = item.removeSilence;
    const seed = item.seed || 0;

    // Create a new audio item using the template
    const audioItem = audioItemTemplate.content.cloneNode(true);
    
    // Set up audio player
    const audioPlayer = audioItem.querySelector('.audio-player');
    if (audioPlayer) {
      audioPlayer.src = 'static/output/' + filename;
      audioPlayer.addEventListener('error', function() {
      });
    }

    // Populate card content
    const setTextContent = (selector, content) => {
      const element = audioItem.querySelector(selector);
      if (element) element.textContent = content;
    };

    setTextContent('.filename', filename);
    setTextContent('.gen-time', 'Generation Time: ' + DOMUtils.formatTime(genTime));
    setTextContent('.audio-prompt', 'Voice: ' + (audioPromptPath === null ? 'Default Voice' : audioPromptPath));
    setTextContent('.exaggeration', 'Exaggeration: ' + exaggeration);
    setTextContent('.temperature', 'Temperature: ' + temperature);
    setTextContent('.cfg-weight', 'CFG Weight: ' + cfgWeight);
    setTextContent('.chunk-size', 'Chunk Size: ' + chunkSize);
    setTextContent('.speed', 'Speed: ' + speed);
    setTextContent('.pitch', 'Pitch: ' + pitch);
    setTextContent('.remove-silence', 'RS: ' + removeSilence);
    setTextContent('.seed', 'Seed: ' + (seed === 0 ? `Random (${item.actualSeed || 'Unknown'})` : seed));
    
    // Additional metadata fields
    setTextContent('.reduce-noise', 'Reduce Noise: ' + (item.reduceNoise ? 'Yes' : 'No'));
    setTextContent('.normalize-audio', 'Normalize Audio: ' + (item.useFfmpegNormalize ? 'Yes' : 'No'));
    setTextContent('.num-candidates', 'Candidates: ' + (item.numCandidates || 3));
    setTextContent('.use-whisper', 'Whisper Validation: ' + (item.useWhisperValidation ? 'Yes' : 'No'));
    setTextContent('.validation-threshold', 'Validation Threshold: ' + (item.validationThreshold || 0.85));
    setTextContent('.whisper-model', 'Whisper Model: ' + (item.whisperModelName || 'medium'));
    setTextContent('.enable-parallel', 'Parallel Processing: ' + (item.enableParallel ? 'Yes' : 'No'));
    setTextContent('.num-workers', 'Workers: ' + (item.numWorkers || 4));

    const textInputElement = audioItem.querySelector('.text-input');
    if (textInputElement) {
      textInputElement.textContent = textInputValue;
    }

    // Set up event handlers
    const downloadButton = audioItem.querySelector('.download-button');
    if (downloadButton) {
      downloadButton.addEventListener('click', function(event) {
        event.preventDefault();
        DOMUtils.downloadFile('static/output/' + filename, filename);
      });
    }

    const deleteButton = audioItem.querySelector('.delete-button');
    if (deleteButton) {
      deleteButton.addEventListener('click', function(event) {
        event.preventDefault();
        const parentCard = event.target.closest('.card');
        
        // No confirmation popup - just delete immediately
        if (parentCard && parentCard.classList.contains('new-card')) {
          progressComplete.textContent = "";
        }
        
        if (parentCard) {
          parentCard.style.opacity = '0.5';
          parentCard.style.transition = 'opacity 0.3s ease';
        }
        
        deleteAudioFile(filename, parentCard);
      });
    }

    // Add new card styling if needed
    const cardElement = audioItem.querySelector('.card');
    if (isNewCard && cardElement) {
      cardElement.classList.add('new-card');
    }

    audioList.appendChild(audioItem);
  }


  function deleteAudioFile(filename, cardElement) {
    fetch('static/output/' + filename, { method: 'DELETE' })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        showFlashMessage(`🗑️ Deleted: ${filename}`, 'success');
        
        // Remove the card from the DOM
        if (cardElement) {
          setTimeout(() => {
            cardElement.remove();
            DOMUtils.updateDeleteAllButton();
          }, 300);
        }
      })
      .catch(error => {
        showFlashMessage(`❌ Failed to delete: ${filename}`, 'error');
        
        // Restore card opacity on error
        if (cardElement) {
          cardElement.style.opacity = '1';
        }
      });
  }

  // Handle more info links
  function initializeMoreInfoLinks() {
    const moreInfoLinks = document.querySelectorAll('.more-info-link');

    moreInfoLinks.forEach(link => {
      link.addEventListener('click', function() {
        const moreInfo = this.previousElementSibling;
        if (moreInfo) {
          moreInfo.classList.toggle('show');
          this.textContent = moreInfo.classList.contains('show') ? 'Less Info' : this.dataset.defaultText;
        }
      });
    
      link.dataset.defaultText = link.textContent; 
    });
  }

  // Handle select container styling
  function initializeSelectContainers() {
    const selectContainers = document.querySelectorAll('.select-container');

    selectContainers.forEach(function(container) {
      const select = container.querySelector('select');
      if (select) {
        select.addEventListener('focus', function() {
          container.classList.add('open');
        });
        
        select.addEventListener('blur', function() {
          container.classList.remove('open');
        });
      }
    });
  }

  // Help dropdown functionality
  function initializeHelpDropdown() {
    const helpDropdown = document.getElementById('help-dropdown');
    const helpContent = document.getElementById('help-content');
    
    if (!helpDropdown || !helpContent) return;
    
    const helpTexts = {
      'break-tags': `
        <h3>Break Tags for Custom Pauses</h3>
        <p>Add custom pauses to your text using break tags. Use <code>&lt;break time="1.5s" /&gt;</code> for a 1.5 second pause, or <code>&lt;break time="500ms" /&gt;</code> for a 500 millisecond pause.</p>
        <p><strong>Best Practices:</strong> Place break tags between complete sentences or paragraphs rather than in the middle of sentences.</p>
        <p><strong>Examples:</strong></p>
        <ul>
          <li>"Hello there. &lt;break time="2s" /&gt; How are you today?"</li>
          <li>"First paragraph here. &lt;break time="1.5s" /&gt; Second paragraph starts here."</li>
          <li>"End of sentence. &lt;break time="800ms" /&gt; Next sentence begins."</li>
        </ul>
      `,
      'reference-audio': `
        <h3>Voice Selection</h3>
        <p><strong>Preset Voices:</strong> A dropdown which allows you to choose from any voice audio reference samples you have placed in the (static > voices) directory.</p>
        <p><strong>Custom Reference Audio:</strong> Upload your own audio file (WAV, MP3, FLAC) to clone specific voice characteristics. The file should be clear speech, ideally 3-10 seconds long.</p>
        <p><strong>Default Voice:</strong> If no preset or custom audio is selected, Chatterbox will use its default built-in voice.</p>
        <p><strong>Note:</strong> Only one voice source can be used at a time - selecting a preset voice will disable custom upload and vice versa.</p>
      `,
      'text-files': `
        <h3>Text Files</h3>
        <p>Upload multiple text files (.txt) to process them in batch. Enable "Generate Separate Audio Per Text File" to create individual audio files for each text file.</p>
      `,
      'exaggeration': `
        <h3>Exaggeration</h3>
        <p>Controls the emotional intensity and expressiveness of the generated speech.</p>
        <ul>
          <li><strong>Lower values (0.25-0.5):</strong> More neutral, conversational speech</li>
          <li><strong>Higher values (0.7-2.0):</strong> More dramatic, expressive speech (may become unstable at extreme values)</li>
        </ul>
      `,
      'temperature': `
        <h3>Temperature</h3>
        <p>Controls the randomness and creativity in the speech generation.</p>
        <ul>
          <li><strong>Lower values (0.1-0.5):</strong> More consistent, predictable speech</li>
          <li><strong>Higher values (0.8-2.0):</strong> More variation and creativity (may reduce quality)</li>
        </ul>
      `,
      'cfg-weight': `
        <h3>CFG Weight / Pace Control</h3>
        <p>Controls the pacing and adherence to the prompt.</p>
        <ul>
          <li><strong>Lower values (0.0-0.3):</strong> Faster, more relaxed speech - ideal for expressive or dramatic content</li>
          <li><strong>Higher values (0.6-1.0):</strong> Slower, more deliberate speech that follows the prompt more closely</li>
        </ul>
        <p><strong>Note:</strong> For fast-speaking reference voices, lower CFG weights work better.</p>
      `,
      'chunk-size': `
        <h3>Chunk Size</h3>
        <p>Controls how much text is processed in each TTS generation call.</p>
        <ul>
          <li><strong>Larger chunks (250-300):</strong> More natural speech flow, fewer processing calls, but may be less stable with complex text</li>
          <li><strong>Smaller chunks (50-150):</strong> More reliable for difficult text but may sound more fragmented</li>
        </ul>
        <p>Chatterbox can handle up to 300 characters per chunk.</p>
      `,
      'seed': `
        <h3>Random Seed</h3>
        <p>Controls the randomness of generation for reproducible results.</p>
        <ul>
          <li><strong>0:</strong> Random output each time</li>
          <li><strong>Specific number:</strong> Identical results with the same settings</li>
        </ul>
        <p>Useful for comparing different parameter changes or recreating exact outputs.</p>
      `,
      'speed-pitch': `
        <h3>Speed and Pitch Adjustment</h3>
        <p>Post-processing effects applied after Chatterbox generates the audio.</p>
        <ul>
          <li><strong>Speed:</strong> Changes playback rate without affecting pitch</li>
          <li><strong>Pitch:</strong> Shifts the audio up or down in semitones</li>
        </ul>
        <p><strong>Note:</strong> Extreme values may introduce artifacts like echo or reverb. For best results, use moderate adjustments.</p>
      `,
      'export-format': `
        <h3>Export Format</h3>
        <p>Choose the audio format for your generated files:</p>
        <ul>
          <li><strong>WAV:</strong> Uncompressed, highest quality</li>
          <li><strong>MP3:</strong> Compressed, smaller file size</li>
          <li><strong>FLAC:</strong> Lossless compression</li>
        </ul>
      `,
      'whisper': `
        <h3>Whisper Settings</h3>
        <p>Whisper is used for transcription and voice analysis.</p>
        <ul>
          <li><strong>Model Size:</strong> Larger models are more accurate but use more VRAM</li>
          <li><strong>faster-whisper:</strong> More efficient backend (recommended)</li>
          <li><strong>Bypass Whisper:</strong> Skip transcription checking</li>
          <li><strong>Use Longest Transcript:</strong> Use the longest transcription attempt on failure</li>
        </ul>
        <p><strong>Parallel Processing:</strong> Enable parallel processing to speed up generation (when using Whisper) by processing multiple chunks simultaneously.</p>
        <p><strong>Parallel Workers:</strong> Number of parallel workers (more workers = faster generation but higher memory/CPU usage).</p>
      `,
      'parallel': `
        <h3>Generation Quality</h3>
        <p><strong>Candidates Per Chunk:</strong> Number of alternative audio samples generated for each text chunk. More candidates = higher quality but slower generation.</p>
        <p><strong>Max Attempts Per Candidate:</strong> How many times to retry failed generation attempts before moving to the next candidate.</p>
      `,
      'text-processing': `
        <h3>Text Processing Options</h3>
        <ul>
          <li><strong>Convert to Lowercase:</strong> Convert all text to lowercase before processing</li>
          <li><strong>Normalize Spacing:</strong> Clean up extra spaces, tabs, and line breaks in text</li>
          <li><strong>Fix Dot Letters:</strong> Convert acronyms like "J.R.R." to "J R R" for better pronunciation</li>
          <li><strong>Remove Reference Numbers:</strong> Remove citation numbers and footnote references from text</li>
          <li><strong>Remove Silence:</strong> Remove silent pauses from the generated audio to make it more compact</li>
        </ul>
      `,
      'noise-reduction': `
        <h3>Noise Reduction (Speech Optimized)</h3>
        <p>Remove background noise from generated TTS audio using ANLMDN (Non-Local Means) algorithm, specifically optimized for speech preservation.</p>
        <ul>
          <li><strong>ANLMDN Algorithm:</strong> Uses non-local means filtering to reduce noise while preserving speech quality</li>
          <li><strong>Strength:</strong> Controls how aggressive the noise reduction is (0.1-1.0)</li>
          <li><strong>FFmpeg Required:</strong> This feature requires FFmpeg to be installed</li>
        </ul>
        <p><strong>Note:</strong> ANLMDN is specifically chosen for TTS applications as it excels at preserving speech characteristics while effectively removing background noise.</p>
      `,
      'normalization': `
        <h3>Audio Normalization</h3>
        <p>Normalize audio levels using FFmpeg.</p>
        <ul>
          <li><strong>EBU Loudness:</strong> Professional loudness standard</li>
          <li><strong>Peak Normalization:</strong> Simple peak-based normalization</li>
          <li><strong>Target Loudness:</strong> Desired loudness level in dB</li>
          <li><strong>True Peak:</strong> Maximum peak level</li>
          <li><strong>Loudness Range:</strong> Dynamic range control</li>
        </ul>
      `,
      'voice-similarity': `
        <h3>Voice Similarity Validation</h3>
        <p>Automatically validates that generated audio chunks match the reference voice characteristics using audio analysis.</p>
        <ul>
          <li><strong>0.0 (Disabled):</strong> No validation - accepts all generated chunks</li>
          <li><strong>0.5-0.6:</strong> Moderate validation - rejects clearly different voices</li>
          <li><strong>0.7-0.8:</strong> Strict validation - enforces close voice matching</li>
          <li><strong>0.8+:</strong> Very strict - may reject valid chunks</li>
        </ul>
        <p><strong>Benefits:</strong> Prevents voice drift and American accent bleeding into non-American voices</p>
        <p><strong>Requirements:</strong> Requires librosa Python library for audio analysis</p>
        <p><strong>Note:</strong> Higher values increase generation time as rejected chunks are regenerated</p>
      `,
      'sound-words': `
        <h3>Sound Words</h3>
        <p>Remove or replace specific words or sounds from the generated audio.</p>
        <ul>
          <li><strong>Format:</strong> Use comma or newline separation</li>
          <li><strong>Remove:</strong> "sss" (removes the sound)</li>
          <li><strong>Replace:</strong> "ahh=>um" (replaces "ahh" with "um")</li>
        </ul>
        <p><strong>Examples:</strong> sss, ss, ahh=>um, hmm</p>
      `
    };
    
    helpDropdown.addEventListener('change', function() {
      const selectedValue = this.value;
      
      if (selectedValue && helpTexts[selectedValue]) {
        helpContent.innerHTML = helpTexts[selectedValue];
        helpContent.classList.remove('hide');
      } else {
        helpContent.classList.add('hide');
      }
    });
  }


  // Initialize components
  loadFormState();
  DOMUtils.initializeMoreInfoLinks();
  DOMUtils.initializeSelectContainers();
  initializeHelpDropdown();
  
  // Initialize console buttons
  const clearConsoleButton = document.getElementById('clear-console-button');
  if (clearConsoleButton) {
    clearConsoleButton.addEventListener('click', function() {
      DOMUtils.clearStatusConsole();
    });
  }
  

  // Keyboard shortcuts
  document.addEventListener('keydown', function(event) {
    // Ctrl/Cmd + Enter to generate
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
      if (!window.WebSocketHandlers.getIsGenerating() && generateButton && !generateButton.disabled) {
        generateButton.click();
      }
    }
    
    // Escape to stop generation (if implemented server-side)
    if (event.key === 'Escape' && window.WebSocketHandlers.getIsGenerating()) {
      // Could implement stop generation functionality here
    }
  });

  // Auto-save form state periodically
  setInterval(() => {
    if (!window.WebSocketHandlers.getIsGenerating()) {
      const state = {
        textInput: textInput.value,
        exaggeration: document.getElementById('exaggeration').value,
        temperature: document.getElementById('temperature').value,
        cfgWeight: document.getElementById('cfg-weight').value,
        chunkSize: document.getElementById('chunk-size').value,
        removeSilence: document.getElementById('remove-silence').checked,
        speed: document.getElementById('speed').value,
        pitch: document.getElementById('pitch').value,
        seed: seedSelect.value,
        customSeed: seedSelect.value === 'custom' ? customSeedInput.value : ''
      };
      saveFormState(state);
    }
  }, 30000); // Auto-save every 30 seconds

  // Voice selection helper functions
  async function loadPresetVoices() {
    try {
      const response = await fetch('/get_voice_files');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      
      // Populate the simplified voice dropdown
      populateVoiceDropdown(data.legacy_voices || []);
    } catch (error) {
      console.error('Error loading preset voices:', error);
      showError('Failed to load preset voices');
    }
  }

  function populateVoiceDropdown(voices) {
    const voiceSelect = document.getElementById('voice-select');
    if (!voiceSelect) return;

    // Clear existing options except the first one
    voiceSelect.innerHTML = '<option value="">Select a voice...</option>';

    // Add voice options
    voices.forEach(voice => {
      const option = document.createElement('option');
      option.value = voice.filename;
      option.textContent = voice.display_name || voice.filename;
      voiceSelect.appendChild(option);
    });

    console.log(`Loaded ${voices.length} voices into dropdown`);
  }

  function clearFileUpload() {
    fileInput.value = '';
    fileNameDisplay.textContent = 'No file selected';
    clearAudioButton.classList.add('hide');
    currentUploadedFile = null;
  }

  function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

});