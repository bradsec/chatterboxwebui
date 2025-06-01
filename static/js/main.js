document.addEventListener('DOMContentLoaded', function() {
  const socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port);

  // DOM element references
  const generateButton = document.getElementById('generate-button');
  const resetButton = document.getElementById('reset-button');
  const audioList = document.getElementById('audio-list');
  const progressContainer = document.querySelector('.progress-container');
  const progressComplete = document.querySelector('.progress-complete');
  const fileInput = document.getElementById('audio-prompt');
  const fileNameDisplay = document.getElementById('file-name-display');
  const clearAudioButton = document.getElementById('clear-audio-button');
  const seedSelect = document.getElementById('seed');
  const customSeedContainer = document.getElementById('custom-seed-container');
  const customSeedInput = document.getElementById('custom-seed-input');
  const textInput = document.getElementById('text-input');

  // Track current uploaded file
  let currentUploadedFile = null;
  let isGenerating = false;

  // Debounce function for preventing rapid clicks
  function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }

  // Input validation functions
  function validateText(text) {
    if (!text || text.trim() === '') {
      return { valid: false, error: 'Text is empty.' };
    }
    if (text.length > 10000) {
      return { valid: false, error: 'Text is too long. Please limit to 10,000 characters.' };
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

  function showError(message) {
    progressComplete.textContent = `Error: ${message}`;
    progressComplete.style.color = '#ff4444';
    progressContainer.classList.add('hide');
    
    setTimeout(() => {
      progressComplete.textContent = "";
      progressComplete.style.color = 'var(--primary-btn-bg-color)';
    }, 5000);
  }

  function updateCharacterCount() {
    const text = textInput.value;
    const charCount = text.length;
    const maxChars = 10000;
    
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
    
    counter.textContent = `${charCount}/${maxChars} characters`;
    counter.style.color = charCount > maxChars ? '#ff4444' : 'var(--font-color)';
  }

  // Handle file input display with validation
  fileInput.addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
      const validation = validateFile(file);
      if (!validation.valid) {
        showError(validation.error);
        fileInput.value = '';
        fileNameDisplay.textContent = 'No file selected';
        clearAudioButton.classList.add('hide');
        return;
      }
      
      fileNameDisplay.textContent = file.name;
      fileNameDisplay.style.color = 'var(--font-color)';
      clearAudioButton.classList.remove('hide');
      currentUploadedFile = null; // Reset since user selected a new file
    } else {
      fileNameDisplay.textContent = 'No file selected';
      fileNameDisplay.style.color = 'var(--font-color)';
      clearAudioButton.classList.add('hide');
      currentUploadedFile = null;
    }
  });

  // Handle clear audio button
  clearAudioButton.addEventListener('click', debounce(clearReferenceAudio, 300));

  function clearReferenceAudio() {
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
          console.log('Reference audio cleared from server:', data.message);
        } else {
          console.error('Error clearing reference audio:', data.error);
        }
      })
      .catch(error => {
        console.error('Error clearing reference audio:', error);
      });
    }
    
    currentUploadedFile = null;
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

  // Character counter for text input
  textInput.addEventListener('input', debounce(updateCharacterCount, 100));

  // Handle slider value updates
  const sliders = [
    { slider: document.getElementById('exaggeration'), display: document.getElementById('exaggeration-value') },
    { slider: document.getElementById('temperature'), display: document.getElementById('temperature-value') },
    { slider: document.getElementById('cfg-weight'), display: document.getElementById('cfg-weight-value') },
    { slider: document.getElementById('chunk-size'), display: document.getElementById('chunk-size-value') }
  ];

  sliders.forEach(({ slider, display }) => {
    if (slider && display) {
      slider.addEventListener('input', function() {
        if (slider.id === 'chunk-size') {
          display.textContent = parseInt(this.value);
        } else {
          display.textContent = parseFloat(this.value).toFixed(2);
        }
      });
      
      // Initialize display value
      if (slider.id === 'chunk-size') {
        display.textContent = parseInt(slider.value);
      } else {
        display.textContent = parseFloat(slider.value).toFixed(2);
      }
    }
  });

  // Reset button functionality
  resetButton.addEventListener('click', function() {
    if (confirm('Reset all settings to defaults? This will clear all form values including the reference audio file.')) {
      resetToDefaults();
    }
  });

  function resetToDefaults() {
    // Clear reference audio first
    clearReferenceAudio();
    
    // Reset text input to sample text
    textInput.value = sampleText;
    updateCharacterCount();
    
    // Reset sliders to defaults
    document.getElementById('exaggeration').value = '0.50';
    document.getElementById('temperature').value = '0.80';
    document.getElementById('cfg-weight').value = '0.50';
    document.getElementById('chunk-size').value = '130';
    
    // Update slider displays
    document.getElementById('exaggeration-value').textContent = '0.50';
    document.getElementById('temperature-value').textContent = '0.80';
    document.getElementById('cfg-weight-value').textContent = '0.50';
    document.getElementById('chunk-size-value').textContent = '130';
    
    // Reset dropdowns to defaults
    document.getElementById('seed').value = '0';
    document.getElementById('speed').value = '1.0';
    document.getElementById('pitch').value = '0';
    
    // Hide custom seed input if visible
    customSeedContainer.classList.add('hide');
    customSeedInput.value = '';
    
    // Reset checkboxes
    document.getElementById('reduce-noise').checked = false;
    document.getElementById('remove-silence').checked = false;
    
    // Clear localStorage
    clearStoredSettings();
    
    console.log('All settings reset to defaults');
  }

  function clearStoredSettings() {
    const settingsKeys = [
      'textInput', 'exaggeration', 'temperature', 'cfgWeight', 'chunkSize',
      'reduceNoise', 'removeSilence', 'speed', 'pitch', 'seed', 'customSeed'
    ];
    
    settingsKeys.forEach(key => localStorage.removeItem(key));
  }

  // Form submission with enhanced validation
  document.getElementById('generator-form').addEventListener('submit', function(event) {
    event.preventDefault();

    if (isGenerating) {
      console.log('Generation already in progress');
      return;
    }

    const textInputValue = textInput.value;
    const audioPromptFile = fileInput.files[0];

    // Validate inputs
    const textValidation = validateText(textInputValue);
    if (!textValidation.valid) {
      showError(textValidation.error);
      textInput.focus();
      return;
    }

    const fileValidation = validateFile(audioPromptFile);
    if (!fileValidation.valid) {
      showError(fileValidation.error);
      return;
    }

    // Get form values with validation
    const exaggeration = parseFloat(document.getElementById('exaggeration').value);
    const temperature = parseFloat(document.getElementById('temperature').value);
    const cfgWeight = parseFloat(document.getElementById('cfg-weight').value);
    const chunkSize = parseInt(document.getElementById('chunk-size').value);
    const reduceNoise = document.getElementById('reduce-noise').checked;
    const removeSilence = document.getElementById('remove-silence').checked;
    const speed = parseFloat(document.getElementById('speed').value);
    const pitch = parseInt(document.getElementById('pitch').value);

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
      reduceNoise,
      removeSilence,
      speed,
      pitch,
      seed: seedValue,
      customSeed: seedValue === 'custom' ? seed : ''
    });

    // Function to start generation
    function startGeneration(audioPromptFilename = null) {
      isGenerating = true;
      socket.emit('start_generation', {
        text_input: textInputValue,
        audio_prompt_path: audioPromptFilename,
        exaggeration,
        temperature,
        cfg_weight: cfgWeight,
        chunk_size: chunkSize,
        reduce_noise: reduceNoise,
        remove_silence: removeSilence,
        speed,
        pitch,
        seed
      });

      generateButton.disabled = true;
      progressComplete.textContent = "";
      progressComplete.style.color = 'var(--primary-btn-bg-color)';
      
      // Reset the progress bar
      if (progress && circumference) {
        progress.style.strokeDashoffset = circumference;
        progress.style.animation = 'spin 4s linear infinite';
      }
      if (progressText) {
        progressText.textContent = '0%';
      }

      // Show the progress bar
      progressContainer.classList.remove('hide');
      progressComplete.classList.remove('hide');
    }

    // Handle file upload if present
    if (audioPromptFile) {
      const formData = new FormData();
      formData.append('audio_file', audioPromptFile);

      // Show uploading status
      progressComplete.textContent = "Uploading reference audio...";
      progressContainer.classList.remove('hide');
      progressComplete.classList.remove('hide');

      fetch('/upload_audio', {
        method: 'POST',
        body: formData
      })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        if (data.success) {
          currentUploadedFile = data.filename;
          startGeneration(data.filename);
        } else {
          throw new Error(data.error || 'Upload failed');
        }
      })
      .catch(error => {
        console.error('Upload error:', error);
        showError('Error uploading file: ' + error.message);
        isGenerating = false;
        generateButton.disabled = false;
        progressContainer.classList.add('hide');
        progressComplete.classList.add('hide');
      });
    } else {
      // No file, start generation directly
      startGeneration();
    }
  });

  function saveFormState(state) {
    Object.entries(state).forEach(([key, value]) => {
      localStorage.setItem(key, value);
    });
  }

  // Progress bar handling - Initialize variables at module level
  let progress = document.querySelector('#progress');
  let progressText = document.querySelector('#progress-text');
  let radius, circumference;
  
  // Initialize progress bar
  function initializeProgressBar() {
    progress = document.querySelector('#progress');
    progressText = document.querySelector('#progress-text');
    
    if (progress && progress.r && progress.r.baseVal) {
      radius = progress.r.baseVal.value;
      circumference = 2 * Math.PI * radius;
      
      progress.style.strokeDasharray = `${circumference} ${circumference}`;
      progress.style.strokeDashoffset = `${circumference}`;
    }
  }
  
  function setProgress(percent) {
    if (progress && circumference) {
      progress.style.animation = 'none';
      const offset = circumference - percent / 100 * circumference;
      progress.style.strokeDashoffset = offset;
    }
    if (progressText) {
      progressText.textContent = `${Math.round(percent)}%`;
    }
  }
  
  // Socket event handlers
  socket.on('generation_progress', function(data) {
    console.log('Generation progress:', data.progress);
    setProgress(data.progress * 100);
  });  

  socket.on('generation_complete', function(data) {
    console.log('Generation completed:', data);
  
    isGenerating = false;
    generateButton.disabled = false;
    progressContainer.classList.add('hide');
    progressComplete.textContent = "Generation completed in " + formatTime(data.generation_time) + ".";
    progressComplete.style.color = 'var(--primary-btn-bg-color)';
    
    // Scroll to the completion message
    progressComplete.scrollIntoView({ behavior: 'smooth' });

    // Highlight new card
    isNewCard = true;
  
    // Load the updated audio list
    if (progressContainer.classList.contains('hide')) {
      loadAudioList(() => {
        isNewCard = false;
      });
    }
  });

  socket.on('error', function(data) {
    console.error('Generation error:', data.error);
    isGenerating = false;
    generateButton.disabled = false;
    showError(data.error);
  });

  socket.on('connect_error', function(error) {
    console.error('Socket connection error:', error);
    showError('Connection error. Please refresh the page.');
  });

  socket.on('disconnect', function(reason) {
    console.log('Socket disconnected:', reason);
    if (isGenerating) {
      showError('Connection lost during generation. Please try again.');
      isGenerating = false;
      generateButton.disabled = false;
      progressContainer.classList.add('hide');
    }
  });

  let sampleText = "Artificial intelligence is transforming the way we interact with machines, making communication more natural and intuitive. Text-to-speech technology, powered by AI, allows computers to convert written words into spoken language with increasing fluency and realism. This capability is essential for accessibility, virtual assistants, and interactive voice response systems.";

  // Load form state from localStorage
  function loadFormState() {
    textInput.value = localStorage.getItem('textInput') || sampleText;
    document.getElementById('exaggeration').value = localStorage.getItem('exaggeration') || '0.50';
    document.getElementById('temperature').value = localStorage.getItem('temperature') || '0.80';
    document.getElementById('cfg-weight').value = localStorage.getItem('cfgWeight') || '0.50';
    document.getElementById('chunk-size').value = localStorage.getItem('chunkSize') || '130';
    document.getElementById('speed').value = localStorage.getItem('speed') || '1.0';
    document.getElementById('pitch').value = localStorage.getItem('pitch') || '0';
    document.getElementById('reduce-noise').checked = localStorage.getItem('reduceNoise') === 'true';
    document.getElementById('remove-silence').checked = localStorage.getItem('removeSilence') === 'true';
    
    // Update slider displays after loading values
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
    updateCharacterCount();
  }

  // Load the audio list on page load
  loadAudioList();

  const audioItemTemplate = document.getElementById('audio-item-template');
  let isNewCard = false;

  function loadAudioList(callback) {
    // Clear the existing audio list
    audioList.innerHTML = '';

    // Fetch the JSON data
    fetch('static/json/data.json')
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
      })
      .catch(error => {
        if (error.message === 'JSON data file not found') {
          console.log('data.json file does not exist');
        } else {
          console.error('Error loading audio list:', error);
        }
      })
      .finally(() => {
        if (callback) callback();
      });
  }

  function createAudioCard(item, key) {
    if (!audioItemTemplate) {
      console.error('Audio item template not found');
      return;
    }

    const filename = item.outputFile;
    const textInputValue = item.textInput;
    const genTime = item.generationTime;
    const audioPromptPath = item.audioPromptPath || 'Default Voice';
    const exaggeration = item.exaggeration;
    const temperature = item.temperature;
    const cfgWeight = item.cfgWeight;
    const chunkSize = item.chunkSize || 130;
    const speed = item.speed;
    const pitch = item.pitch;
    const reduceNoise = item.reduceNoise;
    const removeSilence = item.removeSilence;
    const seed = item.seed || 0;

    // Create a new audio item using the template
    const audioItem = audioItemTemplate.content.cloneNode(true);
    
    // Set up audio player
    const audioPlayer = audioItem.querySelector('.audio-player');
    if (audioPlayer) {
      audioPlayer.src = 'static/output/' + filename;
      audioPlayer.addEventListener('error', function() {
        console.error('Error loading audio file:', filename);
      });
    }

    // Populate card content
    const setTextContent = (selector, content) => {
      const element = audioItem.querySelector(selector);
      if (element) element.textContent = content;
    };

    setTextContent('.filename', filename);
    setTextContent('.gen-time', 'Generation Time: ' + formatTime(genTime));
    setTextContent('.audio-prompt', 'Voice: ' + (audioPromptPath === null ? 'Default Voice' : audioPromptPath));
    setTextContent('.exaggeration', 'Exaggeration: ' + exaggeration);
    setTextContent('.temperature', 'Temperature: ' + temperature);
    setTextContent('.cfg-weight', 'CFG Weight: ' + cfgWeight);
    setTextContent('.chunk-size', 'Chunk Size: ' + chunkSize);
    setTextContent('.speed', 'Speed: ' + speed);
    setTextContent('.pitch', 'Pitch: ' + pitch);
    setTextContent('.reduce-noise', 'RN: ' + reduceNoise);
    setTextContent('.remove-silence', 'RS: ' + removeSilence);
    setTextContent('.seed', 'Seed: ' + (seed === 0 ? 'Random' : seed));

    const textInputElement = audioItem.querySelector('.text-input');
    if (textInputElement) {
      textInputElement.textContent = textInputValue;
    }

    // Set up event handlers
    const downloadButton = audioItem.querySelector('.download-button');
    if (downloadButton) {
      downloadButton.addEventListener('click', function(event) {
        event.preventDefault();
        downloadFile('static/output/' + filename, filename);
      });
    }

    const deleteButton = audioItem.querySelector('.delete-button');
    if (deleteButton) {
      deleteButton.addEventListener('click', function(event) {
        event.preventDefault();
        const parentCard = event.target.closest('.card');
        
        if (confirm('Are you sure you want to delete this audio file?')) {
          if (parentCard && parentCard.classList.contains('new-card')) {
            progressComplete.textContent = "";
          }
          
          if (parentCard) {
            parentCard.classList.add('hide');
          }
          deleteAudioFile(filename);
        }
      });
    }

    // Add new card styling if needed
    const cardElement = audioItem.querySelector('.card');
    if (isNewCard && cardElement) {
      cardElement.classList.add('new-card');
    }

    audioList.appendChild(audioItem);
  }

  function downloadFile(url, filename) {
    try {
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      link.target = '_blank';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error('Error downloading file:', error);
      showError('Error downloading file');
    }
  }

  function deleteAudioFile(filename) {
    fetch('static/output/' + filename, { method: 'DELETE' })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        console.log('File deleted:', filename);
      })
      .catch(error => {
        console.error('Error deleting file:', error);
        showError('Error deleting file');
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

  function formatTime(seconds) {
    if (typeof seconds !== 'number' || isNaN(seconds)) {
      return '0 seconds';
    }

    let hours = Math.floor(seconds / 3600);
    let minutes = Math.floor((seconds % 3600) / 60);
    seconds = Math.floor(seconds % 60);

    let timeStr = '';
    
    if (hours > 0) {
      timeStr += `${hours} hour${hours > 1 ? 's' : ''}, `;
    }
    
    if (minutes > 0) {
      timeStr += `${minutes} minute${minutes > 1 ? 's' : ''}, `;
    }
  
    timeStr += `${seconds} second${seconds !== 1 ? 's' : ''}`;
    return timeStr;
  }

  // Initialize components
  initializeProgressBar();
  loadFormState();
  initializeMoreInfoLinks();
  initializeSelectContainers();

  // Keyboard shortcuts
  document.addEventListener('keydown', function(event) {
    // Ctrl/Cmd + Enter to generate
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
      if (!isGenerating && generateButton && !generateButton.disabled) {
        generateButton.click();
      }
    }
    
    // Escape to stop generation (if implemented server-side)
    if (event.key === 'Escape' && isGenerating) {
      // Could implement stop generation functionality here
      console.log('Escape pressed during generation');
    }
  });

  // Auto-save form state periodically
  setInterval(() => {
    if (!isGenerating) {
      const state = {
        textInput: textInput.value,
        exaggeration: document.getElementById('exaggeration').value,
        temperature: document.getElementById('temperature').value,
        cfgWeight: document.getElementById('cfg-weight').value,
        chunkSize: document.getElementById('chunk-size').value,
        reduceNoise: document.getElementById('reduce-noise').checked,
        removeSilence: document.getElementById('remove-silence').checked,
        speed: document.getElementById('speed').value,
        pitch: document.getElementById('pitch').value,
        seed: seedSelect.value,
        customSeed: seedSelect.value === 'custom' ? customSeedInput.value : ''
      };
      saveFormState(state);
    }
  }, 30000); // Auto-save every 30 seconds

  console.log('Chatterbox Web UI initialized successfully');
});