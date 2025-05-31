document.addEventListener('DOMContentLoaded', function() {
  const socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port);

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

  // Track current uploaded file
  let currentUploadedFile = null;

  // Handle file input display
  fileInput.addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
      fileNameDisplay.textContent = file.name;
      fileNameDisplay.style.color = 'var(--font-color)';
      clearAudioButton.classList.remove('hide');
      // Reset currentUploadedFile since user selected a new file
      currentUploadedFile = null;
    } else {
      fileNameDisplay.textContent = 'No file selected';
      fileNameDisplay.style.color = 'var(--font-color)';
      clearAudioButton.classList.add('hide');
      currentUploadedFile = null;
    }
  });

  // Handle clear audio button
  clearAudioButton.addEventListener('click', function() {
    clearReferenceAudio();
  });

  function clearReferenceAudio() {
    // Clear the file input
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

  // Handle slider value updates
  const sliders = [
    { slider: document.getElementById('exaggeration'), display: document.getElementById('exaggeration-value') },
    { slider: document.getElementById('temperature'), display: document.getElementById('temperature-value') },
    { slider: document.getElementById('cfg-weight'), display: document.getElementById('cfg-weight-value') },
    { slider: document.getElementById('chunk-size'), display: document.getElementById('chunk-size-value') }
  ];

  sliders.forEach(({ slider, display }) => {
    slider.addEventListener('input', function() {
      if (slider.id === 'chunk-size') {
        // Show chunk size as integer
        display.textContent = parseInt(this.value);
      } else {
        // Show other values as decimals
        display.textContent = parseFloat(this.value).toFixed(2);
      }
    });
    
    // Initialize display value
    if (slider.id === 'chunk-size') {
      display.textContent = parseInt(slider.value);
    } else {
      display.textContent = parseFloat(slider.value).toFixed(2);
    }
  });

  // Reset button functionality
  resetButton.addEventListener('click', function() {
    // Confirm reset action
    if (confirm('Reset all settings to defaults? This will clear all form values including the reference audio file.')) {
      resetToDefaults();
    }
  });

  function resetToDefaults() {
    // Clear reference audio first
    clearReferenceAudio();
    
    // Reset text input to sample text
    document.getElementById('text-input').value = sampleText;
    
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
    document.getElementById('custom-seed-input').value = '';
    
    // Reset checkboxes
    document.getElementById('reduce-noise').checked = false;
    document.getElementById('remove-silence').checked = false;
    
    // Clear localStorage
    localStorage.removeItem('textInput');
    localStorage.removeItem('exaggeration');
    localStorage.removeItem('temperature');
    localStorage.removeItem('cfgWeight');
    localStorage.removeItem('chunkSize');
    localStorage.removeItem('reduceNoise');
    localStorage.removeItem('removeSilence');
    localStorage.removeItem('speed');
    localStorage.removeItem('pitch');
    localStorage.removeItem('seed');
    localStorage.removeItem('customSeed');
    
    console.log('All settings reset to defaults');
  }

  document.getElementById('generator-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const textInput = document.getElementById('text-input').value;
    const audioPromptFile = document.getElementById('audio-prompt').files[0];
    const exaggeration = document.getElementById('exaggeration').value;
    const temperature = document.getElementById('temperature').value;
    const cfgWeight = document.getElementById('cfg-weight').value;
    const chunkSize = document.getElementById('chunk-size').value;
    const reduceNoise = document.getElementById('reduce-noise').checked;
    const removeSilence = document.getElementById('remove-silence').checked;
    const speed = document.getElementById('speed').value;
    const pitch = document.getElementById('pitch').value;
    
    // Handle seed value
    let seed = 0;
    const seedValue = document.getElementById('seed').value;
    if (seedValue === 'custom') {
      const customSeed = document.getElementById('custom-seed-input').value;
      seed = customSeed ? parseInt(customSeed) : 0;
    } else {
      seed = parseInt(seedValue);
    }

    // Check if text is empty
    if (textInput.trim() === '') {
      document.getElementById('text-input').focus();
      return;
    }

    // Check text length
    if (textInput.length > 10000) {
      alert('Text is too long. Please limit to 10,000 characters.');
      return;
    }

    // Save form state to localStorage
    localStorage.setItem('textInput', textInput);
    localStorage.setItem('exaggeration', exaggeration);
    localStorage.setItem('temperature', temperature);
    localStorage.setItem('cfgWeight', cfgWeight);
    localStorage.setItem('chunkSize', chunkSize);
    localStorage.setItem('reduceNoise', reduceNoise);
    localStorage.setItem('removeSilence', removeSilence);
    localStorage.setItem('speed', speed);
    localStorage.setItem('pitch', pitch);
    localStorage.setItem('seed', seedValue);
    if (seedValue === 'custom') {
      localStorage.setItem('customSeed', seed);
    }

    // Function to start generation
    function startGeneration(audioPromptFilename = null) {
      socket.emit('start_generation', {
        text_input: textInput,
        audio_prompt_path: audioPromptFilename,
        exaggeration: exaggeration,
        temperature: temperature,
        cfg_weight: cfgWeight,
        chunk_size: chunkSize,
        reduce_noise: reduceNoise,
        remove_silence: removeSilence,
        speed: speed,
        pitch: pitch,
        seed: seed
      });

      generateButton.disabled = true;
      progressComplete.textContent = "";
      
      // Reset the progress bar
      progress.style.strokeDashoffset = circumference;
      progress.style.animation = 'spin 4s linear infinite';
      progressText.textContent = '0%';

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
      .then(response => response.json())
      .then(data => {
        if (data.success) {
          currentUploadedFile = data.filename;
          startGeneration(data.filename);
        } else {
          alert('Error uploading file: ' + data.error);
          generateButton.disabled = false;
          progressContainer.classList.add('hide');
          progressComplete.classList.add('hide');
        }
      })
      .catch(error => {
        console.error('Upload error:', error);
        alert('Error uploading file: ' + error.message);
        generateButton.disabled = false;
        progressContainer.classList.add('hide');
        progressComplete.classList.add('hide');
      });
    } else {
      // No file, start generation directly
      startGeneration();
    }
  });

  let progress = document.querySelector('#progress');
  let progressText = document.querySelector('#progress-text');
  let radius = progress.r.baseVal.value;
  let circumference = 2 * Math.PI * radius;
  
  progress.style.strokeDasharray = `${circumference} ${circumference}`;
  progress.style.strokeDashoffset = `${circumference}`;
  
  function setProgress(percent) {
    progress.style.animation = 'none';
    const offset = circumference - percent / 100 * circumference;
    progress.style.strokeDashoffset = offset;
    progressText.textContent = `${Math.round(percent)}%`;
  }
  
  // Update the progress bar
  socket.on('generation_progress', function(data) {
    console.log('Generation progress:', data.progress);
    setProgress(data.progress * 100);
  });  

  socket.on('generation_complete', function(data) {
    console.log('Generation completed:', data);
  
    generateButton.disabled = false;
    progressContainer.classList.add('hide');
    progressComplete.textContent = "Generation completed in " + formatTime(data.generation_time) + ".";
    
    // Note: We don't clear the reference audio file here anymore
    // The file and UI elements remain so the user can generate again with the same reference
  
    // Scroll to the top of audio-list
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
    generateButton.disabled = false;
    progressContainer.classList.add('hide');
    progressComplete.textContent = "Error: " + data.error;
    progressComplete.style.color = '#ff4444';
    
    setTimeout(() => {
      progressComplete.textContent = "";
      progressComplete.style.color = 'var(--primary-btn-bg-color)';
    }, 5000);
  });

  let sampleText = "Artificial intelligence is transforming the way we interact with machines, making communication more natural and intuitive. Text-to-speech technology, powered by AI, allows computers to convert written words into spoken language with increasing fluency and realism. This capability is essential for accessibility, virtual assistants, and interactive voice response systems.";

  // Load form state from localStorage
  document.getElementById('text-input').value = localStorage.getItem('textInput') || sampleText;
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
    if (slider.id === 'chunk-size') {
      display.textContent = parseInt(slider.value);
    } else {
      display.textContent = parseFloat(slider.value).toFixed(2);
    }
  });
  
  // Load seed settings
  const savedSeed = localStorage.getItem('seed') || '0';
  document.getElementById('seed').value = savedSeed;
  if (savedSeed === 'custom') {
    customSeedContainer.classList.remove('hide');
    document.getElementById('custom-seed-input').value = localStorage.getItem('customSeed') || '';
  }

  // Load the audio list on page load
  loadAudioList();

  const audioItemTemplate = document.getElementById('audio-item-template').content;
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
            const filename = item.outputFile;
            const textInput = item.textInput;
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
            const audioItem = audioItemTemplate.cloneNode(true);
            audioItem.querySelector('.audio-player').src = 'static/output/' + filename;
            audioItem.querySelector('.filename').textContent = filename;
            audioItem.querySelector('.gen-time').textContent = 'Generation Time: ' + formatTime(genTime);
            audioItem.querySelector('.audio-prompt').textContent = 'Voice: ' + (audioPromptPath === null ? 'Default Voice' : audioPromptPath);
            audioItem.querySelector('.exaggeration').textContent = 'Exaggeration: ' + exaggeration;
            audioItem.querySelector('.temperature').textContent = 'Temperature: ' + temperature;
            audioItem.querySelector('.cfg-weight').textContent = 'CFG Weight: ' + cfgWeight;
            audioItem.querySelector('.chunk-size').textContent = 'Chunk Size: ' + chunkSize;
            audioItem.querySelector('.speed').textContent = 'Speed: ' + speed;
            audioItem.querySelector('.pitch').textContent = 'Pitch: ' + pitch;
            audioItem.querySelector('.reduce-noise').textContent = 'RN: ' + reduceNoise;
            audioItem.querySelector('.remove-silence').textContent = 'RS: ' + removeSilence;
            audioItem.querySelector('.seed').textContent = 'Seed: ' + (seed === 0 ? 'Random' : seed);
            audioItem.querySelector('.text-input').textContent = textInput;

            audioItem.querySelector('.download-button').addEventListener('click', function(event) {
              event.preventDefault();
              downloadFile('static/output/' + filename, filename);
            });

            audioItem.querySelector('.delete-button').addEventListener('click', function(event) {
              event.preventDefault();
              const parentCard = event.target.closest('.card');
            
              // Check if the parentCard has the 'new-card' class
              if (parentCard.classList.contains('new-card')) {
                progressComplete.textContent = "";
              }
            
              parentCard.classList.add('hide');
              deleteAudioFile(filename);
            });

            if (isNewCard) {
              audioItem.querySelector('.card').classList.add('new-card');
              isNewCard = false;
            }

            audioList.appendChild(audioItem);
          }
        }
      })
      .catch(error => {
        if (error.message === 'JSON data file not found') {
          console.log('data.json file does not exist');
        } else {
          console.log('Error loading audio list:', error);
        }
      })
      .finally(() => {
        if (callback) callback();
      });
  }

  function downloadFile(url, filename) {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.target = '_blank';
    link.click();
  }

  function deleteAudioFile(filename) {
    fetch('static/output/' + filename, { method: 'DELETE' })
      .then(function() {
        console.log('File deleted: ', filename);
      })
      .catch(function(error) {
        console.log('Error deleting file: ', error);
      });
  }

  // Handle more info links
  const moreInfoLinks = document.querySelectorAll('.more-info-link');

  moreInfoLinks.forEach(link => {
    link.addEventListener('click', function() {
      const moreInfo = this.previousElementSibling;
      moreInfo.classList.toggle('show');
      this.textContent = moreInfo.classList.contains('show') ? 'Less Info' : this.dataset.defaultText;
    });
  
    link.dataset.defaultText = link.textContent; 
  });

  // Handle select container styling
  var selectContainers = document.querySelectorAll('.select-container');

  selectContainers.forEach(function(container) {
    var select = container.querySelector('select');
    
    select.addEventListener('focus', function() {
      container.classList.add('open');
    });
    
    select.addEventListener('blur', function() {
      container.classList.remove('open');
    });
  });

  function formatTime(seconds) {
    let hours = Math.floor(seconds / 3600);
    let minutes = Math.floor((seconds % 3600) / 60);
    seconds = Math.floor(seconds % 60);

    let timeStr = '';
    
    if (hours > 0) {
      timeStr += `${hours} hours, `;
    }
    
    if (minutes > 0) {
      timeStr += `${minutes} minutes, `;
    }
  
    timeStr += `${seconds} seconds`;
    return timeStr;
  }
});