/**
 * WebSocket event handlers for Chatterbox Web UI
 */

let isGenerating = false;
let progress, progressText, radius, circumference;

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

function setupWebSocketHandlers(socket) {
  
  // Socket event handlers
  socket.on('generation_progress', function(data) {
    setProgress(data.progress * 100);
  });

  socket.on('generation_status', function(data) {
    window.DOMUtils.addStatusMessage(data.message);
  });  

  socket.on('generation_complete', function(data) {
  
    isGenerating = false;
    updateGenerateButton(false);
    hideProgressContainer();
    
    // Show flash notification for completed generation
    const generationTime = window.DOMUtils.formatTime(data.generation_time);
    window.DOMUtils.showFlashMessage(`🎉 Audio generation completed successfully in ${generationTime}!`, 'success');

    // Highlight new card
    window.AudioPlayer.setIsNewCard(true);
  
    // Load the updated audio list
    window.AudioPlayer.loadAudioList(() => {
      window.AudioPlayer.setIsNewCard(false);
      // Scroll to the new audio card after it's loaded
      setTimeout(() => {
        const newCard = document.querySelector('.new-card');
        if (newCard) {
          newCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }, 100);
    });
  });

  socket.on('file_generation_complete', function(data) {
    // Handle individual file completion in batch processing
    console.log('Individual file completed:', data.source_file);
    
    // Refresh the audio list to show the newly completed file
    window.AudioPlayer.setIsNewCard(true);
    
    // Add a small delay to ensure file is written before refreshing
    setTimeout(() => {
      window.AudioPlayer.loadAudioList(() => {
        window.AudioPlayer.setIsNewCard(false);
        
        // Scroll to the newest audio card
        setTimeout(() => {
          const audioCards = document.querySelectorAll('.audio-item');
          if (audioCards.length > 0) {
            // Scroll to the first (newest) card
            audioCards[0].scrollIntoView({ behavior: 'smooth', block: 'center' });
          }
        }, 100);
      });
    }, 200); // Shorter delay since it's just one file
    
    // Show progress message
    window.DOMUtils.showFlashMessage(
      `✅ File ${data.file_number}/${data.total_files} completed: ${data.source_file}`, 
      'success'
    );
  });

  socket.on('batch_generation_complete', function(data) {
    
    isGenerating = false;
    
    // Also ensure the button is properly reset
    const generateButton = document.getElementById('generate-button');
    if (generateButton) {
      generateButton.disabled = false;
      generateButton.textContent = 'Generate';
      generateButton.classList.remove('generating');
    }
    
    updateGenerateButton(false);
    hideProgressContainer();
    
    // Show batch completion notification
    const totalFiles = data.total_files;
    const successfulFiles = data.successful_files;
    const failedFiles = data.failed_files;
    
    let message = `🎉 Batch processing completed! ${successfulFiles}/${totalFiles} files processed successfully.`;
    if (failedFiles && failedFiles.length > 0) {
      message += ` Failed files: ${failedFiles.join(', ')}`;
    }
    
    window.DOMUtils.showFlashMessage(message, successfulFiles > 0 ? 'success' : 'warning');

    // Highlight new cards for batch processing
    window.AudioPlayer.setIsNewCard(true);
  
    // Add a small delay to ensure files are written before refreshing
    setTimeout(() => {
      // Load the updated audio list to show all new generated files
      window.AudioPlayer.loadAudioList(() => {
        window.AudioPlayer.setIsNewCard(false);
        // Scroll to the first new audio card after they're loaded
        setTimeout(() => {
          const newCards = document.querySelectorAll('.new-card');
          if (newCards.length > 0) {
            newCards[0].scrollIntoView({ behavior: 'smooth', block: 'center' });
          }
        }, 100);
      });
    }, 500); // Wait 500ms for files to be fully written
  });

  socket.on('error', function(data) {
    isGenerating = false;
    updateGenerateButton(false);
    hideProgressContainer();
    window.DOMUtils.showFlashMessage(`❌ Generation failed: ${data.error}`, 'error');
  });

  socket.on('generation_cancelled', function(data) {
    isGenerating = false;
    updateGenerateButton(false);
    hideProgressContainer();
    window.DOMUtils.showFlashMessage('🛑 Generation cancelled successfully', 'info');
    window.DOMUtils.addStatusMessage('🛑 Generation cancelled by user');
  });

  socket.on('connect_error', function(error) {
    window.DOMUtils.showFlashMessage('🔌 Connection error. Please refresh the page.', 'error');
  });

  socket.on('disconnect', function(reason) {
    if (isGenerating) {
      window.DOMUtils.showFlashMessage('⚠️ Connection lost during generation. Please try again.', 'warning');
      isGenerating = false;
      updateGenerateButton(false);
      hideProgressContainer();
    }
  });
}

function updateGenerateButton(generating) {
  const generateButton = document.getElementById('generate-button');
  const cancelButton = document.getElementById('cancel-button');
  
  if (generateButton) {
    generateButton.disabled = generating;
    generateButton.textContent = generating ? 'Generating...' : 'Generate';
    if (generating) {
      generateButton.classList.add('generating');
    } else {
      generateButton.classList.remove('generating');
    }
  }
  
  // Show/hide cancel button based on generation state
  if (cancelButton) {
    if (generating) {
      cancelButton.classList.remove('hide');
    } else {
      cancelButton.classList.add('hide');
    }
  }
}

function showProgressContainer() {
  const progressContainer = document.querySelector('.progress-container');
  const progressComplete = document.querySelector('.progress-complete');
  
  if (progressContainer) {
    progressContainer.classList.remove('hide');
  }
  if (progressComplete) {
    progressComplete.classList.remove('hide');
    progressComplete.textContent = "";
    progressComplete.style.color = 'var(--primary-btn-bg-color)';
  }
}

function hideProgressContainer() {
  const progressContainer = document.querySelector('.progress-container');
  const progressComplete = document.querySelector('.progress-complete');
  
  if (progressContainer) {
    progressContainer.classList.add('hide');
  }
  if (progressComplete) {
    progressComplete.textContent = "";
  }
}

function startGeneration(socket, formData, audioPromptFilename = null) {
  isGenerating = true;
  
  const generationData = {
    text_input: formData.textInput,
    audio_prompt_path: audioPromptFilename,
    export_formats: formData.exportFormats,
    num_generations: formData.numGenerations,
    num_candidates: formData.numCandidates,
    max_attempts: formData.maxAttempts,
    whisper_model: formData.whisperModel,
    use_faster_whisper: formData.useFasterWhisper,
    bypass_whisper: formData.bypassWhisper,
    use_longest_transcript: formData.useLongestTranscript,
    enable_parallel: formData.enableParallel,
    parallel_workers: formData.parallelWorkers,
    enable_batching: formData.enableBatching,
    smart_batch_short: formData.smartBatchShort,
    to_lowercase: formData.toLowercase,
    normalize_spacing: formData.normalizeSpacing,
    fix_dot_letters: formData.fixDotLetters,
    remove_reference_numbers: formData.removeReferenceNumbers,
    normalize_audio: formData.normalizeAudio,
    normalize_method: formData.normalizeMethod,
    normalize_level: formData.normalizeLevel,
    normalize_tp: formData.normalizeTp,
    normalize_lra: formData.normalizeLra,
    sound_words: formData.soundWords,
    exaggeration: formData.exaggeration,
    temperature: formData.temperature,
    cfg_weight: formData.cfgWeight,
    chunk_size: formData.chunkSize,
    remove_silence: formData.removeSilence,
    speed: formData.speed,
    pitch: formData.pitch,
    seed: formData.seed
  };

  socket.emit('start_generation', generationData);

  updateGenerateButton(true);
  
  // Clear status console and show initial message
  window.DOMUtils.clearStatusConsole();
  window.DOMUtils.addStatusMessage("🎙️ Initializing audio generation...");
  
  // Show the progress container
  showProgressContainer();
}

function handleFileUpload(socket, formData) {
  const audioPromptFile = formData.audioPromptFile;
  
  if (audioPromptFile) {
    const uploadFormData = new FormData();
    uploadFormData.append('audio_file', audioPromptFile);

    // Show uploading status in console
    window.DOMUtils.clearStatusConsole();
    window.DOMUtils.addStatusMessage("📁 Uploading reference audio...");
    showProgressContainer();

    fetch('/upload_audio', {
      method: 'POST',
      body: uploadFormData
    })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      if (data.success) {
        window.DOMUtils.showFlashMessage(`📁 Reference audio uploaded: ${audioPromptFile.name}`, 'success');
        startGeneration(socket, formData, data.filename);
      } else {
        throw new Error(data.error || 'Upload failed');
      }
    })
    .catch(error => {
      window.DOMUtils.showFlashMessage(`Upload failed: ${error.message}`, 'error');
      isGenerating = false;
      updateGenerateButton(false);
      hideProgressContainer();
    });
  } else {
    // No file, start generation directly
    startGeneration(socket, formData);
  }
}

// Export functions and state for use in other modules
window.WebSocketHandlers = {
  initializeProgressBar,
  setProgress,
  setupWebSocketHandlers,
  updateGenerateButton,
  showProgressContainer,
  hideProgressContainer,
  startGeneration,
  handleFileUpload,
  getIsGenerating: () => isGenerating,
  setIsGenerating: (value) => { isGenerating = value; }
};