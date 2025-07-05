/**
 * Audio player and management utilities for Chatterbox Web UI
 */

let isNewCard = false;

// Load the audio list from server
function loadAudioList(callback) {
  const audioList = document.getElementById('audio-list');
  
  // Clear the existing audio list
  audioList.innerHTML = '';

  // Update delete all button visibility
  window.DOMUtils.updateDeleteAllButton();

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
      window.DOMUtils.updateDeleteAllButton();
    })
    .catch(error => {
      if (error.message === 'JSON data file not found') {
        console.log('data.json file does not exist');
      } else {
        console.error('Error loading audio list:', error);
      }
      // Update delete all button even on error
      window.DOMUtils.updateDeleteAllButton();
    })
    .finally(() => {
      if (callback) callback();
    });
}

function createAudioCard(item, key) {
  const audioItemTemplate = document.getElementById('audio-item-template');
  const audioList = document.getElementById('audio-list');
  
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
  setTextContent('.gen-time', 'Generation Time: ' + window.DOMUtils.formatTime(genTime));
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
      window.DOMUtils.downloadFile('static/output/' + filename, filename);
    });
  }

  const deleteButton = audioItem.querySelector('.delete-button');
  if (deleteButton) {
    deleteButton.addEventListener('click', function(event) {
      event.preventDefault();
      const parentCard = event.target.closest('.card');
      
      // No confirmation popup - just delete immediately
      if (parentCard && parentCard.classList.contains('new-card')) {
        const progressComplete = document.querySelector('.progress-complete');
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
      console.log('File deleted:', filename);
      window.DOMUtils.showFlashMessage(`🗑️ Deleted: ${filename}`, 'success');
      
      // Remove the card from the DOM
      if (cardElement) {
        setTimeout(() => {
          cardElement.remove();
          window.DOMUtils.updateDeleteAllButton();
        }, 300);
      }
    })
    .catch(error => {
      console.error('Error deleting file:', error);
      window.DOMUtils.showFlashMessage(`❌ Failed to delete: ${filename}`, 'error');
      
      // Restore card opacity on error
      if (cardElement) {
        cardElement.style.opacity = '1';
      }
    });
}

// Delete all audio files function
function deleteAllAudioFiles() {
  const deleteAllButton = document.getElementById('delete-all-button');
  const progressComplete = document.querySelector('.progress-complete');
  const audioCards = document.querySelectorAll('.audio-item');
  const totalFiles = audioCards.length;

  if (totalFiles === 0) {
    window.DOMUtils.showFlashMessage('No audio files to delete', 'info');
    return;
  }

  // Disable the delete all button to prevent multiple clicks
  if (deleteAllButton) {
    deleteAllButton.disabled = true;
    deleteAllButton.textContent = 'Deleting...';
  }

  // Show progress in flash message
  window.DOMUtils.showFlashMessage(`🗑️ Deleting ${totalFiles} files...`, 'info');

  // Clear progress complete message if it was showing
  if (progressComplete.classList.contains('hide') === false) {
    progressComplete.textContent = "";
  }

  // Fade out all cards immediately for visual feedback
  audioCards.forEach((card) => {
    card.style.opacity = '0.3';
    card.style.transition = 'opacity 0.3s ease';
  });

  // Use the server endpoint for bulk deletion
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
    console.error('Error in delete all operation:', error);
    window.DOMUtils.showFlashMessage(`❌ Failed to delete files: ${error.message}`, 'error');
    
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
  const deleteAllButton = document.getElementById('delete-all-button');
  
  // Re-enable delete all button
  if (deleteAllButton) {
    deleteAllButton.disabled = false;
    deleteAllButton.textContent = 'Delete All';
  }

  // Show result message
  if (failedCount === 0) {
    window.DOMUtils.showFlashMessage(`✅ Successfully deleted all ${deletedCount} audio files`, 'success');
  } else if (deletedCount === 0) {
    window.DOMUtils.showFlashMessage(`❌ Failed to delete any files (${failedCount} errors)`, 'error');
  } else {
    window.DOMUtils.showFlashMessage(`⚠️ Deleted ${deletedCount} files, ${failedCount} failed`, 'warning');
  }

  // Reload the audio list to reflect changes
  setTimeout(() => {
    loadAudioList();
  }, 1000);
}

// Export functions for use in other modules
window.AudioPlayer = {
  loadAudioList,
  createAudioCard,
  deleteAudioFile,
  deleteAllAudioFiles,
  completeDeleteAll,
  setIsNewCard: (value) => { isNewCard = value; }
};