/**
 * DOM utility functions for Chatterbox Web UI
 */

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

// Message system - show messages in Generation Status window
function showFlashMessage(message, type = 'success') {
  const progressContainer = document.querySelector('.progress-container');
  
  // Show the Generation Status window
  progressContainer.classList.remove('hide');
  
  // Add message to status console with appropriate icon
  let icon = '';
  switch(type) {
    case 'success': icon = '✅'; break;
    case 'error': icon = '❌'; break;
    case 'warning': icon = '⚠️'; break;
    case 'info': icon = 'ℹ️'; break;
    default: icon = '💬'; break;
  }
  
  addStatusMessage(`${icon} ${message}`);
}

function showError(message) {
  const progressContainer = document.querySelector('.progress-container');
  const progressComplete = document.querySelector('.progress-complete');
  
  showFlashMessage(message, 'error');
  progressContainer.classList.add('hide');
  progressComplete.textContent = "";
}

function addStatusMessage(message) {
  const statusConsole = document.getElementById('status-console');
  
  if (statusConsole) {
    const timestamp = new Date().toLocaleTimeString();
    const messageElement = document.createElement('div');
    messageElement.className = 'status-message';
    messageElement.innerHTML = `<span class="timestamp">[${timestamp}]</span> ${message}`;
    statusConsole.appendChild(messageElement);
    
    // Auto-scroll to bottom
    statusConsole.scrollTop = statusConsole.scrollHeight;
    
    // Keep only last 50 messages to prevent memory issues
    const messages = statusConsole.querySelectorAll('.status-message');
    if (messages.length > 50) {
      messages[0].remove();
    }
  }
}

function clearStatusConsole() {
  const statusConsole = document.getElementById('status-console');
  if (statusConsole) {
    statusConsole.innerHTML = '';
  }
}

// Handle slider value updates
function initializeSliders() {
  const sliders = [
    { slider: document.getElementById('exaggeration'), display: document.getElementById('exaggeration-value') },
    { slider: document.getElementById('temperature'), display: document.getElementById('temperature-value') },
    { slider: document.getElementById('cfg-weight'), display: document.getElementById('cfg-weight-value') },
    { slider: document.getElementById('chunk-size'), display: document.getElementById('chunk-size-value') },
    { slider: document.getElementById('speed'), display: document.getElementById('speed-value') },
    { slider: document.getElementById('pitch'), display: document.getElementById('pitch-value') },
    { slider: document.getElementById('num-candidates'), display: document.getElementById('num-candidates-value') },
    { slider: document.getElementById('max-attempts'), display: document.getElementById('max-attempts-value') },
    { slider: document.getElementById('parallel-workers'), display: document.getElementById('parallel-workers-value') },
    { slider: document.getElementById('validation-threshold'), display: document.getElementById('validation-threshold-value') },
    { slider: document.getElementById('ae-threshold'), display: document.getElementById('ae-threshold-value') },
    { slider: document.getElementById('ae-margin'), display: document.getElementById('ae-margin-value') },
    { slider: document.getElementById('normalize-level'), display: document.getElementById('normalize-level-value') },
    { slider: document.getElementById('normalize-tp'), display: document.getElementById('normalize-tp-value') },
    { slider: document.getElementById('normalize-lra'), display: document.getElementById('normalize-lra-value') },
    { slider: document.getElementById('noise-strength'), display: document.getElementById('noise-strength-value') },
    { slider: document.getElementById('voice-similarity-threshold'), display: document.getElementById('voice-similarity-threshold-value') }
  ];

  sliders.forEach(({ slider, display }) => {
    if (slider && display) {
      slider.addEventListener('input', function() {
        if (slider.id === 'chunk-size' || slider.id === 'num-candidates' || 
            slider.id === 'max-attempts' || slider.id === 'parallel-workers' || slider.id === 'normalize-level' || 
            slider.id === 'normalize-tp' || slider.id === 'normalize-lra') {
          display.textContent = parseInt(this.value);
        } else {
          display.textContent = parseFloat(this.value).toFixed(2);
        }
      });
      
      // Initialize display value
      if (slider.id === 'chunk-size' || slider.id === 'num-candidates' || 
          slider.id === 'max-attempts' || slider.id === 'parallel-workers' || slider.id === 'normalize-level' || 
          slider.id === 'normalize-tp' || slider.id === 'normalize-lra') {
        display.textContent = parseInt(slider.value);
      } else {
        display.textContent = parseFloat(slider.value).toFixed(2);
      }
    }
  });
  
  return sliders;
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

// Update delete all button visibility and text
function updateDeleteAllButton() {
  const deleteAllButton = document.getElementById('delete-all-button');
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

// Format time helper
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

// Download file helper
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
    showFlashMessage('❌ Error downloading file', 'error');
  }
}

// Export functions for use in other modules
window.DOMUtils = {
  debounce,
  showFlashMessage,
  showError,
  addStatusMessage,
  clearStatusConsole,
  initializeSliders,
  initializeMoreInfoLinks,
  initializeSelectContainers,
  updateDeleteAllButton,
  formatTime,
  downloadFile
};