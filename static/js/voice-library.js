/**
 * Voice Library Management System
 * 
 * Provides voice profile management, multi-voice support, and project management
 * inspired by audiobook production workflows.
 */

class VoiceLibrary {
    constructor() {
        this.voiceProfiles = [];
        this.legacyVoices = [];
        this.currentProject = null;
        this.characters = [];
        this.voiceAssignments = {};
        this.selectedVoice = null;
        this.selectedVoiceType = null;
        
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.loadVoiceLibrary();
    }
    
    bindEvents() {
        // Voice library management
        const createProfileBtn = document.getElementById('create-voice-profile-btn');
        const deleteProfileBtn = document.getElementById('delete-voice-profile-btn');
        const loadProfileBtn = document.getElementById('load-voice-profile-btn');
        const refreshLibraryBtn = document.getElementById('refresh-voice-library-btn');
        
        if (createProfileBtn) {
            createProfileBtn.addEventListener('click', () => this.showCreateProfileModal());
        }
        
        if (deleteProfileBtn) {
            deleteProfileBtn.addEventListener('click', () => this.deleteSelectedProfile());
        }
        
        if (loadProfileBtn) {
            loadProfileBtn.addEventListener('click', () => this.loadSelectedProfile());
        }
        
        // Voice profile dropdown selection
        const voiceProfileSelect = document.getElementById('voice-profile-select');
        if (voiceProfileSelect) {
            voiceProfileSelect.addEventListener('change', (e) => this.onVoiceProfileChange(e));
        }
        
        if (refreshLibraryBtn) {
            refreshLibraryBtn.addEventListener('click', () => this.loadVoiceLibrary());
        }
        
        // Multi-voice functionality
        const parseMultivoiceBtn = document.getElementById('parse-multivoice-btn');
        const createProjectBtn = document.getElementById('create-project-btn');
        
        if (parseMultivoiceBtn) {
            parseMultivoiceBtn.addEventListener('click', () => this.parseMultivoiceText());
        }
        
        if (createProjectBtn) {
            createProjectBtn.addEventListener('click', () => this.createMultivoiceProject());
        }
    }
    
    async loadVoiceLibrary() {
        try {
            const response = await fetch('/get_voice_files');
            const data = await response.json();
            
            if (data.error) {
                DOMUtils.showError(data.error);
                return;
            }
            
            this.voiceProfiles = data.voice_profiles || [];
            this.legacyVoices = data.legacy_voices || [];
            
            console.log('Voice library data loaded:', data);
            console.log('Voice profiles:', this.voiceProfiles);
            console.log('Legacy voices:', this.legacyVoices);
            
            this.updateVoiceLibraryUI();
            this.updateVoiceSelectors();
            
            console.log(`Loaded ${this.voiceProfiles.length} voice profiles and ${this.legacyVoices.length} legacy voices`);
            
        } catch (error) {
            console.error('Error loading voice library:', error);
            DOMUtils.showError('Failed to load voice library');
        }
    }
    
    updateVoiceLibraryUI() {
        const voiceProfilesList = document.getElementById('voice-profiles-list');
        const legacyVoicesList = document.getElementById('legacy-voices-list');
        
        if (voiceProfilesList) {
            voiceProfilesList.innerHTML = '';
            
            if (this.voiceProfiles.length === 0) {
                voiceProfilesList.innerHTML = '<p class="no-voices">No voice profiles found. Create your first voice profile!</p>';
            } else {
                this.voiceProfiles.forEach(profile => {
                    const profileElement = this.createVoiceProfileElement(profile);
                    voiceProfilesList.appendChild(profileElement);
                });
            }
        }
        
        if (legacyVoicesList) {
            legacyVoicesList.innerHTML = '';
            
            if (this.legacyVoices.length === 0) {
                legacyVoicesList.innerHTML = '<p class="no-voices">No legacy voices found.</p>';
            } else {
                this.legacyVoices.forEach(voice => {
                    const voiceElement = this.createLegacyVoiceElement(voice);
                    legacyVoicesList.appendChild(voiceElement);
                });
            }
        }
        
        // Update statistics
        const profileCount = document.getElementById('profile-count');
        const legacyCount = document.getElementById('legacy-count');
        
        if (profileCount) profileCount.textContent = this.voiceProfiles.length;
        if (legacyCount) legacyCount.textContent = this.legacyVoices.length;
    }
    
    updateVoiceSelectors() {
        const voiceSelect = document.getElementById('voice-profile-select');
        const characterVoiceSelects = document.querySelectorAll('.character-voice-select');
        
        if (voiceSelect) {
            voiceSelect.innerHTML = '<option value="">Select a voice profile...</option>';
            
            // Add voice profiles
            this.voiceProfiles.forEach(profile => {
                const option = document.createElement('option');
                option.value = profile.profile_name;
                option.textContent = `${profile.display_name} (Profile)`;
                option.dataset.type = 'profile';
                voiceSelect.appendChild(option);
            });
            
            // Add legacy voices
            this.legacyVoices.forEach(voice => {
                const option = document.createElement('option');
                option.value = voice.filename;
                option.textContent = `${voice.display_name} (Legacy)`;
                option.dataset.type = 'legacy';
                voiceSelect.appendChild(option);
            });
        }
        
        // Update character voice selectors for multi-voice projects
        characterVoiceSelects.forEach(select => {
            const currentValue = select.value;
            select.innerHTML = '<option value="">Select voice for this character...</option>';
            
            this.voiceProfiles.forEach(profile => {
                const option = document.createElement('option');
                option.value = profile.profile_name;
                option.textContent = profile.display_name;
                option.dataset.type = 'profile';
                if (profile.profile_name === currentValue) option.selected = true;
                select.appendChild(option);
            });
            
            this.legacyVoices.forEach(voice => {
                const option = document.createElement('option');
                option.value = voice.filename;
                option.textContent = voice.display_name;
                option.dataset.type = 'legacy';
                if (voice.filename === currentValue) option.selected = true;
                select.appendChild(option);
            });
        });
    }
    
    createVoiceProfileElement(profile) {
        const div = document.createElement('div');
        div.className = 'voice-profile-item';
        div.dataset.profileName = profile.profile_name;
        
        const formatFileSize = (bytes) => {
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            if (bytes === 0) return '0 Bytes';
            const i = parseInt(Math.floor(Math.log(bytes) / Math.log(1024)));
            return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
        };
        
        div.innerHTML = `
            <div class="voice-profile-header">
                <h4>${profile.display_name}</h4>
                <div class="voice-profile-actions">
                    <button class="btn btn-sm btn-primary load-profile-btn" data-profile="${profile.profile_name}">
                        📥 Load
                    </button>
                    <button class="btn btn-sm btn-danger delete-profile-btn" data-profile="${profile.profile_name}">
                        🗑️ Delete
                    </button>
                </div>
            </div>
            <div class="voice-profile-details">
                <p class="voice-description">${profile.description || 'No description'}</p>
                <div class="voice-profile-meta">
                    <span class="voice-meta-item">📁 ${profile.audio_file}</span>
                    <span class="voice-meta-item">📊 ${formatFileSize(profile.size)}</span>
                    <span class="voice-meta-item">🎛️ Ex: ${profile.settings.exaggeration}</span>
                    <span class="voice-meta-item">🌡️ T: ${profile.settings.temperature}</span>
                    <span class="voice-meta-item">⚖️ CFG: ${profile.settings.cfg_weight}</span>
                </div>
            </div>
        `;
        
        // Bind individual profile actions
        const loadBtn = div.querySelector('.load-profile-btn');
        const deleteBtn = div.querySelector('.delete-profile-btn');
        
        loadBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.loadVoiceProfile(profile.profile_name);
        });
        
        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.deleteVoiceProfile(profile.profile_name, profile.display_name);
        });
        
        // Make profile selectable
        div.addEventListener('click', () => {
            document.querySelectorAll('.voice-profile-item').forEach(item => {
                item.classList.remove('selected');
            });
            div.classList.add('selected');
        });
        
        return div;
    }
    
    createLegacyVoiceElement(voice) {
        const div = document.createElement('div');
        div.className = 'legacy-voice-item';
        
        const formatFileSize = (bytes) => {
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            if (bytes === 0) return '0 Bytes';
            const i = parseInt(Math.floor(Math.log(bytes) / Math.log(1024)));
            return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
        };
        
        div.innerHTML = `
            <div class="legacy-voice-header">
                <h5>${voice.display_name}</h5>
                <button class="btn btn-sm btn-secondary use-voice-btn" data-filename="${voice.filename}">
                    🎤 Use
                </button>
            </div>
            <div class="legacy-voice-meta">
                <span class="voice-meta-item">📁 ${voice.filename}</span>
                <span class="voice-meta-item">📊 ${formatFileSize(voice.size)}</span>
            </div>
        `;
        
        const useBtn = div.querySelector('.use-voice-btn');
        useBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.selectLegacyVoice(voice.filename, voice.display_name);
        });
        
        return div;
    }
    
    async loadVoiceProfile(profileName) {
        try {
            const response = await fetch(`/voice_profiles/${profileName}/settings`);
            const data = await response.json();
            
            if (data.error) {
                DOMUtils.showError(data.error);
                return;
            }
            
            const settings = data.settings;
            
            // Apply settings to form controls
            const exaggerationSlider = document.getElementById('exaggeration');
            const temperatureSlider = document.getElementById('temperature');
            const cfgWeightSlider = document.getElementById('cfg-weight');
            
            if (exaggerationSlider) {
                exaggerationSlider.value = settings.exaggeration;
                const exaggerationValue = document.getElementById('exaggeration-value');
                if (exaggerationValue) exaggerationValue.textContent = settings.exaggeration;
            }
            
            if (temperatureSlider) {
                temperatureSlider.value = settings.temperature;
                const temperatureValue = document.getElementById('temperature-value');
                if (temperatureValue) temperatureValue.textContent = settings.temperature;
            }
            
            if (cfgWeightSlider) {
                cfgWeightSlider.value = settings.cfg_weight;
                const cfgWeightValue = document.getElementById('cfg-weight-value');
                if (cfgWeightValue) cfgWeightValue.textContent = settings.cfg_weight;
            }
            
            // Apply voice selection for generation
            this.setSelectedVoice(profileName, settings.display_name, 'profile');
            
            DOMUtils.showFlashMessage(`✅ Loaded voice profile: ${settings.display_name}`, 'success');
            
            // Update the dropdown to reflect the loaded profile
            const voiceSelect = document.getElementById('voice-profile-select');
            if (voiceSelect) {
                voiceSelect.value = profileName;
            }
            
        } catch (error) {
            console.error('Error loading voice profile:', error);
            DOMUtils.showError('Failed to load voice profile');
        }
    }
    
    async deleteVoiceProfile(profileName, displayName) {
        if (!confirm(`Are you sure you want to delete the voice profile "${displayName}"? This action cannot be undone.`)) {
            return;
        }
        
        try {
            const response = await fetch(`/voice_profiles/${profileName}`, {
                method: 'DELETE'
            });
            
            const data = await response.json();
            
            if (data.error) {
                DOMUtils.showError(data.error);
                return;
            }
            
            DOMUtils.showFlashMessage(data.message, 'success');
            this.loadVoiceLibrary(); // Refresh the library
            
        } catch (error) {
            console.error('Error deleting voice profile:', error);
            DOMUtils.showError('Failed to delete voice profile');
        }
    }
    
    selectLegacyVoice(filename, displayName) {
        // Apply legacy voice selection to the system
        this.setSelectedVoice(filename, displayName, 'legacy');
        DOMUtils.showFlashMessage(`✅ Selected legacy voice: ${displayName}`, 'success');
    }
    
    async parseMultivoiceText() {
        const textInput = document.getElementById('text-input');
        const multivoiceContainer = document.getElementById('multivoice-container');
        
        if (!textInput || !textInput.value.trim()) {
            DOMUtils.showError('Please enter text to parse for multi-voice characters');
            return;
        }
        
        try {
            const response = await fetch('/parse_multivoice_text', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text_content: textInput.value })
            });
            
            const data = await response.json();
            
            if (data.error) {
                DOMUtils.showError(data.error);
                return;
            }
            
            if (!data.is_multivoice) {
                DOMUtils.showFlashMessage(data.message, 'info');
                if (multivoiceContainer) {
                    multivoiceContainer.style.display = 'none';
                }
                return;
            }
            
            this.characters = data.characters;
            this.displayCharacterAssignments();
            
            if (multivoiceContainer) {
                multivoiceContainer.style.display = 'block';
            }
            
            DOMUtils.showFlashMessage(data.message, 'success');
            
        } catch (error) {
            console.error('Error parsing multi-voice text:', error);
            DOMUtils.showError('Failed to parse multi-voice text');
        }
    }
    
    displayCharacterAssignments() {
        const characterAssignments = document.getElementById('character-assignments');
        
        if (!characterAssignments) return;
        
        characterAssignments.innerHTML = '';
        
        this.characters.forEach(character => {
            const assignmentDiv = document.createElement('div');
            assignmentDiv.className = 'character-assignment';
            
            assignmentDiv.innerHTML = `
                <div class="character-assignment-header">
                    <h4>🎭 ${character}</h4>
                </div>
                <div class="character-assignment-controls">
                    <label for="voice-${character}">Voice:</label>
                    <select id="voice-${character}" class="character-voice-select form-control" data-character="${character}">
                        <option value="">Select voice for ${character}...</option>
                    </select>
                </div>
            `;
            
            characterAssignments.appendChild(assignmentDiv);
        });
        
        // Update voice selectors with current voice library
        this.updateVoiceSelectors();
        
        // Bind change events to track voice assignments
        document.querySelectorAll('.character-voice-select').forEach(select => {
            select.addEventListener('change', (e) => {
                const character = e.target.dataset.character;
                const voiceValue = e.target.value;
                
                if (voiceValue) {
                    this.voiceAssignments[character] = voiceValue;
                } else {
                    delete this.voiceAssignments[character];
                }
                
                this.updateCreateProjectButton();
            });
        });
    }
    
    updateCreateProjectButton() {
        const createProjectBtn = document.getElementById('create-project-btn');
        
        if (!createProjectBtn) return;
        
        const allAssigned = this.characters.every(character => 
            this.voiceAssignments[character]
        );
        
        createProjectBtn.disabled = !allAssigned;
        
        if (allAssigned) {
            createProjectBtn.textContent = `🎬 Create Project (${this.characters.length} characters)`;
        } else {
            const remaining = this.characters.length - Object.keys(this.voiceAssignments).length;
            createProjectBtn.textContent = `🎬 Create Project (${remaining} assignments needed)`;
        }
    }
    
    async showCreateProfileModal() {
        // Check if there's an uploaded audio file
        const audioPrompt = document.getElementById('audio-prompt');
        if (!audioPrompt || !audioPrompt.files[0]) {
            DOMUtils.showError('Please upload an audio file first');
            return;
        }
        
        const profileName = prompt('Enter voice profile name:');
        if (!profileName) return;
        
        const displayName = prompt('Enter display name for this voice:');
        if (!displayName) return;
        
        const description = prompt('Enter description (optional):') || '';
        
        try {
            // First upload the audio file
            const uploadedFilename = await this.uploadAudioFileForProfile(audioPrompt.files[0]);
            if (!uploadedFilename) {
                DOMUtils.showError('Failed to upload audio file');
                return;
            }
            
            // Then create the voice profile
            await this.createVoiceProfile(profileName, displayName, description, uploadedFilename);
            
        } catch (error) {
            console.error('Error in voice profile creation:', error);
            DOMUtils.showError('Failed to create voice profile');
        }
    }
    
    async uploadAudioFileForProfile(audioFile) {
        try {
            const formData = new FormData();
            formData.append('audio_file', audioFile);
            
            DOMUtils.showFlashMessage("📁 Uploading audio file for profile...", 'info');
            
            const response = await fetch('/upload_audio', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            if (data.success) {
                DOMUtils.showFlashMessage(`📁 Audio uploaded: ${audioFile.name}`, 'success');
                return data.filename;
            } else {
                throw new Error(data.error || 'Audio upload failed');
            }
            
        } catch (error) {
            console.error('Error uploading audio file:', error);
            DOMUtils.showError(`Audio upload failed: ${error.message}`);
            return null;
        }
    }
    
    // Missing method implementations
    loadSelectedProfile() {
        const voiceSelect = document.getElementById('voice-profile-select');
        if (!voiceSelect || !voiceSelect.value) {
            DOMUtils.showError('Please select a voice profile to load');
            return;
        }
        
        const selectedValue = voiceSelect.value;
        const selectedOption = voiceSelect.options[voiceSelect.selectedIndex];
        const voiceType = selectedOption.dataset.type;
        
        if (voiceType === 'profile') {
            this.loadVoiceProfile(selectedValue);
        } else if (voiceType === 'legacy') {
            const legacyVoice = this.legacyVoices.find(v => v.filename === selectedValue);
            if (legacyVoice) {
                this.selectLegacyVoice(selectedValue, legacyVoice.display_name);
            }
        }
    }
    
    deleteSelectedProfile() {
        const voiceSelect = document.getElementById('voice-profile-select');
        if (!voiceSelect || !voiceSelect.value) {
            DOMUtils.showError('Please select a voice profile to delete');
            return;
        }
        
        const selectedValue = voiceSelect.value;
        const selectedOption = voiceSelect.options[voiceSelect.selectedIndex];
        const voiceType = selectedOption.dataset.type;
        
        if (voiceType === 'profile') {
            const profile = this.voiceProfiles.find(p => p.profile_name === selectedValue);
            if (profile) {
                this.deleteVoiceProfile(selectedValue, profile.display_name);
            }
        } else {
            DOMUtils.showError('Cannot delete legacy voices');
        }
    }
    
    onVoiceProfileChange(event) {
        const selectedValue = event.target.value;
        const selectedOption = event.target.options[event.target.selectedIndex];
        
        if (!selectedValue) {
            this.clearSelectedVoice();
            return;
        }
        
        const voiceType = selectedOption.dataset.type;
        const displayName = selectedOption.textContent;
        
        this.setSelectedVoice(selectedValue, displayName, voiceType);
    }
    
    setSelectedVoice(voiceValue, displayName, voiceType) {
        this.selectedVoice = voiceValue;
        this.selectedVoiceType = voiceType;
        
        // Update the main application's voice selection
        if (window.VoiceManager) {
            window.VoiceManager.setSelectedVoice(voiceValue, displayName, voiceType);
        }
        
        console.log(`Selected voice: ${displayName} (${voiceType})`);
    }
    
    clearSelectedVoice() {
        this.selectedVoice = null;
        this.selectedVoiceType = null;
        
        if (window.VoiceManager) {
            window.VoiceManager.clearSelectedVoice();
        }
    }
    
    getCurrentSelectedVoice() {
        return {
            voice: this.selectedVoice,
            type: this.selectedVoiceType
        };
    }
    
    async createVoiceProfile(profileName, displayName, description, audioFilename) {
        try {
            // Get current TTS settings
            const exaggeration = parseFloat(document.getElementById('exaggeration')?.value || '0.5');
            const temperature = parseFloat(document.getElementById('temperature')?.value || '0.3');
            const cfgWeight = parseFloat(document.getElementById('cfg-weight')?.value || '0.7');
            
            const response = await fetch('/voice_profiles', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    profile_name: profileName,
                    display_name: displayName,
                    description: description,
                    audio_filename: audioFilename,
                    exaggeration: exaggeration,
                    temperature: temperature,
                    cfg_weight: cfgWeight
                })
            });
            
            const data = await response.json();
            
            console.log('Voice profile creation response:', data);
            
            if (data.error) {
                DOMUtils.showError(data.error);
                return;
            }
            
            DOMUtils.showFlashMessage(data.message, 'success');
            console.log('Voice profile created successfully:', data.profile_name);
            this.loadVoiceLibrary(); // Refresh the library
            
            // Clear the uploaded file and update UI
            const audioPrompt = document.getElementById('audio-prompt');
            const fileNameDisplay = document.getElementById('file-name-display');
            const clearAudioButton = document.getElementById('clear-audio-button');
            
            if (audioPrompt) audioPrompt.value = '';
            if (fileNameDisplay) fileNameDisplay.textContent = 'No file selected';
            if (clearAudioButton) clearAudioButton.classList.add('hide');
            
        } catch (error) {
            console.error('Error creating voice profile:', error);
            DOMUtils.showError('Failed to create voice profile');
        }
    }
    
    async createMultivoiceProject() {
        const textInput = document.getElementById('text-input');
        
        if (!textInput || !textInput.value.trim()) {
            DOMUtils.showError('Please enter text for the multi-voice project');
            return;
        }
        
        if (this.characters.length === 0) {
            DOMUtils.showError('No characters found. Please analyze text first.');
            return;
        }
        
        // Validate all characters have voice assignments
        const missingAssignments = this.characters.filter(character => 
            !this.voiceAssignments[character]
        );
        
        if (missingAssignments.length > 0) {
            DOMUtils.showError(`Please assign voices to all characters. Missing: ${missingAssignments.join(', ')}`);
            return;
        }
        
        const projectName = prompt('Enter a name for this multi-voice project:');
        if (!projectName) return;
        
        try {
            const response = await fetch('/create_multivoice_project', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    project_name: projectName,
                    text_content: textInput.value,
                    voice_assignments: this.voiceAssignments
                })
            });
            
            const data = await response.json();
            
            if (data.error) {
                DOMUtils.showError(data.error);
                return;
            }
            
            DOMUtils.showFlashMessage(data.message, 'success');
            console.log('Multi-voice project created:', data);
            
            // Reset the UI
            this.resetMultivoiceUI();
            
        } catch (error) {
            console.error('Error creating multi-voice project:', error);
            DOMUtils.showError('Failed to create multi-voice project');
        }
    }
    
    resetMultivoiceUI() {
        const multivoiceContainer = document.getElementById('multivoice-container');
        const characterAssignments = document.getElementById('character-assignments');
        const createProjectBtn = document.getElementById('create-project-btn');
        
        if (multivoiceContainer) {
            multivoiceContainer.style.display = 'none';
        }
        
        if (characterAssignments) {
            characterAssignments.innerHTML = '';
        }
        
        if (createProjectBtn) {
            createProjectBtn.disabled = true;
            createProjectBtn.textContent = '🎬 Create Multi-Voice Project';
        }
        
        // Reset internal state
        this.characters = [];
        this.voiceAssignments = {};
    }
}

// Initialize voice library when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    if (typeof window.voiceLibrary === 'undefined') {
        window.voiceLibrary = new VoiceLibrary();
    }
});