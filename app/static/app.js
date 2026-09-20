document.addEventListener('DOMContentLoaded', () => {
    
    // Status check
    fetch('/health')
        .then(res => res.json())
        .then(data => {
            const statusIndicator = document.querySelector('.status-indicator');
            const statusText = document.getElementById('system-status');
            
            if(data.status === 'ok' && data.model_loaded) {
                statusIndicator.className = 'status-indicator online';
                statusText.innerHTML = `<span class="status-indicator online"></span> Ready (${data.device})`;
            } else {
                statusIndicator.className = 'status-indicator offline';
                statusText.innerHTML = `<span class="status-indicator offline"></span> Model Offline`;
            }
        })
        .catch(err => {
            console.error('Health check failed', err);
            const statusText = document.getElementById('system-status');
            statusText.innerHTML = `<span class="status-indicator offline"></span> Backend Unreachable`;
        });

    // File Drag and Drop
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadPlaceholder = document.getElementById('upload-placeholder');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const recognizeBtn = document.getElementById('recognize-btn');
    const clearBtn = document.getElementById('clear-btn');
    const resultsSection = document.getElementById('results-section');
    
    let selectedFile = null;

    dropZone.addEventListener('click', (e) => {
        if(e.target !== removeBtn && !removeBtn.contains(e.target)) {
            fileInput.click();
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handleFile(fileInput.files[0]);
        }
    });

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearForm();
    });
    
    clearBtn.addEventListener('click', clearForm);

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file (JPG, PNG).');
            return;
        }
        
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadPlaceholder.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            recognizeBtn.disabled = false;
            clearBtn.disabled = false;
            resetStepper();
            resultsSection.classList.add('hidden');
            document.getElementById('limitation-panel').classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    function clearForm() {
        selectedFile = null;
        fileInput.value = '';
        uploadPlaceholder.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        recognizeBtn.disabled = true;
        clearBtn.disabled = true;
        resultsSection.classList.add('hidden');
        document.getElementById('limitation-panel').classList.add('hidden');
        resetStepper();
    }

    // Pipeline Stepper Logic
    const steps = [
        document.getElementById('step-upload'),
        document.getElementById('step-lines'),
        document.getElementById('step-words'),
        document.getElementById('step-ocr'),
        document.getElementById('step-correction'),
        document.getElementById('step-done')
    ];

    function resetStepper() {
        steps.forEach(step => {
            step.classList.remove('active', 'completed');
        });
        steps[0].classList.add('completed');
    }

    // Fake progress animation for UX before showing real result
    function animatePipeline(callback) {
        resetStepper();
        let currentStep = 1; // Start at line segmentation
        
        const interval = setInterval(() => {
            if (currentStep > 1) {
                steps[currentStep-1].classList.remove('active');
                steps[currentStep-1].classList.add('completed');
            }
            
            if (currentStep < steps.length) {
                steps[currentStep].classList.add('active');
                currentStep++;
            } else {
                clearInterval(interval);
                callback();
            }
        }, 400); // 400ms per step animation
    }

    // Form Submission
    recognizeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        recognizeBtn.disabled = true;
        recognizeBtn.textContent = 'Processing...';

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            // Start pipeline UI animation simultaneously with network request
            let apiResponse = null;
            let apiError = null;
            
            const fetchPromise = fetch('/api/ocr', {
                method: 'POST',
                body: formData
            }).then(res => {
                if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
                return res.json();
            }).catch(err => {
                apiError = err;
            });

            animatePipeline(async () => {
                await fetchPromise; // Ensure fetch finishes if it hasn't already
                
                recognizeBtn.disabled = false;
                recognizeBtn.textContent = 'Run OCR';

                if (apiError) {
                    alert('OCR processing failed. See console for details.');
                    console.error(apiError);
                    resetStepper();
                    return;
                }
                
                apiResponse = await fetchPromise;
                displayResults(apiResponse);
            });

        } catch (error) {
            console.error('Error processing image:', error);
            alert('An unexpected error occurred.');
            recognizeBtn.disabled = false;
            recognizeBtn.textContent = 'Run OCR';
            resetStepper();
        }
    });

    function displayResults(data) {
        resultsSection.classList.remove('hidden');
        
        // Metrics
        document.getElementById('lines-badge').textContent = `${data.lines_detected} Lines`;
        document.getElementById('words-badge').textContent = `${data.words_detected} Words`;
        
        if (data.confidence !== undefined) {
            document.getElementById('confidence-badge').textContent = `Recognition confidence: ${(data.confidence * 100).toFixed(1)}%`;
        }
        
        // Show limitation note always when results are present as a general disclaimer
        document.getElementById('limitation-panel').classList.remove('hidden');
        
        // Texts
        document.getElementById('raw-text').textContent = data.raw_text || '(Raw text not provided by backend)';
        document.getElementById('corrected-text').textContent = data.text || '';
        
        // Corrections Accordion
        const correctionsList = document.getElementById('corrections-list');
        const correctionsCount = document.getElementById('corrections-count');
        
        correctionsList.innerHTML = '';
        
        if (data.corrections_list && data.corrections_list.length > 0) {
            correctionsCount.textContent = data.corrections_list.length;
            data.corrections_list.forEach(c => {
                const li = document.createElement('li');
                li.innerHTML = `<span class="token-raw">${c.raw}</span> <span class="token-arrow">→</span> <span class="token-corrected">${c.corrected}</span>`;
                correctionsList.appendChild(li);
            });
        } else {
            correctionsCount.textContent = '0';
            correctionsList.innerHTML = '<li><span class="token-arrow">No corrections were necessary for this image.</span></li>';
        }
    }

    // Accordion Logic
    const accordionBtn = document.getElementById('corrections-accordion');
    const accordionPanel = document.getElementById('corrections-panel');
    
    accordionBtn.addEventListener('click', function() {
        this.classList.toggle('active');
        if (this.classList.contains('active')) {
            accordionPanel.classList.add('show');
        } else {
            accordionPanel.classList.remove('show');
        }
    });
});
