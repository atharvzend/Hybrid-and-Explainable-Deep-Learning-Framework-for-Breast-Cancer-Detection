// Global variables
let uploadedFile = null;
let currentPrediction = null;

// DOM Elements
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const uploadSection = document.getElementById('uploadSection');
const resultsSection = document.getElementById('resultsSection');
const loadingOverlay = document.getElementById('loadingOverlay');
const originalImage = document.getElementById('originalImage');
const predictionClass = document.getElementById('predictionClass');
const predictionDescription = document.getElementById('predictionDescription');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceBadge = document.getElementById('confidenceBadge');
const predictionIcon = document.getElementById('predictionIcon');
const probabilityBars = document.getElementById('probabilityBars');
const explainBtn = document.getElementById('explainBtn');
const explainabilitySection = document.getElementById('explainabilitySection');
const gradcamContainer = document.getElementById('gradcamContainer');
const gradcamImage = document.getElementById('gradcamImage');
const limeContainer = document.getElementById('limeContainer');
const limeImage = document.getElementById('limeImage');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
});

// Setup Event Listeners
function setupEventListeners() {
    // Dropzone click
    dropzone.addEventListener('click', () => fileInput.click());

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    dropzone.addEventListener('dragover', handleDragOver);
    dropzone.addEventListener('dragleave', handleDragLeave);
    dropzone.addEventListener('drop', handleDrop);

    // Analyze button
    analyzeBtn.addEventListener('click', analyzeImage);

    // Explain button
    explainBtn.addEventListener('click', generateExplainability);

    // New analysis button
    newAnalysisBtn.addEventListener('click', resetToUpload);
}

// Handle Drag Over
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.add('dragover');
}

// Handle Drag Leave
function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove('dragover');
}

// Handle Drop
function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// Handle File Select
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// Handle File
function handleFile(file) {
    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
        showToast('Please upload a valid image file (PNG, JPG, JPEG)', 'error');
        return;
    }

    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showToast('File size must be less than 16MB', 'error');
        return;
    }

    uploadedFile = file;

    // Update UI
    const reader = new FileReader();
    reader.onload = function(e) {
        dropzone.innerHTML = `
            <div class="dropzone-content">
                <img src="${e.target.result}" style="max-width: 200px; max-height: 200px; border-radius: 8px;">
                <p class="dropzone-text" style="color: var(--success-color); font-weight: 600;">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align: middle;">
                        <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                    </svg>
                    ${file.name}
                </p>
                <p class="file-info">${(file.size / 1024 / 1024).toFixed(2)} MB</p>
            </div>
        `;
        analyzeBtn.style.display = 'inline-flex';
    };
    reader.readAsDataURL(file);
}

// Analyze Image
async function analyzeImage() {
    if (!uploadedFile) {
        showToast('Please upload an image first', 'error');
        return;
    }

    // Show loading
    loadingOverlay.style.display = 'flex';

    // Create FormData
    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok && data.success) {
            currentPrediction = data.prediction;
            displayResults(data);
            showToast('Analysis complete!', 'success');
        } else {
            showToast(data.message || 'Error analyzing image', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showToast('An error occurred during analysis', 'error');
    } finally {
        loadingOverlay.style.display = 'none';
    }
}

// Display Results
function displayResults(data) {
    const { prediction, original_image } = data;

    // Show original image
    originalImage.src = original_image;

    // Show prediction
    predictionClass.textContent = prediction.class;
    predictionDescription.textContent = prediction.description;
    confidenceValue.textContent = prediction.confidence.toFixed(1);

    // Update confidence badge color
    if (prediction.confidence >= 90) {
        confidenceBadge.style.background = '#D4EDDA';
        confidenceBadge.style.color = '#155724';
    } else if (prediction.confidence >= 70) {
        confidenceBadge.style.background = '#FFF3CD';
        confidenceBadge.style.color = '#856404';
    } else {
        confidenceBadge.style.background = '#F8D7DA';
        confidenceBadge.style.color = '#721C24';
    }

    // Update prediction icon based on class
    updatePredictionIcon(prediction.class);

    // Display probability bars
    displayProbabilities(prediction.all_probabilities);

    // Show results section
    uploadSection.style.display = 'none';
    resultsSection.style.display = 'block';

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Update Prediction Icon
function updatePredictionIcon(className) {
    let iconColor = '';
    let iconSVG = '';

    switch(className.toLowerCase()) {
        case 'benign':
            iconColor = '#28A745';
            iconSVG = '<path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>';
            break;
        case 'malignant':
            iconColor = '#DC3545';
            iconSVG = '<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>';
            break;
        case 'normal':
            iconColor = '#17A2B8';
            iconSVG = '<path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>';
            break;
    }

    predictionIcon.style.background = iconColor + '20';
    predictionIcon.innerHTML = `
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="${iconColor}" stroke-width="2">
            ${iconSVG}
        </svg>
    `;
}

// Display Probabilities
function displayProbabilities(probabilities) {
    probabilityBars.innerHTML = '';

    Object.entries(probabilities).forEach(([className, probability]) => {
        const item = document.createElement('div');
        item.className = 'probability-item';
        item.innerHTML = `
            <div class="probability-label">
                <span class="probability-label-text">${className}</span>
                <span class="probability-value">${probability.toFixed(1)}%</span>
            </div>
            <div class="probability-bar-container">
                <div class="probability-bar" style="width: ${probability}%"></div>
            </div>
        `;
        probabilityBars.appendChild(item);
    });
}

// Generate Explainability
async function generateExplainability() {
    // Show explainability section
    explainabilitySection.style.display = 'block';

    // Reset images
    gradcamImage.style.display = 'none';
    limeImage.style.display = 'none';
    gradcamContainer.classList.add('loading');
    limeContainer.classList.add('loading');

    // Scroll to explainability
    explainabilitySection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Disable button
    explainBtn.disabled = true;
    explainBtn.innerHTML = `
        <div class="spinner" style="width: 20px; height: 20px;"></div>
        Generating...
    `;

    try {
        const response = await fetch('/explain', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        const data = await response.json();

        if (response.ok && data.success) {
            // Display Grad-CAM
            gradcamImage.src = data.gradcam;
            gradcamImage.style.display = 'block';
            gradcamContainer.classList.remove('loading');

            // Display LIME
            limeImage.src = data.lime;
            limeImage.style.display = 'block';
            limeContainer.classList.remove('loading');

            showToast('Explainability generated successfully!', 'success');
        } else {
            showToast(data.message || 'Error generating explainability', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showToast('An error occurred while generating explainability', 'error');
    } finally {
        // Re-enable button
        explainBtn.disabled = false;
        explainBtn.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3"/>
                <line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
            Regenerate Explainability
        `;
    }
}

// Reset to Upload
function resetToUpload() {
    // Reset variables
    uploadedFile = null;
    currentPrediction = null;

    // Reset file input
    fileInput.value = '';

    // Reset dropzone
    dropzone.innerHTML = `
        <div class="dropzone-content">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <path d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/>
            </svg>
            <p class="dropzone-text">Drop your image here or <span class="browse-link">browse</span></p>
            <p class="file-info">Supports: PNG, JPG, JPEG (Max 16MB)</p>
        </div>
    `;

    // Hide analyze button
    analyzeBtn.style.display = 'none';

    // Hide results and show upload
    resultsSection.style.display = 'none';
    uploadSection.style.display = 'block';

    // Hide explainability
    explainabilitySection.style.display = 'none';

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Show Toast Notification
function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    const toastMessage = toast.querySelector('.toast-message');
    const toastIcon = toast.querySelector('.toast-icon');

    // Update content
    toastMessage.textContent = message;

    // Update icon based on type
    if (type === 'error') {
        toast.classList.add('error');
        toastIcon.innerHTML = `
            <circle cx="12" cy="12" r="10"/>
            <line x1="15" y1="9" x2="9" y2="15"/>
            <line x1="9" y1="9" x2="15" y2="15"/>
        `;
    } else {
        toast.classList.remove('error');
        toastIcon.innerHTML = `
            <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
        `;
    }

    // Show toast
    toast.style.display = 'block';

    // Hide after 3 seconds
    setTimeout(() => {
        toast.style.display = 'none';
    }, 3000);
}

// Handle browser back/forward
window.addEventListener('popstate', function() {
    if (resultsSection.style.display === 'block') {
        resetToUpload();
    }
});