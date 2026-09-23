const API_URL = 'http://localhost:8000';

// State
let selectedModel = 'custom';
let imageFile = null;
let previewURL = null;
let predicting = false;
let explaining = false;

// Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const previewImg = document.getElementById('previewImg');
const predictBtn = document.getElementById('predictBtn');
const explainBtn = document.getElementById('explainBtn');
const predictBtnText = document.getElementById('predictBtnText');
const explainBtnText = document.getElementById('explainBtnText');
const errorBox = document.getElementById('errorBox');

const placeholder = document.getElementById('placeholder');
const resultBox = document.getElementById('resultBox');
const resultDisease = document.getElementById('resultDisease');
const resultAccuracy = document.getElementById('resultAccuracy');
const resultModel = document.getElementById('resultModel');
const accuracyFill = document.getElementById('accuracyFill');
const probsTitle = document.getElementById('probsTitle');
const probsList = document.getElementById('probsList');

const xaiCard = document.getElementById('xaiCard');
const xaiOriginal = document.getElementById('xaiOriginal');
const xaiGradcam = document.getElementById('xaiGradcam');

// --- Radio selection ---
document.querySelectorAll('.radio-option').forEach((label) => {
  label.addEventListener('click', () => {
    document.querySelectorAll('.radio-option').forEach((l) => l.classList.remove('active'));
    label.classList.add('active');
    const input = label.querySelector('input[type="radio"]');
    input.checked = true;
    selectedModel = input.value;
  });
});

// --- Upload handling ---
uploadArea.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
  handleFile(e.target.files[0]);
});

uploadArea.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
  uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadArea.classList.remove('dragover');
  if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) {
    showError('Please upload a valid image file.');
    return;
  }
  imageFile = file;
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = URL.createObjectURL(file);
  previewImg.src = previewURL;
  previewContainer.style.display = 'block';

  // Reset results
  hideError();
  placeholder.style.display = 'block';
  resultBox.style.display = 'none';
  xaiCard.style.display = 'none';
  probsList.innerHTML = '';
  probsTitle.style.display = 'none';

  predictBtn.disabled = false;
  explainBtn.disabled = false;
}

// --- Error helpers ---
function showError(msg) {
  errorBox.textContent = '⚠️ ' + msg;
  errorBox.style.display = 'block';
}

function hideError() {
  errorBox.style.display = 'none';
}

// --- Predict ---
predictBtn.addEventListener('click', async () => {
  if (!imageFile || predicting) return;
  predicting = true;
  predictBtn.disabled = true;
  explainBtn.disabled = true;
  predictBtnText.innerHTML = '<span class="loading"></span> Predicting...';
  hideError();
  xaiCard.style.display = 'none';

  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append('model_name', selectedModel);

  try {
    const res = await fetch(`${API_URL}/predict`, { method: 'POST', body: formData });
    const data = await res.json();
    if (data.success) {
      renderResult(data);
    } else {
      showError(data.error || 'Prediction failed.');
    }
  } catch (err) {
    showError('Cannot connect to backend. Make sure the API is running.');
  } finally {
    predicting = false;
    predictBtn.disabled = false;
    explainBtn.disabled = false;
    predictBtnText.innerHTML = '🔍 Predict Disease';
  }
});

// --- Explain (Grad-CAM) ---
explainBtn.addEventListener('click', async () => {
  if (!imageFile || explaining) return;
  explaining = true;
  predictBtn.disabled = true;
  explainBtn.disabled = true;
  explainBtnText.innerHTML = '<span class="loading"></span> Generating XAI...';
  hideError();

  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append('model_name', selectedModel);

  try {
    const res = await fetch(`${API_URL}/explain`, { method: 'POST', body: formData });
    const data = await res.json();
    if (data.success) {
      renderResult(data);
      xaiOriginal.src = previewURL;
      xaiGradcam.src = data.gradcam_image;
      xaiCard.style.display = 'block';
      xaiCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else {
      showError(data.error || 'XAI failed.');
    }
  } catch (err) {
    showError('Cannot connect to backend. Make sure the API is running.');
  } finally {
    explaining = false;
    predictBtn.disabled = false;
    explainBtn.disabled = false;
    explainBtnText.innerHTML = '🧠 Explain (Grad-CAM)';
  }
});

// --- Render result ---
function renderResult(data) {
  placeholder.style.display = 'none';
  resultBox.style.display = 'block';

  resultDisease.textContent = data.disease;
  resultAccuracy.textContent = data.accuracy + '%';
  resultModel.textContent = data.model_used;

  // Animate accuracy bar
  setTimeout(() => {
    accuracyFill.style.width = data.accuracy + '%';
  }, 50);

  // Probabilities
  if (data.all_probabilities) {
    probsTitle.style.display = 'block';
    probsList.innerHTML = '';
    const sorted = Object.entries(data.all_probabilities).sort((a, b) => b[1] - a[1]);
    sorted.forEach(([name, prob]) => {
      const pct = (prob * 100).toFixed(2);
      const item = document.createElement('div');
      item.className = 'prob-item';
      item.innerHTML = `
        <span class="prob-name">${name}</span>
        <div class="prob-bar"><div class="prob-fill" style="width:${pct}%"></div></div>
        <span class="prob-value">${pct}%</span>
      `;
      probsList.appendChild(item);
    });
  } else {
    probsTitle.style.display = 'none';
    probsList.innerHTML = '';
  }
}