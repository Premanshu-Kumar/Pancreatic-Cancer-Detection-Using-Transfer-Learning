/**
 * Cancer Detection System — Frontend JavaScript
 * ================================================
 * Handles drag-and-drop file upload, image preview,
 * form validation, and loading states.
 */

document.addEventListener('DOMContentLoaded', function () {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const dropContent = document.getElementById('drop-zone-content');
    const imagePreview = document.getElementById('image-preview');
    const previewImg = document.getElementById('preview-img');
    const removeBtn = document.getElementById('remove-btn');
    const fileName = document.getElementById('file-name');
    const form = document.getElementById('prediction-form');
    const analyzeBtn = document.getElementById('analyze-btn');
    const btnText = document.querySelector('.btn-text');
    const btnLoader = document.getElementById('btn-loader');

    if (!dropZone) return; // Not on the upload page

    // ─── Drag & Drop ────────────────────────────────────────────────────
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(event => {
        dropZone.addEventListener(event, e => {
            e.preventDefault();
            e.stopPropagation();
        });
    });

    ['dragenter', 'dragover'].forEach(event => {
        dropZone.addEventListener(event, () => {
            dropZone.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(event => {
        dropZone.addEventListener(event, () => {
            dropZone.classList.remove('drag-over');
        });
    });

    dropZone.addEventListener('drop', function (e) {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // ─── Click to Upload ────────────────────────────────────────────────
    dropZone.addEventListener('click', function (e) {
        if (e.target === removeBtn || e.target.closest('.remove-btn')) return;
        fileInput.click();
    });

    fileInput.addEventListener('change', function () {
        if (this.files.length > 0) {
            handleFile(this.files[0]);
        }
    });

    // ─── Handle File Selection ──────────────────────────────────────────
    function handleFile(file) {
        const allowedTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff'];
        const allowedExtensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'];

        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (!allowedExtensions.includes(ext)) {
            showError('Invalid file type. Please upload JPG, PNG, BMP, or TIFF.');
            return;
        }

        // Size check (10MB)
        if (file.size > 10 * 1024 * 1024) {
            showError('File too large. Maximum size is 10MB.');
            return;
        }

        // Set file to input
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInput.files = dataTransfer.files;

        // Show preview
        const reader = new FileReader();
        reader.onload = function (e) {
            previewImg.src = e.target.result;
            dropContent.style.display = 'none';
            imagePreview.style.display = 'flex';
            fileName.textContent = file.name;
            fileName.style.color = '#00cec9';
        };
        reader.readAsDataURL(file);
    }

    // ─── Remove Image ───────────────────────────────────────────────────
    if (removeBtn) {
        removeBtn.addEventListener('click', function (e) {
            e.preventDefault();
            e.stopPropagation();
            clearPreview();
        });
    }

    function clearPreview() {
        fileInput.value = '';
        previewImg.src = '';
        dropContent.style.display = 'flex';
        imagePreview.style.display = 'none';
        fileName.textContent = '';
    }

    // ─── Form Submission ────────────────────────────────────────────────
    if (form) {
        form.addEventListener('submit', function (e) {
            if (!fileInput.files || fileInput.files.length === 0) {
                e.preventDefault();
                showError('Please select an image first.');
                return;
            }

            // Show loading state
            if (btnText) btnText.style.display = 'none';
            if (btnLoader) btnLoader.style.display = 'inline-flex';
            if (analyzeBtn) analyzeBtn.disabled = true;
        });
    }

    // ─── Error Display ──────────────────────────────────────────────────
    function showError(message) {
        // Create toast notification
        const toast = document.createElement('div');
        toast.className = 'error-toast';
        toast.innerHTML = `
            <span class="toast-icon">⚠️</span>
            <span class="toast-message">${message}</span>
        `;

        // Style the toast
        Object.assign(toast.style, {
            position: 'fixed',
            top: '90px',
            right: '24px',
            zIndex: '9999',
            background: 'rgba(225, 112, 85, 0.95)',
            color: 'white',
            padding: '14px 20px',
            borderRadius: '10px',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            fontSize: '0.9rem',
            fontWeight: '500',
            fontFamily: 'Inter, sans-serif',
            boxShadow: '0 8px 30px rgba(225, 112, 85, 0.3)',
            animation: 'fadeInUp 0.3s ease',
            backdropFilter: 'blur(10px)',
        });

        document.body.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(-10px)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 3500);
    }
});
