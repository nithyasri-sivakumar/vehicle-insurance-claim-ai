const uploadBox = document.querySelector('.upload-box');
const fileInput = document.querySelector('#image-upload');
const uploadLabel = document.querySelector('.upload-label');
const uploadPreview = document.querySelector('#upload-preview');

if (uploadBox && fileInput && uploadLabel && uploadPreview) {
    uploadBox.addEventListener('click', () => fileInput.click());

    const showPreview = (file) => {
        if (!file) {
            uploadLabel.textContent = 'Drop your image here or click to browse';
            uploadPreview.removeAttribute('src');
            uploadPreview.classList.remove('is-visible');
            uploadBox.classList.remove('has-preview');
            return;
        }

        uploadLabel.textContent = file.name;
        uploadPreview.src = URL.createObjectURL(file);
        uploadPreview.classList.add('is-visible');
        uploadBox.classList.add('has-preview');
    };

    fileInput.addEventListener('change', () => {
        showPreview(fileInput.files && fileInput.files[0] ? fileInput.files[0] : null);
    });

    uploadPreview.addEventListener('load', () => {
        if (uploadPreview.dataset.objectUrl) {
            URL.revokeObjectURL(uploadPreview.dataset.objectUrl);
        }
        uploadPreview.dataset.objectUrl = uploadPreview.src;
    });
}
