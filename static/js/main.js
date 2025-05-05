// Main JavaScript file for Audio Emotion Detector

// Wait for the DOM to load
document.addEventListener('DOMContentLoaded', function() {
    console.log('Audio Emotion Detector initialized');
    
    // Add custom file input behavior
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            // Get the file name
            const fileName = e.target.files[0]?.name || 'No file chosen';
            
            // Log file selection
            console.log(`File selected: ${fileName}`);
            
            // Validate file type
            if (e.target.files.length > 0) {
                const fileExt = fileName.split('.').pop().toLowerCase();
                if (!['mp3', 'wav', 'mp4'].includes(fileExt)) {
                    // Create an alert for invalid file type
                    const alertDiv = document.createElement('div');
                    alertDiv.className = 'alert alert-danger alert-dismissible fade show mt-3';
                    alertDiv.role = 'alert';
                    alertDiv.innerHTML = `
                        Invalid file type. Please select an MP3, WAV, or MP4 file.
                        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                    `;
                    
                    // Remove any existing alerts
                    const existingAlerts = document.querySelectorAll('.alert-danger');
                    existingAlerts.forEach(alert => alert.remove());
                    
                    // Add the alert before the form
                    const form = document.getElementById('upload-form');
                    form.parentNode.insertBefore(alertDiv, form);
                    
                    // Clear the file input
                    fileInput.value = '';
                }
            }
        });
    }
    
    // Handle form submission
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            // Check if a file is selected
            if (fileInput && fileInput.files.length === 0) {
                e.preventDefault();
                
                // Create an alert for missing file
                const alertDiv = document.createElement('div');
                alertDiv.className = 'alert alert-warning alert-dismissible fade show mt-3';
                alertDiv.role = 'alert';
                alertDiv.innerHTML = `
                    Please select a file to upload.
                    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                `;
                
                // Remove any existing alerts
                const existingAlerts = document.querySelectorAll('.alert-warning');
                existingAlerts.forEach(alert => alert.remove());
                
                // Add the alert before the form
                uploadForm.parentNode.insertBefore(alertDiv, uploadForm);
                
                return false;
            }
            
            // If file is selected, show loading state
            const submitButton = uploadForm.querySelector('button[type="submit"]');
            if (submitButton) {
                submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
                submitButton.disabled = true;
            }
            
            return true;
        });
    }
});

// Function to animate elements when they come into view
function animateOnScroll() {
    const elements = document.querySelectorAll('.animate-on-scroll');
    
    elements.forEach(element => {
        const elementPosition = element.getBoundingClientRect().top;
        const screenPosition = window.innerHeight / 1.3;
        
        if (elementPosition < screenPosition) {
            element.classList.add('animated');
        }
    });
}

// Attach scroll event listener if there are elements to animate
if (document.querySelectorAll('.animate-on-scroll').length > 0) {
    window.addEventListener('scroll', animateOnScroll);
    // Initial check for elements in view
    animateOnScroll();
}
