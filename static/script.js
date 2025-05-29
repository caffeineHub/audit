document.addEventListener('DOMContentLoaded', () => {
    const uploadAudioForm = document.getElementById('uploadAudioForm');
    const uploadZipForm = document.getElementById('uploadZipForm');
    const resultsDiv = document.getElementById('results');
    const errorDiv = document.getElementById('error-message');
    const loadingDiv = document.getElementById('loading-message');

    const handleUpload = async (event, url) => {
        event.preventDefault();
        
        resultsDiv.textContent = ''; // Clear previous results
        errorDiv.textContent = '';   // Clear previous errors
        loadingDiv.textContent = 'Uploading and processing, please wait...';
        loadingDiv.style.display = 'block';

        const formData = new FormData(event.target);
        const fileInput = event.target.querySelector('input[type="file"]');
        
        if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
            errorDiv.textContent = 'Please select a file to upload.';
            loadingDiv.style.display = 'none';
            return;
        }

        let responseText = ''; // To store raw response text if JSON parsing fails

        try {
            const response = await fetch(url, {
                method: 'POST',
                body: formData,
            });

            // Attempt to get raw text first in case of non-JSON error
            responseText = await response.text(); 
            let responseData;
            try {
                responseData = JSON.parse(responseText); // Try to parse the text as JSON
            } catch (e) {
                // If JSON parsing fails, responseData will be undefined.
                // We'll use responseText later if response.ok is false.
                console.warn("Failed to parse server response as JSON. Raw response:", responseText);
            }

            loadingDiv.style.display = 'none';

            if (response.ok) {
                // Even if response.ok, the business logic might have failed (common in AudioAuditResponse/ZipUploadResponse)
                let businessLogicError = '';
                if (url.includes('/upload/audio/') && responseData) { // Single audio upload
                    if (responseData.status === 'FAILED' || responseData.error) {
                        businessLogicError = `Processing failed: ${responseData.error || 'Unknown error'}`;
                        if(responseData.audio_file) businessLogicError += ` (File: ${responseData.audio_file})`;
                    }
                } else if (url.includes('/upload/zip/') && responseData) { // ZIP upload
                    if (responseData.errors && responseData.errors.length > 0) {
                        businessLogicError = `ZIP processing errors: ${responseData.errors.join(', ')}\n`;
                    }
                    if (responseData.processed_files) {
                        const fileErrors = responseData.processed_files
                            .filter(f => f.status === 'FAILED' || f.error)
                            .map(f => `File '${f.audio_file}': ${f.error || 'Failed'}`);
                        if (fileErrors.length > 0) {
                            businessLogicError += `Individual file errors:\n${fileErrors.join('\n')}`;
                        }
                    }
                }

                if (businessLogicError) {
                    errorDiv.innerHTML = `Request successful, but processing issues found:<br><pre>${businessLogicError.replace(/\n/g, '<br>')}</pre>`;
                    resultsDiv.textContent = JSON.stringify(responseData, null, 2); // Still show full JSON for details
                } else {
                    resultsDiv.textContent = JSON.stringify(responseData, null, 2);
                }

            } else { // response.ok is false (HTTP error status)
                let errorMsg = `Error ${response.status}: `;
                if (responseData && responseData.detail) { // FastAPI HTTPException
                    if (typeof responseData.detail === 'string') {
                        errorMsg += responseData.detail;
                    } else {
                        errorMsg += JSON.stringify(responseData.detail, null, 2);
                    }
                } else if (responseData) { // Other JSON error from server
                    errorMsg += JSON.stringify(responseData, null, 2);
                } else { // Non-JSON error response
                    errorMsg += responseText || "Could not retrieve error details.";
                }
                errorDiv.textContent = errorMsg;
            }
        } catch (error) { // Network error or error during fetch/parsing not caught above
            loadingDiv.style.display = 'none';
            errorDiv.textContent = `Network error or server unreachable. Please check your connection. Details: ${error.message || error}`;
            console.error('Upload error:', error);
        } finally {
            // Reset the form to allow re-uploading the same file if needed
             event.target.reset();
        }
    };

    if (uploadAudioForm) {
        uploadAudioForm.addEventListener('submit', (event) => {
            handleUpload(event, '/upload/audio/');
        });
    }

    if (uploadZipForm) {
        uploadZipForm.addEventListener('submit', (event) => {
            handleUpload(event, '/upload/zip/');
        });
    }
});
