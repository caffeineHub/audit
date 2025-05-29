from fastapi import FastAPI, File, UploadFile, Path, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import shutil
import os
import uuid
import zipfile

# Assuming audit_processing.py is in the same directory or accessible in PYTHONPATH
from audit_processing import perform_full_audio_audit

app = FastAPI(title="Customer Service Audio Audit API")

# Mount static files directory (before any conflicting routes)
# Ensure 'static' directory exists at the same level as main.py
STATIC_DIR = "static"
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR) # Create if it doesn't exist, for robustness
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --- Constants ---
TEMP_UPLOADS_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOADS_DIR, exist_ok=True) # Ensure temp directory exists

# --- Pydantic Models ---

class AudioAuditResponse(BaseModel):
    """Response model for the audio audit result."""
    audio_file: str
    transcription: Optional[str] = None
    sentiment: Optional[str] = None
    compliance_issues: Optional[List[str]] = None
    summary: Optional[str] = None
    error: Optional[str] = None
    status: Optional[str] = None # e.g., COMPLETED, FAILED

class FileUploadResponse(BaseModel): # Kept for the ZIP endpoint's current placeholder response
    """Response model for basic file uploads (used by ZIP placeholder)."""
    filename: str
    message: str

class TaskStatusResponse(BaseModel):
    """Response model for task status, adapted for current synchronous processing."""
    task_id: str
    status: str
    message: str
    result: Optional[AudioAuditResponse] = None # Kept for future async, but won't be populated now

class ZipUploadResponse(BaseModel):
    """Response model for ZIP file uploads, containing results for each processed file."""
    message: str
    processed_files: List[AudioAuditResponse] = []
    errors: List[str] = [] # For general errors like "not a zip file"

# --- API Endpoints ---

@app.post("/upload/audio/", response_model=AudioAuditResponse)
async def upload_audio(file: UploadFile = File(...)):
    """
    Accepts a single audio file (e.g., .wav, .mp3) for transcription and analysis.
    The processing is synchronous in this version.
    """
    # Secure filename and construct path
    # Using original filename directly can be risky. For now, let's sanitize lightly or use a generated name.
    # For this implementation, we'll use the original filename within our controlled TEMP_UPLOADS_DIR.
    # Consider more robust sanitization or unique name generation in production.
    filename = file.filename
    if not filename: # Should not happen with UploadFile but good check
        raise HTTPException(status_code=400, detail="No filename provided.")
    
    # Basic sanitization to prevent path traversal, though FastAPI/Starlette might handle some of this.
    # This is a naive sanitization. Production systems need more robust handling.
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename characters.")

    temp_file_path = os.path.join(TEMP_UPLOADS_DIR, filename)

    try:
        print(f"Saving uploaded file to: {temp_file_path}")
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"File saved. Starting audio audit for: {temp_file_path}")
        # This is a synchronous call. For long processing, consider background tasks.
        analysis_result = perform_full_audio_audit(temp_file_path)
        print(f"Audit complete for {filename}. Result: {analysis_result}")

        # Check if the audit itself reported an error
        if analysis_result.get("status") == "FAILED" or "error" in analysis_result:
             # Return the structured error from the audit process
            return AudioAuditResponse(**analysis_result)
            # Or, if we want to raise an HTTP Exception for all errors:
            # raise HTTPException(status_code=500, detail=analysis_result.get("error", "Unknown processing error"))

        return AudioAuditResponse(**analysis_result)

    except FileNotFoundError as e:
        print(f"Error: Input file not found during processing for {filename}: {e}")
        raise HTTPException(status_code=404, detail=f"File not found during processing: {filename}")
    except ValueError as e: # Catching specific errors from audit_processing
        print(f"Error: Value error during processing for {filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid data or processing error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error during processing for {filename}: {e}")
        # Log the full exception here in a real app
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        # Cleanup: Remove the temporary file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"Successfully removed temporary file: {temp_file_path}")
            except OSError as e:
                # Log this error, but don't let cleanup failure break the response
                print(f"Error removing temporary file {temp_file_path}: {e}")
        
        # Ensure file object is closed, though `with open` handles the buffer.
        # `file.file` (SpooledTemporaryFile) should be handled by FastAPI/Starlette.
        if hasattr(file, 'close'):
             file.close() # Not typically needed for file.file due to context manager


@app.post("/upload/zip/", response_model=ZipUploadResponse)
async def upload_zip(file: UploadFile = File(...)):
    """
    Accepts a ZIP file, extracts audio files, processes each, and returns results.
    Processing is synchronous for each file within the ZIP.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided for the ZIP file.")
    
    if not file.filename.endswith(".zip"):
        # It's better to raise HTTPException for input validation
        raise HTTPException(status_code=400, detail="Invalid file type. Only .zip files are accepted.")

    # Generate a unique name for the saved ZIP file to avoid conflicts
    unique_zip_filename = f"{uuid.uuid4().hex}_{file.filename}"
    temp_zip_path = os.path.join(TEMP_UPLOADS_DIR, unique_zip_filename)
    
    # Generate a unique name for the extraction directory
    extraction_subdir_name = f"zip_extraction_{uuid.uuid4().hex}"
    extraction_path = os.path.join(TEMP_UPLOADS_DIR, extraction_subdir_name)

    results: List[AudioAuditResponse] = []
    general_errors: List[str] = []

    SUPPORTED_AUDIO_EXTENSIONS = [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"] # Add more as needed

    try:
        print(f"Saving uploaded ZIP file to: {temp_zip_path}")
        with open(temp_zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"Successfully saved ZIP file: {temp_zip_path}")
        
        # Create the unique extraction directory
        os.makedirs(extraction_path, exist_ok=True)
        print(f"Created extraction directory: {extraction_path}")

        try:
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extraction_path)
            print(f"Successfully extracted ZIP contents to: {extraction_path}")
        except zipfile.BadZipFile:
            print(f"Error: Uploaded file {file.filename} is not a valid ZIP file or is corrupted.")
            # Return error using the ZipUploadResponse model
            # We don't raise HTTPException here to allow cleanup in `finally`
            general_errors.append(f"Uploaded file '{file.filename}' is not a valid ZIP file or is corrupted.")
            # Early exit if not a valid zip, no further processing possible
            return ZipUploadResponse(message="Error processing ZIP file.", errors=general_errors, processed_files=[])


        # Walk through the extracted files and process audio files
        for root, _, files_in_dir in os.walk(extraction_path):
            for item_name in files_in_dir:
                item_path = os.path.join(root, item_name)
                item_ext = os.path.splitext(item_name)[1].lower()

                if item_ext in SUPPORTED_AUDIO_EXTENSIONS:
                    print(f"Found supported audio file: {item_path}. Starting audit...")
                    try:
                        # perform_full_audio_audit expects the file path
                        analysis_result_dict = perform_full_audio_audit(item_path)
                        
                        # Ensure the result is AudioAuditResponse compatible
                        # perform_full_audio_audit already returns a dict suitable for AudioAuditResponse
                        results.append(AudioAuditResponse(**analysis_result_dict))
                        print(f"Audit complete for {item_name}.")
                    except Exception as e:
                        print(f"Error processing audio file {item_name}: {e}")
                        # Add an error entry for this specific file to results
                        results.append(AudioAuditResponse(
                            audio_file=item_name,
                            error=f"Failed to process audio file: {str(e)}",
                            status="FAILED"
                        ))
                else:
                    print(f"Skipping non-audio or unsupported file: {item_path}")
        
        if not results and not general_errors: # No audio files found or processed
            general_errors.append("No supported audio files found in the ZIP archive.")

        return ZipUploadResponse(
            message="ZIP file processed." if not general_errors else "ZIP file processed with errors.",
            processed_files=results,
            errors=general_errors
        )

    except HTTPException: # Re-raise HTTPExceptions if any occurred before main try-catch
        raise
    except Exception as e:
        print(f"An unexpected error occurred during ZIP processing for {file.filename}: {e}")
        # Log the full exception here in a real app
        # Return a general error message using the response model
        general_errors.append(f"An unexpected server error occurred: {str(e)}")
        return ZipUploadResponse(
            message="Unexpected server error during ZIP processing.",
            errors=general_errors,
            processed_files=results # Include any partial results if available
        )
    finally:
        # --- Cleanup ---
        # 1. Delete the temporary saved ZIP file
        if os.path.exists(temp_zip_path):
            try:
                os.remove(temp_zip_path)
                print(f"Successfully removed temporary ZIP file: {temp_zip_path}")
            except OSError as e:
                print(f"Error removing temporary ZIP file {temp_zip_path}: {e}") # Log this
        
        # 2. Delete the temporary extraction subdirectory and its contents
        if os.path.exists(extraction_path):
            try:
                shutil.rmtree(extraction_path)
                print(f"Successfully removed extraction directory: {extraction_path}")
            except OSError as e:
                print(f"Error removing extraction directory {extraction_path}: {e}") # Log this
        
        # Ensure the uploaded file object is closed
        if hasattr(file, 'close'):
            file.close()


@app.get("/results/{task_id}/", response_model=TaskStatusResponse)
async def get_task_results(task_id: str = Path(..., title="Task ID", description="The ID of the processing task.")):
    """
    Retrieves the status of a processing task.
    Currently, as processing is synchronous, this endpoint serves as a placeholder.
    """
    # In a future asynchronous system, this endpoint would query a task manager
    # (e.g., Celery results backend, database) for the status of task_id.
    # For now, it returns a message reflecting synchronous processing.
    
    # Basic check for a "valid" looking task_id, though we don't store them.
    # This is just to make the placeholder slightly more interactive.
    if not task_id or len(task_id) < 3: # Arbitrary minimum length
        raise HTTPException(status_code=400, detail="Invalid Task ID format.")

    return TaskStatusResponse(
        task_id=task_id,
        status="NOT_APPLICABLE_FOR_SYNC_PROCESSING",
        message="For current synchronous processing, results are returned directly with the POST upload request. This endpoint is a placeholder for future asynchronous task status tracking.",
        result=None # No result to fetch for a sync task via this endpoint
    )

# --- Root Endpoint to Serve Frontend ---
@app.get("/", response_class=FileResponse)
async def read_root():
    """Serves the main HTML page for the frontend."""
    html_file_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(html_file_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(html_file_path)

# To run this application (for development):
# uvicorn main:app --reload
#
# Then access the API docs at http://127.0.0.1:8000/docs
# or http://127.0.0.1:8000/redoc (API interface)
# and the frontend at http://127.0.0.1:8000/
