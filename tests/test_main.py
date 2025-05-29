import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os
import shutil
import zipfile
from io import BytesIO

# Ensure the app can be imported.
# This might require adjusting PYTHONPATH or how 'main' is imported if tests are run from 'tests/' directory.
# For now, assuming 'main.py' is in the parent directory of 'tests/'.
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import app, TEMP_UPLOADS_DIR

# Fixture for TestClient
@pytest.fixture(scope="module")
def client():
    # Ensure TEMP_UPLOADS_DIR exists before tests and is cleaned up after
    if os.path.exists(TEMP_UPLOADS_DIR):
        shutil.rmtree(TEMP_UPLOADS_DIR) # Clean up from previous runs if any
    os.makedirs(TEMP_UPLOADS_DIR, exist_ok=True)
    
    with TestClient(app) as c:
        yield c
    
    # Cleanup after all tests in the module are done
    if os.path.exists(TEMP_UPLOADS_DIR):
        shutil.rmtree(TEMP_UPLOADS_DIR)

# Fixture for creating dummy files
@pytest.fixture
def create_dummy_file(tmp_path_factory):
    def _create_dummy_file(filename, content=b"dummy content"):
        file_path = tmp_path_factory.mktemp("data") / filename
        file_path.write_bytes(content)
        return file_path
    return _create_dummy_file

# --- Tests for /upload/audio/ ---

@patch('main.perform_full_audio_audit')
def test_upload_audio_success(mock_perform_audit, client, create_dummy_file):
    mock_perform_audit.return_value = {
        "audio_file": "sample_valid.mp3",
        "transcription": "test transcription",
        "sentiment": "neutral",
        "compliance_issues": [],
        "summary": "Test summary",
        "status": "COMPLETED"
    }
    
    dummy_file_path = create_dummy_file("sample_valid.mp3")
    
    with open(dummy_file_path, "rb") as f:
        response = client.post("/upload/audio/", files={"file": ("sample_valid.mp3", f, "audio/mpeg")})
    
    assert response.status_code == 200
    data = response.json()
    assert data["audio_file"] == "sample_valid.mp3"
    assert data["transcription"] == "test transcription"
    assert data["status"] == "COMPLETED"
    mock_perform_audit.assert_called_once()
    # Check if the temp file was cleaned up by main.py's logic
    assert not os.path.exists(os.path.join(TEMP_UPLOADS_DIR, "sample_valid.mp3"))


@patch('main.perform_full_audio_audit')
def test_upload_audio_processing_error_response(mock_perform_audit, client, create_dummy_file):
    mock_perform_audit.return_value = {
        "audio_file": "error_sample.mp3",
        "error": "mocked processing error",
        "status": "FAILED"
    }
    dummy_file_path = create_dummy_file("error_sample.mp3")
    with open(dummy_file_path, "rb") as f:
        response = client.post("/upload/audio/", files={"file": ("error_sample.mp3", f, "audio/mpeg")})
    
    assert response.status_code == 200 # API returns 200 but body indicates failure
    data = response.json()
    assert data["audio_file"] == "error_sample.mp3"
    assert data["error"] == "mocked processing error"
    assert data["status"] == "FAILED"
    mock_perform_audit.assert_called_once()

@patch('main.perform_full_audio_audit')
def test_upload_audio_processing_exception(mock_perform_audit, client, create_dummy_file):
    mock_perform_audit.side_effect = ValueError("Something went very wrong during processing")
    dummy_file_path = create_dummy_file("exception_sample.mp3")
    with open(dummy_file_path, "rb") as f:
        response = client.post("/upload/audio/", files={"file": ("exception_sample.mp3", f, "audio/mpeg")})
    
    assert response.status_code == 400 # Based on main.py's ValueError handling
    data = response.json()
    assert "Invalid data or processing error: Something went very wrong during processing" in data["detail"]
    mock_perform_audit.assert_called_once()

def test_upload_audio_no_file(client):
    response = client.post("/upload/audio/", files={"file": ("empty.mp3", b"", "audio/mpeg")})
    # This will likely be caught by perform_full_audio_audit if it tries to process an empty file.
    # If perform_full_audio_audit is robust, it might return a specific error.
    # If not, it might raise an exception caught by the general error handlers.
    # For this test, since perform_full_audio_audit is not mocked here, it will actually run.
    # Given the constraints, we expect it to fail as dependencies are not present.
    # This test will be more meaningful if perform_full_audio_audit is mocked.
    # For now, let's expect a 500 or similar if the real function is hit without mocks and fails due to missing deps.
    # However, the goal is to test main.py's handling.
    # Let's assume 0-byte file makes `perform_full_audio_audit` (mocked) return an error
    with patch('main.perform_full_audio_audit') as mock_audit_empty:
        mock_audit_empty.return_value = {"audio_file": "empty.mp3", "error": "Cannot process empty file", "status": "FAILED"}
        response = client.post("/upload/audio/", files={"file": ("empty.mp3", BytesIO(b""), "audio/mpeg")}) # Use BytesIO for empty file
    
    assert response.status_code == 200
    data = response.json()
    assert data["error"] == "Cannot process empty file"


# --- Tests for /upload/zip/ ---

@patch('main.perform_full_audio_audit')
def test_upload_zip_success_multiple_files(mock_perform_audit, client, tmp_path):
    # Create a dummy zip file
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("audio1.mp3", b"dummy audio data 1")
        zf.writestr("audio2.wav", b"dummy audio data 2")
        zf.writestr("readme.txt", b"this is not audio")
    zip_buffer.seek(0)

    # Define different return values for each call to the mock
    mock_perform_audit.side_effect = [
        {"audio_file": "audio1.mp3", "transcription": "trans1", "status": "COMPLETED"},
        {"audio_file": "audio2.wav", "transcription": "trans2", "status": "COMPLETED"},
    ]

    response = client.post("/upload/zip/", files={"file": ("archive.zip", zip_buffer, "application/zip")})
    
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "ZIP file processed."
    assert len(data["processed_files"]) == 2
    assert data["processed_files"][0]["audio_file"] == "audio1.mp3"
    assert data["processed_files"][1]["audio_file"] == "audio2.wav"
    assert mock_perform_audit.call_count == 2
    # Check that the temporary zip and extraction folder are cleaned up (harder to check specific names due to UUID)
    # We rely on the client fixture's overall cleanup of TEMP_UPLOADS_DIR for now.

@patch('main.perform_full_audio_audit')
def test_upload_zip_mixed_results(mock_perform_audit, client, tmp_path):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("good.mp3", b"good audio")
        zf.writestr("bad.mp3", b"bad audio that will fail")
    zip_buffer.seek(0)

    mock_perform_audit.side_effect = [
        {"audio_file": "good.mp3", "transcription": "good trans", "status": "COMPLETED"},
        {"audio_file": "bad.mp3", "error": "processing failed for bad", "status": "FAILED"},
    ]

    response = client.post("/upload/zip/", files={"file": ("mixed.zip", zip_buffer, "application/zip")})
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "ZIP file processed." # Or "ZIP file processed with errors." depending on how we define this
    assert len(data["processed_files"]) == 2
    assert data["processed_files"][0]["status"] == "COMPLETED"
    assert data["processed_files"][1]["status"] == "FAILED"
    assert data["processed_files"][1]["error"] == "processing failed for bad"
    assert mock_perform_audit.call_count == 2

def test_upload_zip_invalid_zip_format(client):
    # Upload a text file as a zip
    response = client.post("/upload/zip/", files={"file": ("notazip.zip", BytesIO(b"this is not a zip file"), "application/zip")})
    assert response.status_code == 200 # main.py returns 200 with error in body for BadZipFile
    data = response.json()
    assert "Error processing ZIP file." in data["message"]
    assert any("not a valid ZIP file or is corrupted" in err for err in data["errors"])

def test_upload_zip_empty_zip_file(client):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        pass # Create an empty zip
    zip_buffer.seek(0)
    
    response = client.post("/upload/zip/", files={"file": ("empty.zip", zip_buffer, "application/zip")})
    assert response.status_code == 200
    data = response.json()
    assert "No supported audio files found in the ZIP archive." in data["errors"]
    assert len(data["processed_files"]) == 0

@patch('main.perform_full_audio_audit') # Mock to prevent actual calls
def test_upload_zip_no_audio_files(mock_perform_audit, client):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("info.txt", b"some text data")
        zf.writestr("document.pdf", b"pdf data")
    zip_buffer.seek(0)

    response = client.post("/upload/zip/", files={"file": ("no_audio.zip", zip_buffer, "application/zip")})
    assert response.status_code == 200
    data = response.json()
    assert "No supported audio files found in the ZIP archive." in data["errors"]
    assert len(data["processed_files"]) == 0
    mock_perform_audit.assert_not_called()


# --- Tests for /results/{task_id}/ ---

def test_get_results_placeholder(client):
    response = client.get("/results/some-task-id/")
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == "some-task-id"
    assert data["status"] == "NOT_APPLICABLE_FOR_SYNC_PROCESSING"
    assert "results are returned directly" in data["message"]

def test_get_results_invalid_task_id_format(client):
    response = client.get("/results/s/") # Task ID too short based on current main.py logic
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Invalid Task ID format."

# --- Test for root path ---
def test_read_root(client):
    # Need to ensure static/index.html exists for this test or mock FileResponse
    # For now, let's assume it exists as created in previous steps.
    # If not, this test would fail with 404 from the app, or 500 if check not in app.
    
    # Create a dummy static/index.html for test environment if it's not there
    static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
    os.makedirs(static_dir, exist_ok=True)
    with open(os.path.join(static_dir, "index.html"), "w") as f:
        f.write("<html><body>Test Index</body></html>")
        
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert "Test Index" in response.text

# Comments below are for context and were part of the thought process,
# ensuring they are properly commented out for the Python interpreter.

# It's important that main.py's TEMP_UPLOADS_DIR is correctly managed by tests,
# especially ensuring it's clean before tests and potentially after each test or module.
# The client fixture handles initial cleanup and creation.
# Individual tests also check for file cleanup where appropriate.

# To run these tests:
# Ensure PYTHONPATH includes the project root if running from the 'tests' directory:
# export PYTHONPATH=$PYTHONPATH:$(pwd)/.. (if in tests dir)
# Then run:
# pytest
# Or from project root:
# pytest tests/test_main.py

# Note on mocking `audit_processing` functions within `main.py`:
# If `from audit_processing import perform_full_audio_audit` is used in `main.py`,
# then the patch target should be `'main.perform_full_audio_audit'`.
# The current patch `'main.perform_full_audio_audit'` assumes this is the case.

# The `create_dummy_file` fixture uses `tmp_path_factory` which is a pytest fixture
# providing temporary directories, separate from TEMP_UPLOADS_DIR.
# This is good for providing input files that are not expected to be in TEMP_UPLOADS_DIR.
# The actual uploaded files will be written to TEMP_UPLOADS_DIR by the app's logic.
# The tests correctly verify that files in TEMP_UPLOADS_DIR are cleaned up by the app.

# The test `test_upload_audio_no_file` was modified to use BytesIO for a truly empty file payload
# and explicitly mocks `perform_full_audio_audit` for that specific case.

# The test for `test_read_root` creates a dummy `static/index.html` to ensure it passes
# in CI environments where the file might not exist from previous steps.

# Cleanup check in `test_upload_audio_success`:
# `assert not os.path.exists(os.path.join(TEMP_UPLOADS_DIR, "sample_valid.mp3"))`
# This verifies the cleanup logic in the `upload_audio` endpoint.

# Cleanup for ZIP uploads:
# The temporary zip file and extraction directory are cleaned up by the `finally` block
# in the `upload_zip` endpoint in `main.py`. Verifying this precisely is tricky due to UUIDs.
# The overall cleanup of TEMP_UPLOADS_DIR by the client fixture provides general safety.

# `sys.path.insert` is used to allow `from main import app` to work when running pytest
# from the project root and the test file is in a subdirectory.
# The project structure assumed is standard.

# The client fixture scope is "module", meaning TEMP_UPLOADS_DIR is set up and torn down
# once per test file (module). This is generally acceptable as `main.py` cleans its own
# temporary files per request.

# Tests for ZIP uploads with `BadZipFile`, empty ZIPs, or no audio files correctly assert
# a 200 OK status, because `main.py` handles these by returning an error message
# within the `ZipUploadResponse` body, not by raising an HTTPException.
# This reflects the current error handling strategy in `main.py`.
