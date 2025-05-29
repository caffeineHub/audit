# Customer Service Audio Recording Audit Project

## Project Purpose

This project is designed to audit customer service audio recordings. Its primary goal is to analyze interactions between customer service representatives and customers. This process involves processing audio files to extract conversational data, which can then be reviewed and analyzed to assess service quality, identify areas for improvement, and ensure compliance with company policies.

Key objectives include:
- Transcribing audio recordings of customer service calls.
- Analyzing the transcribed text to identify keywords, sentiment, and topics of discussion.
- Providing insights into customer satisfaction and representative performance.
- Flagging calls that may require further review based on predefined criteria.

By leveraging audio processing and natural language understanding techniques, this project aims to provide a comprehensive solution for auditing customer service interactions and deriving actionable insights from them.

## Main Functionalities

The project involves the following key steps:

1.  **Audio Preprocessing:**
    *   **Purpose:** To improve the quality of the audio recordings before transcription. This step focuses on reducing background noise and enhancing speech clarity.
    *   **Tool:** `resemble_enhance` is utilized for its capabilities in denoising and enhancing audio files, making the subsequent transcription more accurate.

2.  **Speech-to-Text Transcription:**
    *   **Purpose:** To convert the processed audio recordings into text.
    *   **Tool:** `faster_whisper` is employed for this task. It is a robust and efficient speech recognition model that provides accurate transcriptions of the customer service calls.

3.  **LLM-based Analysis:**
    *   **Purpose:** To analyze the transcribed text to extract meaningful insights. This includes understanding customer sentiment, ensuring compliance with company policies, and verifying if callbacks were appropriately actioned.
    *   **Tools:** `ollama` is used as the framework to run local Large Language Models (LLMs). Specific models that can be leveraged include:
        *   `deepseek-r1:14b`: For complex analysis tasks requiring deep understanding.
        *   `gemma3:4b`: For more lightweight analysis and quicker processing.
    *   **Tasks:**
        *   **Sentiment Analysis:** Determining the emotional tone of the customer and the representative.
        *   **Compliance Checking:** Verifying if the representative adhered to regulatory guidelines and company scripts.
        *   **Callback Verification:** Confirming if promises of callbacks were made and whether they were fulfilled or need follow-up.

## Core Technologies

The project leverages the following key libraries and models:

*   **Audio Enhancement:**
    *   `resemble_enhance`: Used for preprocessing audio files to improve clarity and reduce noise, leading to better transcription accuracy.
*   **Speech-to-Text Transcription:**
    *   `faster_whisper`: A highly efficient and accurate speech recognition model used to convert audio conversations into text.
*   **LLM Interface:**
    *   `ollama`: A framework that facilitates the use of local Large Language Models, enabling offline analysis of transcribed text.
*   **Large Language Models (LLMs):**
    *   `deepseek-r1:14b`: A powerful LLM used for in-depth analysis of conversations, such as complex compliance checks and nuanced sentiment detection.
    *   `gemma3:4b`: A more lightweight LLM suitable for faster analysis tasks and general-purpose queries.

## Web Application Interface

This project includes a web application built with FastAPI that provides a user-friendly interface for interacting with the audio audit functionalities. Users can upload single audio files or ZIP archives containing multiple audio files directly through their web browser.

## Setup and Running the Web Application

Follow these steps to set up and run the web application on your local machine.

### 1. Dependencies

Ensure you have Python 3.8+ installed. The core dependencies for this project are listed in the `requirements.txt` file. These include:

*   **Core Application:**
    *   `fastapi`: For building the API.
    *   `uvicorn[standard]`: For running the FastAPI server.
    *   `python-multipart`: For handling file uploads.
    *   `ollama`: For interfacing with local LLMs.
    *   `torch` & `torchaudio`: For audio processing and machine learning tasks.
    *   `faster-whisper`: For speech-to-text transcription.
    *   `resemble-enhance`: For audio denoising and enhancement.
*   **Development & Testing (Optional, but good for contribution):**
    *   `pytest`: For running automated tests.
    *   `httpx`: For `TestClient` in FastAPI tests.
    *   `Jinja2`: Often a sub-dependency for web templating.

To install all necessary dependencies, navigate to the project's root directory in your terminal and run:
```bash
pip install -r requirements.txt
```

**Note on Heavy Libraries:** Libraries like `torch`, `torchaudio`, `faster-whisper`, and `resemble-enhance` can be large and have specific system dependencies (e.g., CUDA for GPU support). Ensure your environment is suitable for them. The application includes mechanisms to be importable even if these are not initially present (for API testing or if only using parts of the functionality), but core audio processing will fail.

**Ollama Setup:** This application uses `ollama` to run local Large Language Models. You need to have Ollama installed and running separately, with the models specified in `audit_processing.py` (e.g., `deepseek-r1:14b`, `gemma3:4b`) pulled and available. Refer to the [Ollama official website](https://ollama.com/) for installation instructions.

### 2. Running the Server

Once dependencies are installed, you can start the FastAPI application using Uvicorn:

```bash
uvicorn main:app --reload
```

*   `main:app` refers to the `app` instance of FastAPI in the `main.py` file.
*   `--reload` enables auto-reloading the server when code changes, which is useful for development. For production, you might run it without `--reload`.

The server will typically start on `http://127.0.0.1:8000`.

## Using the Web Interface

1.  **Access the Interface:** Open your web browser and navigate to `http://127.0.0.1:8000/`.
2.  **Upload Single Audio File:**
    *   Use the "Upload Single Audio File" form.
    *   Click "Choose File", select an audio file (e.g., .mp3, .wav).
    *   Click "Upload Audio".
3.  **Upload ZIP File:**
    *   Use the "Upload ZIP File" form.
    *   Click "Choose File", select a `.zip` archive containing your audio files.
    *   Click "Upload ZIP".
4.  **View Results:**
    *   A "Processing..." message will appear.
    *   Once processing is complete, the JSON results (or errors) will be displayed in the "Results" section.
    *   Any specific errors encountered during the process will be shown in the "Errors" section.

## API Endpoints

The FastAPI application exposes the following main API endpoints:

*   `GET /`: Serves the main HTML page for the web interface.
*   `POST /upload/audio/`: Accepts a single audio file for processing. Returns a JSON object with the audit results or error details.
*   `POST /upload/zip/`: Accepts a ZIP archive containing multiple audio files. Returns a JSON object summarizing the results for all processed files, including any errors per file or for the ZIP processing itself.
*   `GET /results/{task_id}/`: Currently a placeholder endpoint. As processing is synchronous, results are returned directly in the response to upload requests. This endpoint indicates this and is reserved for potential future asynchronous task tracking.

For detailed API documentation (Swagger UI or ReDoc), once the server is running, you can visit:
*   `http://127.0.0.1:8000/docs` (Swagger UI)
*   `http://127.0.0.1:8000/redoc` (ReDoc)
