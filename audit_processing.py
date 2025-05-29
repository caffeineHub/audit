import json
import os
import subprocess
import ollama # Ollama is usually lightweight enough to import
import tempfile
import uuid

# --- Conditional Imports for Heavy Libraries & Global Placeholders ---
torch = None
torchaudio = None
WhisperModel = None  # Placeholder for the class from faster_whisper
denoise_func = None       # Placeholder for denoise function from resemble_enhance
enhance_func = None       # Placeholder for enhance function from resemble_enhance
DEVICE = "cpu"       # Default device
WHISPER_MODEL_INSTANCE = None  # Placeholder for the initialized WhisperModel instance
OLLAMA_MODEL_NAME = "deepseek-r1:14b"

_heavy_libs_successfully_imported = False

try:
    # Attempt to import heavy libraries
    import torch
    import torchaudio
    from faster_whisper import WhisperModel as FWModel # Use an alias
    from resemble_enhance.enhancer.inference import denoise as res_denoise, enhance as res_enhance

    # If imports succeed, assign them to global names
    WhisperModel = FWModel # Now WhisperModel refers to the class
    denoise_func = res_denoise
    enhance_func = res_enhance
    _heavy_libs_successfully_imported = True
    print("Successfully imported heavy audio library modules (torch, torchaudio, faster_whisper, resemble_enhance).")

    # Determine device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device set to: {DEVICE}")

    # Attempt to initialize WHISPER_MODEL_INSTANCE
    try:
        if WhisperModel is not None: # Ensure the class was actually imported
            WHISPER_MODEL_INSTANCE = WhisperModel("medium", device=DEVICE, compute_type="int8" if DEVICE == "cpu" else "float16")
            print(f"WHISPER_MODEL_INSTANCE initialized: {WHISPER_MODEL_INSTANCE is not None}")
        else:
            print("WhisperModel class not available, skipping WHISPER_MODEL_INSTANCE initialization.")
            WHISPER_MODEL_INSTANCE = None
    except Exception as e:
        print(f"Warning: Failed to initialize WHISPER_MODEL_INSTANCE globally: {e}. "
              "This is expected if model files are not available. "
              "Functions relying on it will fail if not mocked during tests.")
        WHISPER_MODEL_INSTANCE = None # Ensure it's None if initialization fails

except ImportError as e:
    print(f"Warning: Failed to import one or more heavy audio libraries: {e}. "
          "Audio processing features requiring these libraries will not be available. "
          "Ensure these are mocked for tests if you intend to test functions from audit_processing.py directly.")
    # Explicitly set all to None if any critical import fails
    torch = None
    torchaudio = None
    WhisperModel = None
    denoise_func = None
    enhance_func = None
    WHISPER_MODEL_INSTANCE = None

# --- Core Functions ---

def preprocess_audio(input_audio_path: str, output_dir: str) -> str:
    """
    Preprocesses an audio file by converting to 16kHz mono WAV,
    denoising, and enhancing.
    """
    if not _heavy_libs_successfully_imported or not torch or not torchaudio or not denoise_func or not enhance_func:
        raise ImportError("Core audio processing libraries (torch, torchaudio, resemble_enhance) are not available for preprocess_audio.")

    if not os.path.exists(input_audio_path):
        raise FileNotFoundError(f"Input file {input_audio_path} does not exist.")

    os.makedirs(output_dir, exist_ok=True)
    base, ext = os.path.splitext(os.path.basename(input_audio_path))
    enhanced_filename = f"{base}_enhanced_{uuid.uuid4().hex[:8]}.wav"
    output_enhanced_path = os.path.join(output_dir, enhanced_filename)
    temp_wav_file = None

    try:
        current_audio_path = input_audio_path
        if ext.lower() != ".wav":
            temp_wav_path = os.path.join(output_dir, f"{base}_temp_conversion_{uuid.uuid4().hex[:8]}.wav")
            temp_wav_file = temp_wav_path
            subprocess.run(
                ["ffmpeg", "-y", "-i", input_audio_path, "-ar", "16000", "-ac", "1", temp_wav_path],
                check=True, capture_output=True
            )
            current_audio_path = temp_wav_path

        dwav, sr = torchaudio.load(current_audio_path)
        dwav = dwav.mean(dim=0)

        wav_denoised, sr_denoised = denoise_func(dwav, sr, DEVICE)
        wav_enhanced, sr_enhanced = enhance_func(
            wav_denoised, sr_denoised, DEVICE, nfe=64, solver="midpoint", lambd=0.1, tau=0.5
        )

        if torch.max(torch.abs(wav_enhanced)) < 1e-5:
            raise ValueError("Enhanced audio is almost silent, processing likely failed.")

        torchaudio.save(output_enhanced_path, wav_enhanced.unsqueeze(0).to(torch.float32), sr_enhanced)
        return output_enhanced_path
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
        raise
    finally:
        if temp_wav_file and os.path.exists(temp_wav_file):
            try:
                os.remove(temp_wav_file)
            except OSError as e:
                print(f"Error removing temporary file {temp_wav_file}: {e}")


def transcribe_audio(audio_file_path: str, temp_processing_dir: str) -> str:
    """
    Preprocesses and transcribes an audio file.
    """
    if not _heavy_libs_successfully_imported or WhisperModel is None:
        raise ImportError("faster_whisper.WhisperModel class not imported. Cannot transcribe.")
    
    if WHISPER_MODEL_INSTANCE is None:
        raise ImportError("WHISPER_MODEL_INSTANCE is not initialized. Cannot transcribe.")

    enhanced_file_path = None
    try:
        enhanced_file_path = preprocess_audio(audio_file_path, temp_processing_dir)
        segments, info = WHISPER_MODEL_INSTANCE.transcribe(
            enhanced_file_path, language="zh", vad_filter=True
        )
        print(f"Detected language: {info.language}, probability: {info.language_probability:.2f}")
        transcription = "".join([seg.text for seg in segments])
        return transcription
    except Exception as e:
        print(f"Error during transcription of {audio_file_path}: {e}")
        raise
    finally:
        if enhanced_file_path and os.path.exists(enhanced_file_path):
            try:
                os.remove(enhanced_file_path)
            except OSError as e:
                print(f"Error removing temporary enhanced file {enhanced_file_path}: {e}")


def analyze_text(text: str, ollama_model: str = OLLAMA_MODEL_NAME) -> dict:
    """
    Analyzes transcribed text for sentiment and compliance using an Ollama model.
    """
    if ollama is None:
        raise ImportError("Ollama library is not available.")
        
    prompt = f"""
你是一个客服录音审核助手。分析以下中文客服对话文本，可能包含转录错误或不完整内容。完成以下任务：
1. 情感分析：判断对话的情感倾向（正面、负面或中性）。若文本不完整，基于上下文推断。
2. 合规性检查：检测是否存在不当用语（如“辱骂”“威胁”）或违规行为（如“违反政策”）。
3. 提供简短的分析摘要，说明情感和合规性结论。

对话文本：
{text}

输出格式（JSON）：
{{
    "sentiment": "正面/负面/中性",
    "compliance_issues": ["问题1", "问题2", ...] 或 [],
    "summary": "分析摘要"
}}
"""
    try:
        response = ollama.generate(model=ollama_model, prompt=prompt, stream=False)
        if "response" not in response or not isinstance(response["response"], str):
            return {
                "sentiment": "未知", "compliance_issues": ["LLM response format error"],
                "summary": f"Ollama returned an unexpected response structure: {text[:100]}..."
            }
        analysis_result = json.loads(response["response"])
        return analysis_result
    except json.JSONDecodeError as e:
        raw_resp = response.get('response', '')[:100] if isinstance(response, dict) else str(response)[:100]
        return {
            "sentiment": "未知", "compliance_issues": [f"JSON parsing error: {str(e)}"],
            "summary": f"无法解析LLM输出。Raw: {raw_resp}..."
        }
    except Exception as e:
        return {
            "sentiment": "未知", "compliance_issues": [f"Ollama request failed: {str(e)}"],
            "summary": "LLM分析请求失败。"
        }

def perform_full_audio_audit(audio_file_path: str) -> dict:
    """
    Performs a full audit of an audio file: preprocess, transcribe, and analyze.
    """
    with tempfile.TemporaryDirectory(prefix="audio_audit_") as temp_dir:
        try:
            # Ensure heavy libraries are checked at the start of the actual processing path
            # if perform_full_audio_audit is called directly without mocks.
            if not _heavy_libs_successfully_imported or WHISPER_MODEL_INSTANCE is None:
                 raise ImportError("Core audio processing components are not available for a full audit.")

            transcription = transcribe_audio(audio_file_path, temp_processing_dir=temp_dir)
            if not transcription.strip():
                return {
                    "audio_file": os.path.basename(audio_file_path), "transcription": "",
                    "sentiment": "未知", "compliance_issues": ["Empty transcription"],
                    "summary": "音频转录结果为空，无法进行分析。"
                }
            analysis = analyze_text(transcription)
            analysis["audio_file"] = os.path.basename(audio_file_path)
            analysis["transcription"] = transcription
            return analysis
        except FileNotFoundError:
            return {"audio_file": os.path.basename(audio_file_path), "error": "Input audio file not found.", "status": "FAILED"}
        except ValueError as ve:
             return {"audio_file": os.path.basename(audio_file_path), "error": f"Processing error: {str(ve)}", "status": "FAILED"}
        except subprocess.CalledProcessError as cpe:
            err_msg = cpe.stderr.decode() if cpe.stderr else str(cpe)
            return {"audio_file": os.path.basename(audio_file_path), "error": f"Audio conversion failed: {err_msg}", "status": "FAILED"}
        except ImportError as ie: # Catch ImportErrors from transcribe_audio/preprocess_audio
             return {"audio_file": os.path.basename(audio_file_path), "error": f"Import error during processing: {str(ie)}", "status": "FAILED"}
        except Exception as e:
            return {"audio_file": os.path.basename(audio_file_path), "error": f"An unexpected error occurred: {str(e)}", "status": "FAILED"}

if __name__ == '__main__':
    print(f"Running audit_processing.py directly (Device: {DEVICE}, Heavy Libs Imported: {_heavy_libs_successfully_imported}, Whisper Instance: {WHISPER_MODEL_INSTANCE is not None})...")
    # Simplified __main__ for basic check, actual audio processing would likely fail if libs are missing
    if _heavy_libs_successfully_imported and WHISPER_MODEL_INSTANCE:
        print("Core libraries and Whisper model appear to be initialized.")
        # Add test calls here if you have sample files and expect full functionality
    else:
        print("Core libraries or Whisper model not fully initialized. Full functionality tests in __main__ might fail or be skipped.")

    test_text_sample = "客服：您好，请问有什么可以帮您？客户：我的订单一直没有收到，你们怎么搞的！"
    try:
        analysis_only = analyze_text(test_text_sample)
        print("\nDirect Text Analysis Result (JSON):")
        print(json.dumps(analysis_only, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error during direct text analysis test: {e}")

    print("\n--- audit_processing.py direct execution finished ---")
