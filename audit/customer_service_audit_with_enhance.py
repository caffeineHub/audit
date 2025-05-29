import json
import torch
import torchaudio
import os
import subprocess
from faster_whisper import WhisperModel
from resemble_enhance.enhancer.inference import denoise, enhance
import ollama

# 设置设备（建议 GPU 环境使用 "cuda" 或 "mps"）
device = "cpu"

# 音频预处理（去噪 + 增强）
def preprocess_audio(audio_file, output_file="enhanced_audio.wav"):
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Input file {audio_file} does not exist.")

    # 转为 16kHz 单声道 wav 格式
    if not audio_file.endswith(".wav"):
        wav_file = "temp_input.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_file, "-ar", "16000", "-ac", "1", wav_file
        ], check=True)
    else:
        wav_file = audio_file

    # 加载音频并去噪+增强
    dwav, sr = torchaudio.load(wav_file)
    dwav = dwav.mean(dim=0)  # 转为单声道

    wav_denoised, new_sr = denoise(dwav, sr, device)
    torchaudio.save("denoised.wav", wav_denoised.unsqueeze(0).to(torch.float32), new_sr)

    wav_enhanced, new_sr = enhance(
        wav_denoised, new_sr, device, nfe=64, solver="midpoint", lambd=0.1, tau=0.5
    )

    # 检查是否为静音（增强失败）
    if torch.max(torch.abs(wav_enhanced)) < 1e-5:
        raise ValueError("增强后的音频几乎为静音，可能处理失败。")

    # 保存增强后音频，确保 float32 格式
    torchaudio.save(output_file, wav_enhanced.unsqueeze(0).to(torch.float32), new_sr)
    return output_file

# Whisper 模型初始化
whisper_model = WhisperModel("medium", device=device, compute_type="int8")

# 音频转文字
def transcribe_audio(audio_file):
    enhanced_file = preprocess_audio(audio_file)
    segments, info = whisper_model.transcribe(
        enhanced_file, language="zh", vad_filter=True
    )
    print(f"Detected language: {info.language}, probability: {info.language_probability:.2f}")
    
    # 可视化每个语音段落
    for seg in segments:
        print(f"[{seg.start:.2f} - {seg.end:.2f}]: {seg.text}")

    text = "".join([seg.text for seg in segments])
    return text

# Ollama 模型设定
OLLAMA_MODEL = "deepseek-r1:14b"

# 对文本进行情感和合规性分析
def analyze_text(text):
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
    response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)
    try:
        result = json.loads(response["response"])
    except json.JSONDecodeError:
        result = {
            "sentiment": "未知",
            "compliance_issues": [],
            "summary": f"无法解析LLM输出，可能因文本不完整：{text[:50]}..."
        }
    return result

# 主函数：传入音频或直接测试文本
def main(audio_file=None, test_text=None):
    if test_text:
        print(f"Input Text: {test_text}")
        analysis = analyze_text(test_text)
        print("Analysis Result:")
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
    elif audio_file:
        print(f"Processing audio: {audio_file}")
        transcription = transcribe_audio(audio_file)
        print(f"Transcription: {transcription}")
        analysis = analyze_text(transcription)
        print("Analysis Result:")
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
    else:
        print("Please provide an audio file or test text.")

# 示例入口
if __name__ == "__main__":
    # 可直接测试文本分析
    # main(test_text="客户：你们服务怎么这么差？我投诉你！")

    # 测试音频文件（input.mp3 需替换为你的实际文件）
    audio_file = "input.mp3"
    main(audio_file=audio_file)
