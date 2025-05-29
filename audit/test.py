import sys
from faster_whisper import WhisperModel
import requests
import os
import time  # 新增导入 time 模块

# 记录开始时间
start_time = time.time()

# 获取命令行参数中的音频文件路径
if len(sys.argv) < 2:
    print("Usage: python transcribe_and_analyze.py <audio_file>")
    sys.exit(1)

audio_path = sys.argv[1]
if not os.path.exists(audio_path):
    print(f"音频文件不存在: {audio_path}")
    sys.exit(1)

# 加载 Faster Whisper 模型
print("加载模型中，请稍候...")
model = WhisperModel("medium", device="cpu", compute_type="int8")

# 执行转录
print("开始语音转文字...")
segments_generator, info = model.transcribe(audio_path)

# ✅ 转换为 list，防止生成器被多次消费后为空
segments = list(segments_generator)

print(f"\n识别语言：{info.language}（置信度：{info.language_probability:.2f}）")
print("\n转录文本内容：")
for segment in segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")

# 合并转录文本
full_transcription = " ".join([segment.text for segment in segments]).strip()

if not full_transcription:
    print("⚠️ 语音内容识别为空，请检查音频质量或格式是否正确。")
    sys.exit(1)

# 构造 Prompt（用于分析是否真实回访客户）
# prompt = f"""
# 你是一位联昊通速递智能质检分析助手，负责分析客服与客户之间的通话内容（由语音转写而来），判断客服是否进行了真实的客户回访。

# 请根据以下标准进行判断：
# 1. 是否有主动问候客户或表明身份；
# 2. 是否提及客户的具体情况（如订单、服务、投诉等）；
# 3. 是否包含实际交流互动，而非只是模板化独白；
# 4. 是否存在录音造假、复读、胡言乱语或空白无声片段；
# 5. 若文本过于混乱、杂音多或语义不清，请据此说明。

# 请返回以下结构化结果（JSON格式）：
# {{
#   "real_callback": true/false,
#   "reasons": ["原因1", "原因2", ...],
#   "summary": "一句话总结分析结论"
# }}

# 以下是对话文本：
# {full_transcription}
# """

# 构造 Prompt（先语义复原，再判断是否真实回访）
prompt = f"""
你是一位联昊通速递的智能质检分析助手，专门分析客服与客户的电话通话内容（以下文本来自语音转文字，可能存在部分识别错误）。

你的任务分为两个步骤：

### 第一步：语义复原
请先根据上下文和常识，对识别出的对话内容进行语义修复，使其尽可能还原为完整、通顺、真实的人类对话（注意不添加凭空信息，只做合理修正）。

### 第二步：判断是否为真实客户回访
请根据以下标准，判断是否为**真实有效的客户回访**：

1. 是否有客服主动问候或表明身份；
2. 是否提到客户的订单、服务、问题等个性化内容；
3. 是否包含双向互动，而非只读模板；
4. 是否存在异常，如：无交流、复读、胡言乱语、音频空白；
5. 若识别结果本身含混不清，也请说明。

请返回以下 JSON 格式结构化结果：
{{
  "real_callback": true/false,          // 是否为真实客户回访
  "reasons": ["原因1", "原因2", ...],    // 判断依据
  "summary": "一句话总结分析结论",       // 总结性陈述
  "restored_dialogue": "修复后的通话文本" // 修复后的语义清晰版本
}}

以下是识别出的对话原文（请先修复语义）：
{full_transcription}
"""


# 配置 Ollama 请求
ollama_url = "http://localhost:11434/api/generate"
payload = {
    "model": "gemma3:4b",  # 替换成你正在运行的本地模型名
    "prompt": prompt.strip(),
    "stream": False
}

print("\n正在调用 Ollama 分析...")

try:
    response = requests.post(ollama_url, json=payload)
    response.raise_for_status()
    result = response.json()
    print("\n🧠 Ollama 分析结果：")
    print(result.get("response", "⚠️ 未获取到有效分析结果"))
except requests.RequestException as e:
    print(f"\n请求 Ollama 时出错: {e}")
except ValueError as e:
    print(f"\n解析 Ollama 响应时出错: {e}")

# 记录结束时间并计算总耗时
end_time = time.time()
total_time = end_time - start_time
print(f"\n脚本执行完成，总耗时: {total_time:.2f} 秒")