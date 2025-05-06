# webui.py

import os
import re
import base64
import io
import gradio as gr
import numpy as np
import torch
import torchaudio
import tempfile
import librosa
from functools import lru_cache

# 导入自定义模块
from asr_engine import ASREngine
import yaml
from utils.logger import setup_logger
import logging
setup_logger(level=logging.INFO)

# 加载配置
with open("config.yaml", "r", encoding="utf-8") as f:
    try:
        config = yaml.safe_load(f)
        logging.info("config.yaml loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")

# 初始化 ASR 引擎
asr_engine = ASREngine(config)

# Emoji 映射表
emo_dict = {
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
}

event_dict = {
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|Cry|>": "😭",
    "<|Sneeze|>": " sneeze sound>",
    "<|Breath|>": "",
    "<|Cough|>": "😷",
}

emoji_dict = {k: v for d in [emo_dict, event_dict] for k, v in d.items()}

emo_set = set(emo_dict.values())
event_set = set(event_dict.values())

def generate_text_file(text, audio_path):
    """
    将识别结果保存为文本文件，文件名与音频文件名相同
    """
    if not text or not audio_path:
        return None

    # 获取原始文件名（不含扩展名）并加上 _asr.txt
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_filename = f"{base_name}.txt"

    # 创建临时目录保存文件
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, output_filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

    return file_path

def save_transcript(text, filename):
    if not text or text.strip() == "":
        return None

    # 如果没有扩展名，则默认加 _asr.txt
    if not filename.endswith(".txt"):
        filename = f"{filename}_asr.txt"

    # 创建临时目录保存文件
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, filename)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        return file_path
    except Exception as e:
        logging.error(f"Failed to save transcript: {e}")
        return None

def load_audio_from_path(file_path, target_sr=16000):
    """
    使用 librosa 加载音频文件，并返回采样率和 NumPy 数组
    """
    try:
        audio_np, fs = librosa.load(file_path, sr=None)  # 保留原始采样率
        if len(audio_np.shape) > 1:
            audio_np = np.mean(audio_np, axis=1)  # 转为单声道
        return fs, audio_np
    except Exception as e:
        logging.error(f"Failed to load audio from {file_path}: {e}")
        raise

# def format_str_v3(s):
#     def get_emo(s):
#         return s[-1] if s[-1] in emo_set else None
#     def get_event(s):
#         return s[0] if s[0] in event_set else None

#     s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
#     for lang in ["<|zh|>", "<|en|>", "<|yue|>", "<|ja|>", "<|ko|>", "<|nospeech|>"]:
#         s = s.replace(lang, "<|lang|>")
#     s_list = [format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]
#     new_s = " " + s_list[0]
#     cur_ent_event = get_event(new_s)
#     for i in range(1, len(s_list)):
#         if len(s_list[i]) == 0:
#             continue
#         if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) != None:
#             s_list[i] = s_list[i][1:]
#         cur_ent_event = get_event(s_list[i])
#         if get_emo(s_list[i]) != None and get_emo(s_list[i]) == get_emo(new_s):
#             new_s = new_s[:-1]
#         new_s += s_list[i].strip().lstrip()
#     new_s = new_s.replace("The.", " ")
#     return new_s.strip()

def format_str_v3(s):
    def get_emo(s):
        return s[-1] if s[-1] in emo_set else None
    def get_event(s):
        return s[0] if s[0] in event_set else None

    # 删除语言标签
    for lang in ["<|zh|>", "<|en|>", "<|yue|>", "<|ja|>", "<|ko|>", "<|nospeech|>"]:
        s = s.replace(lang, "")

    s = s.replace("<|nospeech|><|Event_UNK|>", "❓")

    s_list = [format_str_v2(s_i).strip() for s_i in s.split("<|lang|>")]  # 不要 strip 太狠
    new_s = s_list[0]
    cur_ent_event = get_event(new_s)

    for i in range(1, len(s_list)):
        if not s_list[i].strip():
            continue
        current_event = get_event(s_list[i])
        if current_event == cur_ent_event and current_event is not None:
            s_list[i] = s_list[i][1:]  # 去掉开头 event emoji
        cur_ent_event = current_event
        if get_emo(s_list[i]) != get_emo(new_s):
            new_s += " "
        new_s += s_list[i]

    new_s = re.sub(r'\.([a-zA-Z])', r'. \1', new_s)  # 自动加句号后空格
    new_s = re.sub(r'([a-zA-Z])([\.!?])', r'\1\2 ', new_s)  # 自动加句号前空格
    new_s = re.sub(r'\s+', ' ', new_s).strip()

    logging.info(f"[DEBUG] Final formatted text: {new_s}")
    return new_s


# def format_str_v2(s):
#     sptk_dict = {}
#     for sptk in emoji_dict:
#         sptk_dict[sptk] = s.count(sptk)
#         s = s.replace(sptk, "")
#     emo = "<|NEUTRAL|>"
#     for e in emo_dict:
#         if sptk_dict[e] > sptk_dict[emo]:
#             emo = e
#     for e in event_dict:
#         if sptk_dict[e] > 0:
#             s = event_dict[e] + s
#     s = s + emo_dict[emo]

#     for emoji in emo_set.union(event_set):
#         s = s.replace(" " + emoji, emoji)
#         s = s.replace(emoji + " ", emoji)
#     return s.strip()


def format_str_v2(s):
    sptk_dict = {}
    for sptk in emoji_dict:
        sptk_dict[sptk] = s.count(sptk)
        s = s.replace(sptk, " ")  # 替换为空格而不是空字符串

    emo = "<|NEUTRAL|>"
    for e in emo_dict:
        if sptk_dict[e] > sptk_dict[emo]:
            emo = e
    for e in event_dict:
        if sptk_dict[e] > 0:
            s = event_dict[e] + " " + s  # 加个空格避免粘连

    s = s + " " + emo_dict[emo]

    # 避免多个空格连在一起
    for emoji in emo_set.union(event_set):
        s = s.replace(" " + emoji, emoji)
        s = s.replace(emoji + " ", emoji)

    return s.strip()

@lru_cache(maxsize=5)  # 缓存最近处理过的采样率转换器
def _get_resampler(orig_freq, target_freq):
    return torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=target_freq)

def preprocess_audio(wav_input, fs=16000):
    """
    将 Gradio 的 Audio 输入预处理为标准格式 (numpy array, 16kHz)
    """
    if isinstance(wav_input, tuple):
        fs, audio_np = wav_input
    else:
        audio_np = wav_input

    if not isinstance(audio_np, np.ndarray):
        raise ValueError("Audio input must be a numpy array")

    # 转换为 float32 并归一化
    audio_np = audio_np.astype(np.float32) / np.iinfo(np.int16).max

    # 如果是多通道，转为单通道
    if len(audio_np.shape) > 1:
        audio_np = np.mean(audio_np, axis=-1)

    # 如果不是 16kHz，进行重采样
    if fs != 16000:
        resampler = _get_resampler(fs, 16000)
        audio_torch = torch.from_numpy(audio_np).to(torch.float32)
        audio_np = resampler(audio_torch[None, :])[0, :].numpy()

    return audio_np

def model_inference(wav_input, language):
    try:
        logging.info("Received audio input for inference.")

        # 提取文件路径或 NumPy 数组
        if isinstance(wav_input, dict) and "name" in wav_input:
            audio_path = wav_input["name"]
            audio_for_model = audio_path
        elif isinstance(wav_input, str):
            audio_path = wav_input
            audio_for_model = wav_input
        elif isinstance(wav_input, tuple):
            fs, audio_np = wav_input
            audio_path = "microphone_recording"
            audio_for_model = audio_np
        else:
            audio_path = "unknown"
            audio_for_model = wav_input

        # 执行 ASR 推理
        text = asr_engine.run_inference(
            audio_input=audio_for_model,
            language=language,
            use_itn=True,
            batch_size_s=60,
            merge_vad=True
        )

        logging.info(f"🔶RAW ASR result: {text}")
        text = format_str_v3(text)
        logging.info(f"✋Formatted ASR result: {text}")

        # 自动生成 txt 文件
        file_path = generate_text_file(text, audio_path)

        return text, file_path  # 返回识别结果和文件路径
    except Exception as e:
        logging.error(f"Inference error: {str(e)}")
        return f"Error during inference: {str(e)}", None

# 示例音频路径
audio_examples = [
    ["example/zh.mp3", "zh"],
    ["example/yue.mp3", "yue"],
    ["example/en.mp3", "en"],
    ["example/ja.mp3", "ja"],
    ["example/ko.mp3", "ko"],
    ["example/emo_1.wav", "auto"],
    ["example/emo_2.wav", "auto"],
    ["example/emo_3.wav", "auto"],
]

# HTML 描述
html_content = """
<div>
    <h2 style="font-size: 22px;margin-left: 0px;">Voice Understanding Model: SenseVoice-Small</h2>
    <p style="font-size: 18px;margin-left: 20px;">SenseVoice-Small is an encoder-only speech foundation model designed for rapid voice understanding. It encompasses a variety of features including automatic speech recognition (ASR), spoken language identification (LID), speech emotion recognition (SER), and acoustic event detection (AED).</p>
    <h2 style="font-size: 22px;margin-left: 0px;">Usage</h2> 
    <p style="font-size: 18px;margin-left: 20px;">Upload an audio file or use the microphone. Select the language and click Start.</p>
    <p style="font-size: 18px;margin-left: 20px;">The output includes emojis for emotions 😊 and events 🎼.</p>
    <h2 style="font-size: 22px;margin-left: 0px;">Repo</h2>
    <ul>
        <li><a href="https://github.com/FunAudioLLM/SenseVoice" target="_blank">SenseVoice</a></li>
        <li><a href="https://github.com/modelscope/FunASR" target="_blank">FunASR</a></li>
        <li><a href="https://github.com/FunAudioLLM/CosyVoice" target="_blank">CosyVoice</a></li>
    </ul>
</div>
"""


def launch(inbrowser=False):
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.HTML(html_content)
        with gr.Row():
            with gr.Column():
                audio_inputs = gr.Audio(label="Upload or record audio", type="filepath")
                with gr.Accordion("Configuration"):
                    language_inputs = gr.Dropdown(
                        choices=["auto", "zh", "en", "yue", "ja", "ko", "nospeech"],
                        value="auto",
                        label="Language"
                    )
                fn_button = gr.Button("Start", variant="primary")
                text_outputs = gr.Textbox(label="Transcription Result", lines=10)
                file_output = gr.File(label="Download Text File")

            gr.Examples(
                examples=audio_examples,
                inputs=[audio_inputs, language_inputs],
                label="Examples",
                examples_per_page=20
            )

        # 同时更新文本和文件输出
        fn_button.click(
            fn=model_inference,
            inputs=[audio_inputs, language_inputs],
            outputs=[text_outputs, file_output]
        )

    demo.launch(share=False, inbrowser=inbrowser)


if __name__ == "__main__":
    launch(inbrowser=True)