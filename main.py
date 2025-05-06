# main.py

# ====== 🛠️ 手动配置区域（开发时可直接修改）=====
# INPUT_PATH = "example/zh.mp3"              # 输入路径（文件或目录）
INPUT_PATH = "example/zh.mp3"                # 默认输入路径
OUTPUT_DIR = ""                             # 输出目录（默认为当前目录）
LANGUAGE = "auto"                           # 默认语言
MODEL_TYPE = "SenseVoice"                   # 模型类型 (SenseVoice/Paraformer)
# ===================================================

import os
import argparse
import yaml
from utils.file_utils import get_audio_files, get_output_path, save_text_to_file
from asr_engine import ASREngine
from utils.logger import setup_logger
import logging
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# 尝试从local_config.py加载本地配置（不会被git追踪）
try:
    from local_config import LOCAL_INPUT_PATH
    INPUT_PATH = LOCAL_INPUT_PATH
except ImportError:
    # 如果没有local_config.py，使用默认值
    pass

def process_audio_files(asr, audio_files, output_dir=None, language="auto", time_format="sec", rich=None, use_postprocess=True):
    """
    对于每个音频文件，执行推理，并保存结果到txt文件

    Args:
        asr: ASREngine 实例
        audio_files: 音频文件列表
        output_dir: 输出目录
        language: 语言代码
        time_format: 时间格式（仅对Paraformer有效）
        rich: 是否使用富文本（仅对SenseVoice有效）
        use_postprocess: 是否使用自动选择的后处理方法
    """
    for audio in audio_files:
        try:
            logging.info(f"🔶Processing {audio}...")
            result = asr.run_model(
                audio, 
                language=language,
            )
            # debug
            print(result)
            print()
            print()
            print(rich_transcription_postprocess(result[0]["text"]))
            output_path = get_output_path(audio, output_dir)
            
            # 选择后处理方法
            if use_postprocess:
                # 使用自动选择的后处理方法
                text = asr.postprocess(result, time_format=time_format, rich=rich)
            else:
                # 根据模型类型手动选择后处理方法
                if asr.model_type == "Paraformer":
                    text = asr.postprocess_Paraformer(result, time_format=time_format)
                else:  # SenseVoice
                    text = asr.postprocess_general(result, rich=True if rich is None else rich)
                
            save_text_to_file(text, output_path)
            logging.info(f"✅ Saved to {output_path}")
        except Exception as e:
            logging.error(f"Error processing {audio}: {e}")

def main():
    # ====== 解析命令行参数（优先级高于手动配置）======
    parser = argparse.ArgumentParser(description="批量语音识别")
    parser.add_argument("--input", type=str, help="音频文件或目录")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--lang", type=str, default="auto", help="语言 (auto/zh/en/yue/ja/ko)")
    parser.add_argument("--model", type=str, default=None, help="模型类型 (SenseVoice/Paraformer)")
    parser.add_argument("--time_format", type=str, default="sec", choices=["sec", "min"], 
                        help="时间格式 (仅对Paraformer有效)")
    parser.add_argument("--rich", action="store_true", help="使用富文本 (仅对SenseVoice有效)")
    parser.add_argument("--no_rich", action="store_true", help="不使用富文本 (仅对SenseVoice有效)")
    parser.add_argument("--manual_postprocess", action="store_true", help="手动指定后处理方法而不是自动选择")
    args = parser.parse_args()

    # ====== 参数优先级：命令行 > 手动配置 ======
    input_path = args.input or INPUT_PATH
    output_dir = args.output_dir or OUTPUT_DIR
    language = args.lang or LANGUAGE
    model_type = args.model or MODEL_TYPE
    time_format = args.time_format
    use_postprocess = not args.manual_postprocess
    
    # 处理富文本参数
    rich = None
    if args.rich:
        rich = True
    elif args.no_rich:
        rich = False

    # ====== 初始化日志 & 加载配置 ======
    setup_logger(level=logging.INFO)
    logging.info(f"Starting ASR CLI with model: {model_type}")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        logging.info("config.yaml loaded successfully.")

    # ====== 初始化引擎 + 执行推理 ======
    asr = ASREngine(config, model_type=model_type)
    audio_files = get_audio_files(input_path)
    process_audio_files(asr, audio_files, output_dir, language, time_format, rich, use_postprocess)

if __name__ == "__main__":
    main()