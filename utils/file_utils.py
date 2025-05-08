# utils/file_utils.py
import os
import logging

def get_audio_files(input_path):
    """
    获取音频文件列表
    - input_path: 音频文件路径或目录
    - 返回音频文件列表
    """
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        return [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith((".wav", ".mp3", ".aac", ".flac"))
        ]
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

def get_output_path(audio_path, output_dir=None, model_type=None):
    """
    - audio_path: 音频文件路径
    - 默认输出路径为当前目录下的同名txt
    - 如果指定了输出目录，则在该目录下创建同名txt
    - model_type: 模型类型 (SenseVoiceSmall 或 Paraformer 或 SenseVoiceSmall+Paraformer)，将添加到文件名中
    """
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # Add model type to filename if provided
    if model_type:
        output_file = f"{base_name}_{model_type}.txt"
    else:
        output_file = f"{base_name}.txt"
        
    if output_dir:
        return os.path.join(output_dir, output_file)
    else:
        return os.path.join(os.path.dirname(audio_path), output_file)
    
def save_text_to_file(text, file_path="output.txt"):
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text)
        logging.info(f"Text successfully saved to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save text to {file_path}: {e}")