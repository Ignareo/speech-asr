from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import os

# model_dir = "FunAudioLLM/SenseVoiceSmall"
model_dir = os.path.dirname(os.path.abspath(__file__))

# 默认音频路径
default_audio_path = "example/en.mp3"

# 尝试从local_config.py加载本地配置（不会被git追踪）
try:
    from local_config import LOCAL_DEMO_PATH
    audio_path = LOCAL_DEMO_PATH
except ImportError:
    # 如果没有local_config.py或没有定义LOCAL_DEMO_PATH，使用默认值
    audio_path = f"{model_dir}/{default_audio_path}"

# AutoModel 定义
model = AutoModel(
    model=model_dir,            # 本地磁盘中的模型路径
    vad_model="fsmn-vad",       # 将长音频切割成短音频
    vad_kwargs={"max_single_segment_time": 30000},  # vad_model 最大切割音频时长
    device="cuda:0",
    hub="hf",
    disable_update=True,
)

# AutoModel 推理
res = model.generate(
    input=audio_path,
    cache={},
    language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,
    merge_length_s=15,
    # output_dir="output",  # 输出原始信息的目录，默认None不启用
)

def save_text_to_file(text, file_path="output.txt"):
    """
    Save the given text to a .txt file and provide feedback.

    Args:
        text (str): The text to save.
        file_path (str): The path to the output file. Defaults to 'output.txt'.

    Returns:
        None
    """
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text)
        print(f"Text successfully saved to {file_path}")
    except Exception as e:
        print(f"Failed to save text to {file_path}: {e}")


text = rich_transcription_postprocess(res[0]["text"])
print(text)
# save_text_to_file(text, "fangfang.txt")

