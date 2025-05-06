# https://github.com/modelscope/FunASR/issues/2072#issuecomment-2577108214


from funasr import AutoModel
from typing import List, Dict
import yaml
import logging
from utils.logger import setup_logger
from utils.file_utils import get_audio_files, get_output_path
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import json


# 默认音频路径
audio_path = "example/en.mp3"

# 尝试从local_config.py加载本地配置（不会被git追踪）
try:
    from local_config import LOCAL_DEMO_SPEAKER_PATH
    audio_path = LOCAL_DEMO_SPEAKER_PATH
except ImportError:
    # 如果没有local_config.py或没有定义LOCAL_DEMO_SPEAKER_PATH，使用默认值
    pass

# ====== 初始化日志 & 加载配置 ======
setup_logger(level=logging.INFO)
logging.info("Starting ASR CLI processing...")

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    logging.info("config.yaml loaded successfully.")


def format_recognition_result(res: List[Dict]) -> str:
    formatted_output = []
    for result in res:
        sentences = result["sentence_info"]
    # formatted_output.append("语音识别结果：\n")
    for sentence in sentences:
        speaker_id = sentence["spk"]
        text = sentence["text"]
        start_time = sentence["start"] / 1000 
        end_time = sentence["end"] / 1000

        formatted_sentence = (
            f"S-{speaker_id} "
            f"[{start_time:.2f}s - {end_time:.2f}s]: "
            f"{text}"
        )
        formatted_output.append(formatted_sentence)
    return "\n".join(formatted_output)

model = AutoModel(
    model=config["paraformer_path_2"],
    vad_model=config["vad_model_path"],
    punc_model=config["punc_model_path"],
    spk_model=config["spk_model_path"],
    disable_update=True,
)

res = model.generate(
    input=audio_path,
    batch_size_s=300,
)

print(format_recognition_result(res))


text = rich_transcription_postprocess(res[0]["text"])
print(text)

# print(res)


# output_path = get_output_path(audio_path, None)
# with open(output_path, "w", encoding="utf-8") as f:
#     f.write(res)
# logging.info(f"Saved to {output_path}")