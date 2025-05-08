# asr_engine.py
# 加载模型
# 执行推理
# 后处理文本输出

import os
import logging
from typing import List, Dict
from utils.logger import Colors
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


class ASREngine:
    def __init__(self, config, model_type="SenseVoice"):
        self.config = config
        self.model_type = model_type
        logging.info(f"{Colors.CYAN}正在加载 {model_type} 模型...{Colors.RESET}")
        self.model = self._load_model()
        logging.info(f"{Colors.GREEN}模型加载完成{Colors.RESET}")

    def _load_model(self):
        """
        加载模型，根据 model_type 参数选择加载不同的模型
        - SenseVoice: 语音识别 (效果好, 带情感检测、emoji)
        - Paraformer: 语音识别 (带时间戳，说话人分离)
            - 增加了punc_model和spk_model
        - Whisper: 语音识别 (中文识别巨差无比)(已删除)
        """
        # 设置是否禁用FunASR内部进度条
        disable_pbar = self.config.get("disable_pbar", False)
        
        if self.model_type == "Paraformer":
            logging.debug(f"加载Paraformer模型，路径: {self.config['Paraformer_path']}")
            model = AutoModel(
                model=self.config["Paraformer_path"],
                punc_model=self.config["punc_model_path"],  # 符号预测模型
                spk_model=self.config["spk_model_path"],    # 说话人分离模型
                vad_model=self.config["vad_model_path"],
                vad_kwargs=self.config.get("vad_kwargs", {"max_single_segment_time": 30000}),
                device=self.config.get("device", "cpu"),
                hub=self.config.get("hub", "hf"),
                disable_update=self.config.get("disable_update", True),
                disable_pbar=disable_pbar,  # 禁用FunASR内部进度条
            )
        else:  # 默认为 SenseVoice
            logging.debug(f"加载SenseVoice模型，路径: {self.config['SenseVoice_path']}")
            model = AutoModel(
                model=self.config["SenseVoice_path"],
                vad_model=self.config["vad_model_path"],
                vad_kwargs=self.config.get("vad_kwargs", {"max_single_segment_time": 30000}),
                # punc_model=self.config["punc_model_path"],  # 符号预测模型
                device=self.config.get("device", "cpu"),
                # hub=self.config.get("hub", "hf"),
                disable_update=self.config.get("disable_update", True),
                disable_pbar=disable_pbar,  # 禁用FunASR内部进度条
            )
        return model

    def run_model(self, audio_input, language="auto", batch_size_s=60, disable_pbar=None, **kwargs):
        """
        执行推理
        
        Args:
            audio_input: 音频文件路径
            language: 语言(auto/zh/en/yue/ja/ko)
            batch_size_s: 批处理大小（秒）
            disable_pbar: 是否禁用进度条（如果为None则使用模型默认值）
            **kwargs: 其他参数传递给模型
        """
        logging.debug(f"开始处理音频: {audio_input}")
        logging.debug(f"使用语言: {language}, 批处理大小: {batch_size_s}秒")
        
        # 合并所有参数
        params = {
            "input": audio_input,
            "use_itn": True,       # 输出结果中是否包含标点与逆文本正则化
            "language": language,
            "batch_size_s": batch_size_s,  # 采用动态batch，batch中总音频时长，单位为秒s
            "merge_length_s": 15,
            # ban_emo_unk=True,     # 禁用emo_unk标签，禁用后所有的句子都会被赋与情感标签
        }
        
        # 如果指定了disable_pbar，添加到参数中
        if disable_pbar is not None:
            params["disable_pbar"] = disable_pbar
            
        # 添加其他参数
        params.update(kwargs)
            
        result = self.model.generate(**params)
        
        logging.debug(f"音频处理完成，结果包含 {len(result)} 个部分")
        return result
    
    def postprocess(self, result, time_format="sec", rich=None):
        """
        根据模型类型选择合适的后处理方法
        
        Args:
            result: 识别结果
            time_format: 时间格式，可选值为 "sec"（仅显示秒）或 "min"（显示分钟和秒）
            rich: 是否使用富文本（如果为None，则根据模型类型自动选择）
        """
        logging.debug(f"使用模型 {self.model_type} 的后处理方法")
        
        if self.model_type == "Paraformer":
            return self.postprocess_Paraformer(result, time_format=time_format)
        else:  # 默认为 SenseVoice
            # 如果明确指定了rich参数，则使用指定的值
            if rich is not None:
                return self.postprocess_general(result, rich=rich)
            # 否则使用默认值（SenseVoice默认使用rich=True）
            return self.postprocess_general(result, rich=True)
    
    def postprocess_general(self, result, rich=True):
        """
        通用后处理函数
        
        Args:
            result: 识别结果
            rich: 是否使用富文本，否则仅提取文本
        
        Note:
            富文本仅对 SenseVoice 有效
        """
        logging.debug(f"使用通用后处理，富文本: {rich}")
        
        if rich and self.model_type == "SenseVoice":
            text = rich_transcription_postprocess(result[0]["text"])
        else:
            text = result[0]["text"]
        return text
    
    def postprocess_Paraformer(self, res: List[Dict], time_format: str = "sec") -> str:
        """
        Paraformer的后处理函数
        - 分离说话人
        - 添加时间戳
        
        Args:
            res: 识别结果
            time_format: 时间格式，可选值为 "sec"（仅显示秒）或 "min"（显示分钟和秒）
        """
        logging.debug(f"使用Paraformer后处理，时间格式: {time_format}")
        
        formatted_output = []
        for result in res:
            if "sentence_info" not in result:
                continue
                
            sentences = result["sentence_info"]
            logging.debug(f"处理 {len(sentences)} 个句子片段")
            
            for sentence in sentences:
                speaker_id = sentence["spk"]
                text = sentence["text"]
                start_time = sentence["start"] / 1000 
                end_time = sentence["end"] / 1000

                if time_format == "min":
                    # 转换为分钟和秒的格式
                    start_min, start_sec = divmod(start_time, 60)
                    end_min, end_sec = divmod(end_time, 60)
                    time_str = f"[{int(start_min)}min{start_sec:.2f}s - {int(end_min)}min{end_sec:.2f}s]"
                else:
                    # 默认秒格式
                    time_str = f"[{start_time:.2f}s - {end_time:.2f}s]"

                formatted_sentence = (
                    f"S-{speaker_id} "
                    f"{time_str}: "
                    f"{text}"
                )
                formatted_output.append(formatted_sentence)
                
        return "\n".join(formatted_output)