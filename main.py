# main.py

# ====== 🛠️ 手动配置区域（开发时可直接修改）=====
# INPUT_PATH = "example/zh.mp3"              # 输入路径（文件或目录）
INPUT_PATH = ""                # 默认输入路径
OUTPUT_DIR = ""                             # 输出目录（默认为当前目录）
LANGUAGE = "auto"                           # 默认语言
MODEL_TYPE = "Both"                   # 模型类型 (SenseVoice/Paraformer/Both)
# ===================================================

import os
import argparse
import yaml
import time
import sys
from tqdm import tqdm
from utils.file_utils import get_audio_files, get_output_path, save_text_to_file
from asr_engine import ASREngine
from utils.logger import setup_logger, Colors
import logging
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# 尝试从local_config.py加载本地配置（不会被git追踪）
try:
    from local_config import LOCAL_INPUT_PATH
    INPUT_PATH = LOCAL_INPUT_PATH
except ImportError:
    # 如果没有local_config.py，使用默认值
    pass

def print_config_summary(input_path, output_dir, language, model_type, time_format, rich, use_postprocess):
    """打印配置摘要，提高可读性"""
    
    # 准备一个横线分隔符
    separator = "=" * 60
    
    # 配置标题
    title = f"{Colors.BOLD}{Colors.CYAN}🔷 ASR配置摘要{Colors.RESET}"
    
    # 基本配置
    model_info = f"{Colors.BOLD}模型类型:{Colors.RESET} {model_type}"
    lang_info = f"{Colors.BOLD}语言:{Colors.RESET} {language}"
    
    # 输入输出信息
    if os.path.isfile(input_path):
        input_type = "单个文件"
    elif os.path.isdir(input_path):
        input_type = "文件夹"
    else:
        input_type = "未指定"
    
    input_info = f"{Colors.BOLD}输入:{Colors.RESET} {input_type} - {input_path}"
    output_info = f"{Colors.BOLD}输出目录:{Colors.RESET} {'同输入目录' if not output_dir else output_dir}"
    
    # 格式化选项
    format_options = []
    
    if time_format and (model_type == "Paraformer" or model_type == "Both"):
        format_options.append(f"时间格式: {time_format}")
    
    if rich is not None and (model_type == "SenseVoice" or model_type == "Both"):
        format_options.append(f"富文本: {'是' if rich else '否'}")
    
    if use_postprocess:
        format_options.append("自动后处理")
    else:
        format_options.append("手动指定后处理")
    
    format_info = f"{Colors.BOLD}选项:{Colors.RESET} {', '.join(format_options)}"
    
    # 打印摘要
    print(separator)
    print(title)
    print(model_info)
    print(lang_info)
    print(input_info)
    print(output_info)
    print(format_info)
    print(separator)
    print()

def process_audio_files(asr, audio_files, output_dir=None, language="auto", time_format="sec", rich=None, use_postprocess=True, quiet=False):
    """
    对于每个音频文件，执行推理，并保存结果到txt文件

    Args:
        asr: ASREngine 实例或包含两个模型的字典 {"SenseVoice": asr1, "Paraformer": asr2}
        audio_files: 音频文件列表
        output_dir: 输出目录
        language: 语言代码
        time_format: 时间格式（仅对Paraformer有效）
        rich: 是否使用富文本（仅对SenseVoice有效）
        use_postprocess: 是否使用自动选择的后处理方法
        quiet: 是否在FunASR运行时禁用tqdm进度条
    """
    total_files = len(audio_files)
    if total_files == 0:
        logging.warning("没有找到音频文件")
        return
    
    # 创建进度条以显示总体处理进度
    pbar = tqdm(total=total_files, desc=f"{Colors.BOLD}{Colors.BLUE}总进度{Colors.RESET}", 
                unit="文件", position=0, leave=True, 
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    
    start_time = time.time()
    
    for idx, audio in enumerate(audio_files):
        try:
            # 更新进度条
            pbar.set_description(f"{Colors.BOLD}{Colors.BLUE}总进度{Colors.RESET} - 处理 {os.path.basename(audio)}")
            
            # 显示处理信息
            logging.info(f"{Colors.BOLD}{Colors.YELLOW}🔶 处理第 {idx+1}/{total_files} 个文件: {audio}{Colors.RESET}")
            
            # 设置FunASR运行参数，禁用其内部进度条
            run_options = {
                "language": language,
                "disable_pbar": quiet  # 传递参数以禁用或启用FunASR的进度条
            }
            
            # 检查是否使用两个模型
            if isinstance(asr, dict) and "SenseVoice" in asr and "Paraformer" in asr:
                # 使用两个模型
                logging.info(f"{Colors.BOLD}使用SenseVoiceSmall模型处理...{Colors.RESET}")
                sense_result = asr["SenseVoice"].run_model(audio, **run_options)
                sense_text = asr["SenseVoice"].postprocess(sense_result, rich=rich) if use_postprocess else asr["SenseVoice"].postprocess_general(sense_result, rich=True if rich is None else rich)
                
                logging.info(f"{Colors.BOLD}使用Paraformer模型处理...{Colors.RESET}")
                para_result = asr["Paraformer"].run_model(audio, **run_options)
                para_text = asr["Paraformer"].postprocess(para_result, time_format=time_format) if use_postprocess else asr["Paraformer"].postprocess_Paraformer(para_result, time_format=time_format)
                
                # 合并结果
                combined_text = "========SenseVoiceSmall==========\n" + sense_text + "\n\n===========Paraformer==============\n" + para_text
                
                # 使用基础文件名保存，添加特殊模型标识
                output_path = get_output_path(audio, output_dir, "SenseVoiceSmall+Paraformer")
                save_text_to_file(combined_text, output_path)
                logging.info(f"{Colors.BOLD}{Colors.GREEN}✅ 已保存合并结果到 {output_path}{Colors.RESET}")
            else:
                # 使用单个模型
                result = asr.run_model(audio, **run_options)
                
                output_path = get_output_path(audio, output_dir, asr.model_type)
                
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
                logging.info(f"{Colors.BOLD}{Colors.GREEN}✅ 已保存到 {output_path}{Colors.RESET}")
            
            # 更新总进度条
            pbar.update(1)
            
        except Exception as e:
            logging.error(f"{Colors.BOLD}处理 {audio} 时出错: {e}{Colors.RESET}")
    
    # 关闭进度条
    pbar.close()
    
    # 显示总耗时
    total_time = time.time() - start_time
    logging.info(f"{Colors.BOLD}{Colors.GREEN}🎉 全部处理完成! 总耗时: {total_time:.2f}秒{Colors.RESET}")

def main():
    # ====== 解析命令行参数（优先级高于手动配置）======
    parser = argparse.ArgumentParser(description="批量语音识别")
    parser.add_argument("--input", type=str, help="音频文件或目录")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--lang", type=str, default="auto", help="语言 (auto/zh/en/yue/ja/ko)")
    parser.add_argument("--model", type=str, default=None, help="模型类型 (SenseVoice/Paraformer/Both)")
    parser.add_argument("--time_format", type=str, default="sec", choices=["sec", "min"], 
                        help="时间格式 (仅对Paraformer有效)")
    parser.add_argument("--rich", action="store_true", help="使用富文本 (仅对SenseVoice有效)")
    parser.add_argument("--no_rich", action="store_true", help="不使用富文本 (仅对SenseVoice有效)")
    parser.add_argument("--manual_postprocess", action="store_true", help="手动指定后处理方法而不是自动选择")
    parser.add_argument("--no_color", action="store_true", help="禁用彩色输出")
    parser.add_argument("--quiet", action="store_true", help="静默模式，隐藏FunASR进度条")
    parser.add_argument("--verbose", action="store_true", help="显示详细输出，包括FunASR进度条")
    args = parser.parse_args()

    # ====== 参数优先级：命令行 > 手动配置 ======
    input_path = args.input or INPUT_PATH
    output_dir = args.output_dir or OUTPUT_DIR
    language = args.lang or LANGUAGE
    model_type = args.model or MODEL_TYPE
    time_format = args.time_format
    use_postprocess = not args.manual_postprocess
    enable_color = not args.no_color
    
    # 处理富文本参数
    rich = None
    if args.rich:
        rich = True
    elif args.no_rich:
        rich = False
        
    # 处理静默模式参数（优先级：quiet > verbose）
    quiet_mode = args.quiet or (not args.verbose)

    # ====== 初始化日志 & 加载配置 ======
    setup_logger(level=logging.INFO, enable_color=enable_color)
    
    # 打印欢迎信息
    logging.info(f"{Colors.BOLD}{Colors.MAGENTA}🎤 欢迎使用 Speech ASR CLI{Colors.RESET}")
    
    # 显示配置摘要
    print_config_summary(
        input_path=input_path,
        output_dir=output_dir,
        # language=language,
        language="en",
        model_type=model_type,
        time_format=time_format,
        rich=rich,
        use_postprocess=use_postprocess
    )

    # 加载配置文件
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            logging.info(f"{Colors.BOLD}已成功加载 config.yaml{Colors.RESET}")
    except Exception as e:
        logging.error(f"加载配置文件失败: {e}")
        return

    # ====== 初始化引擎 ======
    # 添加静默模式设置
    config["disable_pbar"] = quiet_mode
    
    try:
        if model_type.lower() == "both":
            logging.info(f"{Colors.BOLD}正在初始化 SenseVoice 模型，请稍候...{Colors.RESET}")
            sense_asr = ASREngine(config, model_type="SenseVoice")
            logging.info(f"{Colors.BOLD}{Colors.GREEN}✅ SenseVoice 模型初始化完成{Colors.RESET}")
            
            logging.info(f"{Colors.BOLD}正在初始化 Paraformer 模型，请稍候...{Colors.RESET}")
            para_asr = ASREngine(config, model_type="Paraformer")
            logging.info(f"{Colors.BOLD}{Colors.GREEN}✅ Paraformer 模型初始化完成{Colors.RESET}")
            
            # 创建模型字典
            asr = {
                "SenseVoice": sense_asr,
                "Paraformer": para_asr
            }
        else:
            logging.info(f"{Colors.BOLD}正在初始化 {model_type} 模型，请稍候...{Colors.RESET}")
            asr = ASREngine(config, model_type=model_type)
            logging.info(f"{Colors.BOLD}{Colors.GREEN}✅ 模型初始化完成{Colors.RESET}")
    except Exception as e:
        logging.error(f"初始化ASR引擎失败: {e}")
        return
    
    # ====== 获取音频文件列表 ======
    try:
        audio_files = get_audio_files(input_path)
        logging.info(f"找到 {len(audio_files)} 个音频文件准备处理")
    except FileNotFoundError as e:
        logging.error(f"找不到输入路径: {e}")
        return
    except Exception as e:
        logging.error(f"获取音频文件列表失败: {e}")
        return
    
    # ====== 执行推理 ======
    process_audio_files(
        asr, 
        audio_files, 
        output_dir, 
        language, 
        time_format, 
        rich, 
        use_postprocess,
        quiet=quiet_mode
    )

if __name__ == "__main__":
    main()