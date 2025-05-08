# 🎙️ 语音识别项目：SenseVoice-Small ASR 系统

> 本项目基于 [FunASR](https://github.com/modelscope/FunASR) 和 [SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice) 模型，提供多语言、高精度、低延迟的语音识别服务。支持命令行和 Web UI 两种使用方式。

2025.5.6完成初版
2025.5.7新增SenseVoice和Paraformer联用功能

后续改进：
- 优化联用模式下的处理速度
- 添加更多模型对比数据


---

## 📁 项目结构

```
speech-asr/
├── asr_engine.py            # 核心 ASR 推理引擎（封装模型加载 + 推理逻辑）
├── main.py                  # CLI 命令行工具入口
├── webui.py                 # Web 用户界面（Gradio 实现）
├── config.yaml              # 配置文件（模型路径、设备等）
├── utils/
│   ├── file_utils.py        # 文件处理工具（遍历音频、生成输出路径）
│   └── logger.py            # 日志模块（统一日志配置）
└── requirements.txt         # Python 依赖列表
```

---

## 🧠 功能亮点

- ✅ 多语言识别：支持中文、英文、粤语、日语、韩语等
- ✅ 支持语音事件识别（笑声、背景音乐、掌声等）
- ✅ 支持情感识别（高兴、悲伤、愤怒、中性等）
- ✅ Web UI 可视化 + 下载识别结果
- ✅ CLI 批量处理本地音频文件
- ✅ 自动命名输出文件（如 `audio.mp3` → `audio_asr.txt`）

---

## 模型介绍


### 对比

以下是不同模型的性能对比（音频长度：210秒/3分30秒）：

| 模型 | 处理时间 | 特点 |
|------|----------|------|
| SenseVoiceSmall | 14秒 | 速度快，支持情感识别和事件检测 |
| Paraformer | - | 支持时间戳，适合长音频 |
| SenseVoiceSmall+Paraformer（联用） | 113.52秒 | 综合两种模型优势，输出更全面 |

### SenseVoiceSmall





### Paraformer


HF上的[funasr/paraformer-zh](https://huggingface.co/funasr/paraformer-zh)没有带时间戳

### Whisper

Whisper-large-v3-turbo（1.5GB），包含约15亿参数

使用的是阿里团队修改后的[Whisper-large-v3-turbo 魔塔](https://www.modelscope.cn/models/iic/Whisper-large-v3-turbo/files)，参数用`pt`保存。
原始的是[Whisper-large-v3-turbo HF](https://huggingface.co/openai/whisper-large-v3-turbo)，使用`safetensors`保存参数。

---

## 🛠️ 使用方式

### 方式一：命令行批量识别（CLI）

#### 运行示例：

```bash
python main.py --input example/zh.mp3 --lang zh
python main.py --input example/ --lang auto
python main.py --input example/zh.mp3 --model Both  # 同时使用SenseVoiceSmall和Paraformer
```

| 参数           | 描述                                      |
| -------------- | ----------------------------------------- |
| `--input`      | 音频文件或目录路径                        |
| `--output_dir` | 输出文本文件保存目录（默认为当前目录）    |
| `--lang`       | 语言选择（auto/zh/en/yue/ja/ko/nospeech） |
| `--model`      | 模型类型（SenseVoice/Paraformer/Both）    |

识别结果会自动保存为 `.txt` 文件，文件名格式为：
- 单模型：`原始文件名_模型名称.txt`
- 双模型：`原始文件名_SenseVoiceSmall+Paraformer.txt`（包含两个模型的识别结果）

---

### 方式二：Web 界面交互识别（GUI）

#### 启动 WebUI：

```bash
python webui.py
```

打开浏览器访问：[http://127.0.0.1:7860](http://127.0.0.1:7860)

#### 特性说明：

- **2025.5.5存在问题：英文会失去空格（raw result正常，format后有问题）**

- 🎤 支持麦克风录音识别
- 📂 支持上传 `.mp3`, `.wav`, `.aac` 等常见格式音频
- 🌍 支持语言选择（auto/zh/en/yue/ja/ko）
- 📝 识别结果带表情符号（😊, 👏, 🎼 等）

---

## 📦 模块功能说明

### `asr_engine.py`

- 封装了 `AutoModel` 的初始化和推理过程
- 提供统一接口调用模型识别
- 支持 NumPy 数组或文件路径作为输入

### `main.py`

- CLI 入口脚本
- 支持批量识别音频文件
- 结果写入磁盘（自动命名）

### `webui.py`

- Gradio Web UI 入口
- 支持上传音频、麦克风录音
- 自动显示识别结果并提供下载链接

### `config.yaml`

- 存放模型路径、设备等配置参数

### `utils/file_utils.py`

- 文件路径处理
- 获取音频文件列表
- 构建输出路径

### `utils/logger.py`

- 统一日志系统
- 支持控制台和文件日志记录

---

## 📦 依赖安装

```bash
pip install funasr gradio numpy torchaudio librosa pydub
```

---

## 原理

### `AutoModel` 的参数

1. **模型路径**: 每个模型的路径可以是本地路径或远程模型名称。
2. **参数配置**: 每个模型的参数可以通过对应的 `*_kwargs` 参数单独配置。
3. **设备共享**: 所有模型默认使用同一个设备（通过 `device` 参数指定）。

1. `model`: 指定模型的名称或路径。可以是本地路径或远程模型名称（如 Hugging Face Hub 的模型名称）。
   - 示例: 
     ```python
     model="path/to/local/model"
     model="huggingface/model-name"
     ```

2. `device`: 指定运行设备。支持 `"cuda"`, `"cpu"`, `"mps"`（Mac M1/M2 GPU），或具体的 GPU 编号（如 `"cuda:0"`）。
   - 默认值: `"cuda"`
   - 示例:
     ```python
     device="cuda:0"
     device="cpu"
     ```

3. `vad_model`: 指定用于语音活动检测（VAD）的模型路径或名称。
   - 示例:
     ```python
     vad_model="path/to/vad/model"
     ```

4. `vad_kwargs`: 配置 VAD 模型的参数，例如最大切割时长。
   - 默认值: `{"max_single_segment_time": 30000}`
   - 示例:
     ```python
     vad_kwargs={"max_single_segment_time": 15000}
     ```

5. `punc_model`: 指定用于标点符号预测的模型路径或名称。
   - 示例:
     ```python
     punc_model="path/to/punctuation/model"
     ```

6. `punc_kwargs`: 配置标点符号模型的参数。
   - 默认值: 空字典 `{}`

7. `spk_model`: 指定用于说话人分离的模型路径或名称。
   - 示例:
     ```python
     spk_model="path/to/speaker/model"
     ```

8. `spk_kwargs`: 配置说话人分离模型的参数。
   - 默认值: 空字典 `{}`

9. `spk_mode`: 指定说话人分离的模式。支持 `"default"`, `"vad_segment"`, `"punc_segment"`。
   - 默认值: `"punc_segment"`

10. `hub`: 指定模型下载的来源。支持 `"hf"`（Hugging Face）或 `"ms"`（ModelScope）。
    - 默认值: `"hf"`

11. `disable_update`: 是否禁用版本检查。
    - 默认值: `False`

12. `log_level`: 设置日志级别。支持 `"INFO"`, `"DEBUG"`, `"ERROR"` 等。
    - 默认值: `"INFO"`

13. `batch_size`: 指定推理时的批量大小。
    - 默认值: `1`

14. `fp16` / `bf16`: 是否启用半精度（FP16 或 BF16）推理。
    - 默认值: `False`

15. `seed`: 设置随机种子以确保结果可复现。
    - 默认值: `0`

### 推理

`result = model.generate( ... )`

- model_dir：模型名称，或本地磁盘中的模型路径。

- vad_model：表示开启VAD，VAD的作用是将长音频切割成短音频，此时推理耗时包括了VAD与SenseVoice总耗时，为链路耗时，如果需要单独测试SenseVoice模型耗时，可以关闭VAD模型。

- vad_kwargs：表示VAD模型配置,max_single_segment_time: 表示vad_model最大切割音频时长, 单位是毫秒ms。

- use_itn：输出结果中是否包含标点与逆文本正则化。

- batch_size_s 表示采用动态batch，batch中总音频时长，单位为秒s。

- merge_vad：是否将 vad 模型切割的短音频碎片合成，合并后长度为merge_length_s，单位为秒s。

- ban_emo_unk：禁用emo_unk标签，禁用后所有的句子都会被赋与情感标签。



主要参数除了 input 和 input_len 外，还包括各种控制参数，如：
merge_vad: 是否合并短的 VAD 片段
batch_size_s: 批处理大小（秒）
return_spk_res: 是否返回说话人识别结果
sentence_timestamp: 是否返回句子级时间戳
return_raw_text: 是否返回未加标点的原始文本

`generate` 函数是 `AutoModel` 类的主要接口，用于处理语音识别请求。
根据是否配置了 VAD (Voice Activity Detection) 模型来决定调用路径：

```python
def generate(self, input, input_len=None, **cfg):
    if self.vad_model is None:
        return self.inference(input, input_len=input_len, **cfg)
    else:
        return self.inference_with_vad(input, input_len=input_len, **cfg)
```

`inference_with_vad` 是带有语音活动检测的推理函数，用于处理较长音频中有效语音片段的识别

1. 使用 VAD 模型检测音频中的语音片段：
2. 可选地合并短的 VAD 片段：
3. 设置 ASR 处理的批次大小：
4. 加载音频数据：
5. 根据 VAD 模型检测到的语音片段，将长音频切分并排序：
6. 批处理音频片段：
   - 将长度相近的音频片段组合成批次
   - 对每个批次进行 ASR 推理
   - 如果配置了说话人识别模型，还会处理说话人嵌入
7. 整合结果：
   - 恢复片段的原始顺序
   - 合并文本、时间戳和说话人信息
8. 标点符号处理（如果配置了标点模型）：
   - `if self.punc_model is not None: ...`
9. 说话人聚类（如果配置了说话人模型）：
   - `if self.spk_model is not None ...`
10. 句子级时间戳处理：
    - 根据标点符号和 ASR 的时间戳将文本分割成句子
    - 为每个句子分配说话人（如果有说话人识别）

这个函数的核心优势是可以处理长音频，通过以下方式提高效率：
1. 使用 VAD 跳过无语音部分
2. 对检测到的语音片段进行批处理
3. 集成多种模型（ASR、VAD、标点、说话人识别）形成完整的流水线

### 后处理

函数`rich_transcription_postprocess`是用于后处理富文本转录结果的，主要处理情感符号和事件标记。以下是该函数的工作原理：

1. 首先，函数将特殊标记`<|nospeech|><|Event_UNK|>`替换为问号表情符号"❓"
2. 将所有语言标记（如`<|zh|>`, `<|en|>`等）替换为统一的`<|lang|>`标记
3. 以`<|lang|>`为分隔符拆分字符串，并对每个部分使用`format_str_v2`函数处理
4. `format_str_v2`函数的作用是：
   - 统计并移除所有特殊标记
   - 识别主要情感标记（如高兴、悲伤等）
   - 添加事件标记（如背景音乐、掌声等）
   - 在文本末尾添加情感表情符号

5. 最后，函数将处理后的各段文本重新组合，同时：
   - 避免重复的事件标记
   - 移除重复的情感符号
   - 进行一些额外的文本清理（如移除"The."）

这个函数主要用于将ASR结果转换成更易读的形式，保留情感和非语音事件信息，并以表情符号的形式展示出来。

---

## 📚 说话人分离

- 说话人分离依赖时间戳，只有paraformer可以生成时间戳，SenseVoiceSmall不行

```bash
Only 'iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch' and 'iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch' can predict timestamp, and speaker diarization relies on timestamps.
```

## 附-平台

国内：ModelScope

- 自带命令行下载：`modelscope download`，在香港用比较慢
- 用git clone会比较快
- 注意环境变量`MODELSCOPE_CACHE`  （默认模型下载地址）

国外：Huggingface

- 注意环境变量`HF_HOME`  （默认模型下载地址）

---

推荐下载方式：

1. 先使用`GIT_LFS_SKIP_SMUDGE=1 git clone xxx.git` 跳过大文件下载

2. 再进入仓库，单独下载大文件`git lfs pull`

> 直接git clone无法显示进度

### Git LFS

Git LFS（Large File Storage）是 Git 的一个扩展

作用：专门用于管理和存储大文件

问题：当项目包含大文件时，会导致克隆和拉取速度变慢，仓库体积过大……

原理：将大文件替换为指针文件

- Git 只提交指针文件，而不是实际的大文件

- 大文件上传到远程 LFS 服务器，而不是 Git 仓库

## 问题

### Whisper不支持batch

1. 使用Whisper时报错：`batch decoding is not implemented`

Github issue有人遇到，但是尚无解决
https://github.com/modelscope/FunASR/issues/2273

源：`.venv\Lib\site-packages\funasr\models\whisper\model.py`


#### 1. 为什么会有这个问题

这个问题的根本原因在于FunASR库中Whisper模型实现的局限性：

1. **批处理未实现**：在FunASR库的WhisperWarp类中，`inference`方法明确检查`batch_size`参数，当它大于1时会抛出错误"batch decoding is not implemented"。这是因为Whisper模型的实现没有支持批处理功能。

2. **VAD分段处理**：当处理较长音频时，FunASR会先用VAD（语音活动检测）模型将音频切分成多个片段，然后批量送入ASR模型处理。对于其他模型（如SenseVoice和Paraformer）这种方式效率很高，但Whisper模型不支持批处理。

3. **参数配置不合理**：默认的`batch_size_s`参数为60秒，这会导致批处理大小非常大（内部转换为毫秒后约60000），而Whisper模型更适合处理较短的音频段。

#### 2. 为此做的改动

为了解决这个问题，我实现了以下优化：

1. **创建Whisper补丁**：
   - 通过猴子补丁（Monkey Patch）技术修改了WhisperWarp类的`inference`方法
   - 当`batch_size > 1`时，不抛出错误，而是改为逐个处理每个样本，然后合并结果

2. **限制批处理大小**：
   - 在补丁中添加了逻辑，将过大的`batch_size`值（>100）自动降低到合理的值（5）
   - 在ASR引擎中为Whisper模型设置了较小的`batch_size_s`值（5秒而非默认的60秒）

3. **优化VAD分段**：
   - 为Whisper模型修改了VAD的最大分段长度，从默认的30000毫秒（30秒）减少到5000毫秒（5秒）
   - 这使得Whisper模型可以处理更短的音频段，提高处理效率

4. **补丁AutoModel**：
   - 修改了AutoModel的`generate`方法，在检测到使用Whisper模型时自动调整批处理参数
   - 这确保了在整个处理流程中，Whisper模型都使用最优的参数设置

5. **参数传递优化**：
   - 在`run_model`方法中增加了对`vad_max_segment_length`参数的支持
   - 改进了参数传递逻辑，使配置更加灵活

这些改动使得Whisper模型可以高效处理长音频，大大提高了处理速度，同时保持了识别质量。我们没有修改FunASR库的源代码，而是通过补丁的方式在你的项目中实现了这些优化，确保了代码的兼容性和可维护性。

## 🖥️ 终端输出优化

本项目对命令行终端的输出进行了全面优化，使识别过程更加直观、友好。以下是主要改进：

### 1. 问题识别

在运行ASR时，终端输出存在以下问题：
- 重复的进度条信息（每个音频段显示两次相同的进度条）
- 缺乏整体处理进度显示
- 模型加载过程冗长且难以阅读
- 配置信息分散且不突出

### 2. 改进方案

我们实施了以下优化措施：

#### 2.1 彩色输出支持

为提高可读性，添加了颜色支持系统：

```python
# utils/logger.py
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    # ...其他颜色定义

class ColoredFormatter(logging.Formatter):
    """自定义日志格式化器，支持彩色输出"""
    
    FORMATS = {
        logging.INFO: '%(asctime)s - ' + Colors.GREEN + '%(levelname)s' + Colors.RESET + ' - %(message)s',
        logging.WARNING: '%(asctime)s - ' + Colors.YELLOW + '%(levelname)s' + Colors.RESET + ' - %(message)s',
        logging.ERROR: '%(asctime)s - ' + Colors.RED + '%(levelname)s' + Colors.RESET + ' - %(message)s',
        # ...其他日志级别格式
    }
```

同时添加了Windows平台支持：

```python
# 在Windows平台上启用ANSI颜色支持
if platform.system() == 'Windows':
    try:
        import colorama
        colorama.init()
    except ImportError:
        # 尝试使用Windows API启用ANSI
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except:
            pass
```

#### 2.2 配置摘要显示

在程序开始时，添加了清晰的配置摘要显示：

```python
def print_config_summary(input_path, output_dir, language, model_type, time_format, rich, use_postprocess):
    """打印配置摘要，提高可读性"""
    
    # 准备一个横线分隔符
    separator = "=" * 60
    
    # 配置标题与内容
    title = f"{Colors.BOLD}{Colors.CYAN}🔷 ASR配置摘要{Colors.RESET}"
    model_info = f"{Colors.BOLD}模型类型:{Colors.RESET} {model_type}"
    lang_info = f"{Colors.BOLD}语言:{Colors.RESET} {language}"
    # ...其他配置信息
    
    # 打印摘要
    print(separator)
    print(title)
    print(model_info)
    # ...打印其他配置信息
    print(separator)
```

#### 2.3 整体进度显示

添加了总体进度条，显示已处理文件数/总文件数：

```python
def process_audio_files(...):
    # 创建进度条以显示总体处理进度
    pbar = tqdm(total=total_files, 
                desc=f"{Colors.BOLD}{Colors.BLUE}总进度{Colors.RESET}", 
                unit="文件", position=0, leave=True, 
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    
    start_time = time.time()
    
    for idx, audio in enumerate(audio_files):
        # 更新进度条描述，显示当前处理的文件
        pbar.set_description(f"{Colors.BOLD}{Colors.BLUE}总进度{Colors.RESET} - 处理 {os.path.basename(audio)}")
        
        # ...处理音频文件
        
        # 更新进度条
        pbar.update(1)
    
    # 关闭进度条
    pbar.close()
    
    # 显示总耗时
    total_time = time.time() - start_time
    logging.info(f"{Colors.BOLD}{Colors.GREEN}🎉 全部处理完成! 总耗时: {total_time:.2f}秒{Colors.RESET}")
```

#### 2.4 控制FunASR进度条

为避免重复的进度条，修改了与FunASR的交互方式：

```python
def run_model(self, audio_input, language="auto", batch_size_s=60, disable_pbar=None, **kwargs):
    """执行推理"""
    # 合并所有参数
    params = {
        "input": audio_input,
        "use_itn": True,
        "language": language,
        "batch_size_s": batch_size_s,
        "merge_length_s": 15,
    }
    
    # 如果指定了disable_pbar，添加到参数中
    if disable_pbar is not None:
        params["disable_pbar"] = disable_pbar
        
    # 添加其他参数
    params.update(kwargs)
        
    result = self.model.generate(**params)
    return result
```

同时添加了命令行参数以控制详细程度：

```python
parser.add_argument("--quiet", action="store_true", help="静默模式，隐藏FunASR进度条")
parser.add_argument("--verbose", action="store_true", help="显示详细输出，包括FunASR进度条")
```

### 3. 优化效果

优化后的终端输出具有以下特点：

- **更清晰的信息层次**：通过颜色和格式区分不同类型的信息
- **精简的进度显示**：去除重复进度条，仅保留有效信息
- **总体进度可视化**：清晰展示整体处理进度和预计剩余时间
- **个性化控制**：用户可通过命令行参数控制输出详细程度
- **Windows兼容**：自动检测并启用Windows终端的颜色支持

### 4. 用法示例

使用彩色终端输出（默认）：
```bash
python main.py --input example/zh.mp3
```

使用静默模式（隐藏FunASR进度条）：
```bash
python main.py --input example/zh.mp3 --quiet
```

禁用彩色输出：
```bash
python main.py --input example/zh.mp3 --no_color
```

显示详细输出，包括FunASR进度条：
```bash
python main.py --input example/zh.mp3 --verbose
```

## 结束