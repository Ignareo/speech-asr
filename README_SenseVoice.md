# SenseVoice

github [repo](https://github.com/FunAudioLLM/SenseVoice) : https://github.com/FunAudioLLM/SenseVoice

SenseVoice是具有音频理解能力的音频基础模型，包括语音识别（ASR）、语种识别（LID）、语音情感识别（SER）和声学事件分类（AEC）或声学事件检测（AED）。本项目提供SenseVoice模型的介绍以及在多个任务测试集上的benchmark，以及体验模型所需的环境安装的与推理方式。

# 核心功能 🎯
**SenseVoice**专注于高精度多语言语音识别、情感辨识和音频事件检测
- **多语言识别：** 采用超过40万小时数据训练，支持超过50种语言，识别效果上优于Whisper模型。
- **富文本识别：** 
  - 具备优秀的情感识别，能够在测试数据上达到和超过目前最佳情感识别模型的效果。
  - 支持声音事件检测能力，支持音乐、掌声、笑声、哭声、咳嗽、喷嚏等多种常见人机交互事件进行检测。
- **高效推理：** SenseVoice-Small模型采用非自回归端到端框架，推理延迟极低，10s音频推理仅耗时70ms，15倍优于Whisper-Large。
- **微调定制：** 具备便捷的微调脚本与策略，方便用户根据业务场景修复长尾样本问题。
- **服务部署：** 具有完整的服务部署链路，支持多并发请求，支持客户端语言有，python、c++、html、java与c#等。

# 安装依赖环境 🐍

```shell
pip install -r requirements.txt
```

<a name="用法教程"></a>
# 用法 🛠️

## 推理



### 使用funasr推理

支持任意格式音频输入，支持任意时长输入

```python
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "FunAudioLLM/SenseVoiceSmall"


model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
    hub="hf",
)

# en
res = model.generate(
    input=f"{model.model_path}/example/en.mp3",
    cache={},
    language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
text = rich_transcription_postprocess(res[0]["text"])
print(text)
```
参数说明：
- `model_dir`：模型名称，或本地磁盘中的模型路径。
- `device`(str)： `cuda:0`（默认gpu0），使用 GPU 进行推理，指定。如果为`cpu`，则使用 CPU 进行推理。`mps`：mac电脑M系列新品试用mps进行推理。`xpu`：使用英特尔gpu进行推理。
- `vad_model`：表示开启VAD，VAD的作用是将长音频切割成短音频，此时推理耗时包括了VAD与SenseVoice总耗时，为链路耗时，如果需要单独测试SenseVoice模型耗时，可以关闭VAD模型。
- `vad_kwargs`：表示VAD模型配置,`max_single_segment_time`: 表示`vad_model`最大切割音频时长, 单位是毫秒ms。
- `use_itn`：输出结果中是否包含标点与逆文本正则化。
- `batch_size_s` 表示采用动态batch，batch中总音频时长，单位为秒s。
- `merge_vad`：是否将 vad 模型切割的短音频碎片合成，合并后长度为`merge_length_s`，单位为秒s。

如果输入均为短音频（小于30s），并且需要批量化推理，为了加快推理效率，可以移除vad模型，并设置`batch_size`

```python
model = AutoModel(model=model_dir, trust_remote_code=True, device="cuda:0", hub="hf")

res = model.generate(
    input=f"{model.model_path}/example/en.mp3",
    cache={},
    language="auto", # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size=64, 
)
```

更多详细用法，请参考 [文档](https://github.com/modelscope/FunASR/blob/main/docs/tutorial/README.md)







## `model`参数

在 `AutoModel` 的定义中，`model` 参数实际上是一个路径（`path`），它被传递到 `build_model` 方法中进行处理。以下是 `model` 参数（路径）在 `AutoModel` 中的处理流程：

---

### **1. `model` 参数的传递**
在外部调用中，`model` 被传递为一个路径字符串（如 `model_dir`），并通过 `kwargs` 传递到 `AutoModel` 的构造函数中：

```python
model = AutoModel(
    model=model_dir,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
    hub="hf",
    disable_update=True,
)
```

在 `AutoModel` 的 `__init__` 方法中，`kwargs` 包含了 `model` 参数：

```python
model, kwargs = self.build_model(**kwargs)
self.model = model
```

---

### **2. `build_model` 方法的处理**
`build_model` 方法是处理 `model` 参数的核心逻辑。以下是关键步骤：

#### **2.1 检查 `model` 参数**
`build_model` 方法首先检查 `kwargs` 中是否包含 `model` 参数：

```python
assert "model" in kwargs
```

如果 `model` 参数不存在，则会抛出异常。

#### **2.2 下载模型（如果需要）**
如果 `kwargs` 中没有提供 `model_conf`，则会尝试从模型中心（默认modelscope）下载模型：

```python
if "model_conf" not in kwargs:
    logging.info("download models from model hub: {}".format(kwargs.get("hub", "ms")))
    kwargs = download_model(**kwargs)
```

- `download_model` 方法会根据 `model` 参数（路径）下载模型文件，并将相关配置更新到 `kwargs` 中。
- 如果 `model` 是一个本地路径，则直接使用该路径。

#### **2.3 构建模型类**
在下载或加载模型配置后，`build_model` 方法会根据 `model` 参数的值，从注册表中获取对应的模型类：

```python
model_class = tables.model_classes.get(kwargs["model"])
assert model_class is not None, f'{kwargs["model"]} is not registered'
```

- `tables.model_classes` 是一个注册表，存储了所有支持的模型类。
- `kwargs["model"]` 是模型的名称或路径，用于从注册表中查找对应的模型类。

#### **2.4 初始化模型**
找到模型类后，`build_model` 方法会使用模型配置（`model_conf`）初始化模型实例：

```python
model_conf = {}
deep_update(model_conf, kwargs.get("model_conf", {}))
deep_update(model_conf, kwargs)
model = model_class(**model_conf)
```

- `deep_update` 方法将 `kwargs` 中的配置合并到 `model_conf` 中。
- 使用合并后的配置初始化模型类。

#### **2.5 加载预训练参数**
如果 `kwargs` 中提供了 `init_param` 参数（预训练模型路径），则会加载预训练参数：

```python
init_param = kwargs.get("init_param", None)
if init_param is not None:
    if os.path.exists(init_param):
        logging.info(f"Loading pretrained params from {init_param}")
        load_pretrained_model(
            model=model,
            path=init_param,
            ignore_init_mismatch=kwargs.get("ignore_init_mismatch", True),
            oss_bucket=kwargs.get("oss_bucket", None),
            scope_map=kwargs.get("scope_map", []),
            excludes=kwargs.get("excludes", None),
        )
    else:
        print(f"error, init_param does not exist!: {init_param}")
```

#### **2.6 设置设备**
最后，`build_model` 方法会将模型移动到指定的设备（如 `cuda:0`）：

```python
device = kwargs.get("device", "cuda")
if not torch.cuda.is_available() or kwargs.get("ngpu", 1) == 0:
    device = "cpu"
    kwargs["batch_size"] = 1
kwargs["device"] = device
model.to(device)
```

---

### **3. 总结**
`model` 参数（路径）在 `AutoModel` 中的处理流程如下：

1. **传递到 `build_model` 方法**：
   - `model` 参数作为路径字符串，通过 `kwargs` 传递到 `build_model` 方法。

2. **下载或加载模型**：
   - 如果 `model` 是一个远程路径（如 Hugging Face Hub 的模型名称），则会下载模型文件。
   - 如果 `model` 是一个本地路径，则直接加载。

3. **查找模型类**：
   - 根据 `model` 参数的值，从注册表中查找对应的模型类。

4. **初始化模型**：
   - 使用 `kwargs` 中的配置初始化模型实例。

5. **加载预训练参数**：
   - 如果提供了预训练模型路径，则加载预训练参数。

6. **设置设备**：
   - 将模型移动到指定的设备（如 GPU 或 CPU）。

最终，处理后的模型实例被返回并赋值给 `self.model`，供后续推理或训练使用。



## 直接推理

支持任意格式音频输入，输入音频时长限制在30s以下

```python
from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "FunAudioLLM/SenseVoiceSmall"
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device="cuda:0", hub="hf")
m.eval()

res = m.inference(
    data_in=f"{kwargs['model_path']}/example/en.mp3",
    language="auto", # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=False,
    **kwargs,
)

text = rich_transcription_postprocess(res[0][0]["text"])
print(text)
```




## WebUI

```shell
python webui.py
```



