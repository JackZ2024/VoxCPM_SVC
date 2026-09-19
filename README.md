# VoxCPM2-SVC

基于 **VoxCPM2** 的网页端极致声音克隆服务，提供 FastAPI 后端和 Web 界面，并保留 So-VITS-SVC、RVC 音色转换。

声音克隆以参考音频和其对应文本为条件；LoRA 用于指定语种、风格或领域的微调，不能替代参考音频。

## 功能

- 仅支持 VoxCPM2。
- 语种由 `refs/<语种>/` 中的参考音频决定，不再由基础模型决定。
- 同一语种可有多个 LoRA；模型列表始终包含 `None`，表示基础 VoxCPM2，不使用 LoRA。
- 兼容 LoRA 热切换；结构不兼容时自动安全重建模型。
- 支持 So-VITS-SVC 和 RVC 后处理、音调偏移、RVC 索引率。
- `cross_fade_duration` 用于控制相邻文本行之间的静音时长；不会删除生成音频中的静音。
- 可按行保存成品音频，并直接导入音色转换面板。
- 任务、进度与输出持久化到 SQLite；服务重启时未完成任务会标记失败，历史和已生成文件保留。
- VoxCPM2 默认空闲 30 分钟自动卸载，释放显存；下一次生成自动加载。

## 环境要求

- Windows 10/11
- NVIDIA GPU 与对应 CUDA 驱动
- Python 3.10+；使用内置环境时为 `runtime\python.exe`
- Git（初始化子模块时需要）

项目 `setup.bat` 使用 PyTorch `2.8.0 + cu129` 和 `triton-windows==3.4.0.post21`。自行更换 PyTorch 时，请选择与它兼容的 Triton。默认不启用 `torch.compile`，可避免大多数 Triton/Inductor 版本冲突。

## 安装

首次克隆后初始化 VoxCPM 子模块：

```powershell
git submodule update --init --recursive
```

使用项目内置运行时：

```bat
setup.bat
runtime\python.exe -m pip install -e .\VoxCPM
```

`-e .\VoxCPM` 确保服务使用当前子模块的 VoxCPM 代码，包括 LoRA 热切换接口。

使用自己的 Python 环境时，先安装与 CUDA 匹配的 PyTorch/torchaudio，再执行：

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .\VoxCPM
```

## 启动

```bat
RunAPI.bat
```

默认监听 `0.0.0.0:7862`，浏览器访问 `http://127.0.0.1:7862/`。

部署到公网时请配置反向代理、访问控制与 HTTPS；服务默认允许跨域，不建议直接暴露给不受信任的网络。

## 目录约定

```text
VoxCPM_SVC/
├─ pretrained_model/              # VoxCPM2 基础模型，缺失时自动下载
├─ refs/
│  └─ 中文/
│     ├─ demo.wav                 # 参考音频
│     └─ demo.txt                 # 与音频逐字对应的参考文本
├─ lora_models/
│  └─ 中文/
│     └─ my_lora/
│        ├─ lora_config.json
│        └─ lora_weights.safetensors
├─ sovits-models/                 # 可选：So-VITS-SVC 模型
├─ rvc-models/                    # 可选：RVC 模型及 .index
├─ RVC/                            # RVC 推理运行时
├─ last_audio/                    # 任务输出目录
└─ api.db                          # 任务历史数据库
```

### 参考音频

每个参考音频必须有同名 `.txt` 文件。例如 `refs/中文/demo.wav` 必须有 `refs/中文/demo.txt`。参考文本应与音频内容尽量逐字一致，这是克隆质量的关键。

语种列表只扫描 `refs/`。VoxCPM2 支持多语种；某个语种有 LoRA 时，LoRA 只会显示在该语种的模型列表内。

### 基础模型和 LoRA

服务优先从 `pretrained_model/` 加载模型；若缺少 `config.json`、主权重或 AudioVAE 权重，则自动从 Hugging Face `openbmb/VoxCPM2` 下载。

可通过环境变量修改来源或目录：

```bat
set VOXCPM_MODEL_ID=openbmb/VoxCPM2
set VOXCPM_MODEL_DIR=D:\models\VoxCPM2
```

每个 LoRA 目录至少需要：

- `lora_config.json`
- `lora_weights.safetensors`、`lora_weights.ckpt` 或 `lora_weights.pth`

选择 `None` 时调用 `set_lora_enabled(False)`。切换 LoRA 时服务调用 `unload_lora()`、`load_lora()` 和 `set_lora_enabled(True)`；如果 rank、目标模块等层结构不一致，会重新构建模型以保证权重正确加载。

## 网页使用

1. 选择语种、参考音频和模型（`None` 或对应 LoRA）。
2. 填写与参考音频一致的参考文本。
3. 输入待生成文本；每个非空行独立生成，之后按 `cross_fade_duration` 指定的静音时长拼接。
4. 可填写提示词以控制 VoxCPM2 的风格。
5. 需要后处理时启用 SVC，选择 `sovits` 或 `rvc` 及对应模型。
6. 勾选“按行保存音频”会额外输出每一行的成品音频。

若输入行以 `001-`、`001_` 等数字/连字符/下划线开头，该前缀会作为文件编号且不被朗读；否则使用行号。分段文件以 `segm_audio-` 开头，音色转换面板可自动识别，例如：

```text
segm_audio-第01章-001--001.wav
```

最终合并文件为 `输出音频文件名.wav`。未填写名称时，会按 VoxCPM2、语种、模型、转换方式和随机种子自动命名。Windows 文件名非法字符会自动替换，重复行编号不会覆盖文件。

## 空闲卸载

默认最后一个任务结束 1800 秒（30 分钟）后卸载 VoxCPM2 与 LoRA，并清理 CUDA 缓存。任务运行中不会卸载。

```bat
REM 改为 10 分钟
set VOXCPM_IDLE_UNLOAD_SECONDS=600

REM 关闭自动卸载，始终常驻显存
set VOXCPM_IDLE_UNLOAD_SECONDS=0
```

请在启动 `RunAPI.bat` 前设置环境变量，或写入启动脚本后重启服务。

## API 摘要

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/options` | 获取语种、参考音频、LoRA、SVC/RVC 选项 |
| `POST` | `/api/generate-audio` | 创建生成任务 |
| `GET` | `/api/task/{session_id}/progress` | 查询进度和状态 |
| `GET` | `/api/task/{session_id}/files` | 获取已完成任务的输出 |
| `GET` | `/api/task/{session_id}/download?filename=...` | 下载指定文件 |
| `GET` | `/api/task/{session_id}/cancel` | 取消任务 |
| `GET` | `/api/global-status` | 查询服务忙碌状态 |
| `GET` | `/api/files-history?num=20` | 查询近期文件 |

创建任务示例：

```json
{
  "ref_audio_orig": "demo.wav",
  "ref_text": "这是参考音频对应的完整文本。",
  "gen_texts": "001- 第一行待生成文本。\n002- 第二行待生成文本。",
  "language": "中文",
  "model_name": "None",
  "gen_title": "第01章",
  "seed": -1,
  "cross_fade_duration": 0.15,
  "nfe_step": 10,
  "style_prompt": "自然、平稳、清晰",
  "save_line_audio": true,
  "enable_svc": false,
  "svc_type": "",
  "svc_model": "",
  "tone_shift": 0,
  "rvc_index_rate": 0.8
}
```

`seed: -1` 表示随机种子，创建任务的响应会返回实际使用的种子。

## 常见问题

### `No module named 'infer'`

RVC 原始源码使用 `infer.lib`、`infer.modules` 导入，而本项目是 `RVC/lib`、`RVC/modules` 的扁平结构。后处理层已提供兼容映射；请确认完整保留 `RVC/` 目录并重启服务。

### Triton / Inductor 报错

默认 `VOXCPM_OPTIMIZE=false`，不会使用 `torch.compile`。如果自行启用后发生 Triton 报错，请让 Triton 与 PyTorch 版本匹配，或关闭优化：

```bat
set VOXCPM_OPTIMIZE=false
```

### 服务重启后显示“使用中”

服务启动时会将数据库中遗留的 `pending`、`running` 任务标记为失败，并保留历史与输出。客户端仍显示旧状态时，请刷新页面或重新查询 `/api/global-status`。

## 数据与 Git

模型、参考音频、输出、运行时环境和数据库已在 `.gitignore` 中排除。不要提交私有语音、模型权重或 `api.db`。更新 VoxCPM 子模块：

```powershell
git submodule update --remote VoxCPM
```
