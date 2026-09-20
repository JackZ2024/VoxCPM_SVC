# -*- coding: utf-8 -*-

"""VoxCPM 极致克隆 API 服务端。

TODO
后端
    - [x] 添加标题，加入文件命名规则
    - [x] 提供查询历史记录的界面
    - [x] 会话历史存入数据库，定时清理旧会话
    - [ ] 兼容中文版

前端
    - [ ] 兼容中文版
    - [x] 添加标题，加入文件命名规则
    - [x] 提供查询历史记录的界面
    - [x] 只显示 svc 音频下载
    - [x] 显示音频波形图
    - [x] 选择语言后自动选择其他选项的默认值
"""

import os
import gc
import json
import io
import random
import uuid
import threading
import traceback
import zipfile
from contextlib import contextmanager
from enum import Enum
from typing import Optional, Dict, Any, List, Generator
from datetime import datetime, timezone
from pathlib import Path
# from sys import exit
# from typing import Any, Generator
import re
# import json
import shutil
import time

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select, update, delete
import numpy as np
import soundfile as sf
from pydub import AudioSegment

from api_config import BASE_DIR, DEBUG, EXPIRED_SECONDS, VERSION
from api_database import Base, engine, SessionLocal, Session, Task


VOXCPM_MODEL_ID = os.environ.get("VOXCPM_MODEL_ID", "openbmb/VoxCPM2")
VOXCPM_MODEL_NAME = os.environ.get("VOXCPM_MODEL_NAME", "VoxCPM2")
VOXCPM_MODEL_DIR = Path(os.environ.get("VOXCPM_MODEL_DIR", str(BASE_DIR / "pretrained_model")))
VOXCPM_LORA_DIRS = (BASE_DIR / "lora_models", VOXCPM_MODEL_DIR / "lora")
# torch.compile/Inductor requires a Triton version matched to the installed
# PyTorch build. Keep it disabled by default for deployment compatibility.
VOXCPM_OPTIMIZE = os.environ.get("VOXCPM_OPTIMIZE", "false").strip().lower() in {"1", "true", "yes"}
# Set to 0 to keep VoxCPM2 resident permanently.  A positive value releases
# model/LoRA GPU memory after the final request has been idle for that period.
try:
    VOXCPM_IDLE_UNLOAD_SECONDS = max(0.0, float(os.environ.get("VOXCPM_IDLE_UNLOAD_SECONDS", "1800")))
except ValueError:
    VOXCPM_IDLE_UNLOAD_SECONDS = 1800.0
_voxcpm_model = None
_active_lora_dir: Optional[Path] = None
_voxcpm_model_lock = threading.RLock()
_voxcpm_active_users = 0
_voxcpm_last_released_at = time.monotonic()


def _has_complete_voxcpm_model(model_dir: Path) -> bool:
    """Check the minimum files required by ``VoxCPM.from_pretrained``."""
    has_main_weights = (model_dir / "model.safetensors").is_file() or (model_dir / "pytorch_model.bin").is_file()
    has_audio_vae = (model_dir / "audiovae.safetensors").is_file() or (model_dir / "audiovae.pth").is_file()
    return (model_dir / "config.json").is_file() and has_main_weights and has_audio_vae


def _get_lora_config(lora_dir: Optional[Path]):
    """Build a LoRA topology for hot swapping, even when the base model is used."""
    from voxcpm.model.voxcpm2 import LoRAConfig
    if lora_dir is not None:
        config_path = lora_dir / "lora_config.json"
        with config_path.open("r", encoding="utf-8") as config_file:
            config_data = json.load(config_file).get("lora_config", {})
        if config_data:
            return LoRAConfig(**config_data)
    # The upstream VoxCPM WebUI uses this topology for a base model so that a
    # matching LoRA can later be loaded without rebuilding model weights.
    return LoRAConfig(
        enable_lm=True,
        enable_dit=True,
        r=32,
        alpha=16,
        target_modules_lm=["q_proj", "v_proj", "k_proj", "o_proj"],
        target_modules_dit=["q_proj", "v_proj", "k_proj", "o_proj"],
    )


def _same_lora_topology(current_config, new_config) -> bool:
    """LoRA weights can be hot-swapped only when layer topology is identical."""
    return current_config is not None and current_config.model_dump() == new_config.model_dump()


def get_voxcpm_model(lora_dir: Optional[Path] = None):
    """Load VoxCPM2 once and hot-swap compatible LoRA weights when requested."""
    global _voxcpm_model, _active_lora_dir
    with _voxcpm_model_lock:
        if not _has_complete_voxcpm_model(VOXCPM_MODEL_DIR):
            from huggingface_hub import snapshot_download
            print(f"VoxCPM weights missing; downloading {VOXCPM_MODEL_ID} to {VOXCPM_MODEL_DIR} …")
            VOXCPM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=VOXCPM_MODEL_ID,
                local_dir=str(VOXCPM_MODEL_DIR),
            )
        if not _has_complete_voxcpm_model(VOXCPM_MODEL_DIR):
            raise FileNotFoundError(f"VoxCPM download is incomplete: {VOXCPM_MODEL_DIR}")

        target_config = _get_lora_config(lora_dir)
        needs_rebuild = (
            _voxcpm_model is None
            or not _same_lora_topology(_voxcpm_model.tts_model.lora_config, target_config)
        )
        if needs_rebuild:
            from voxcpm import VoxCPM
            _voxcpm_model = VoxCPM.from_pretrained(
                str(VOXCPM_MODEL_DIR),
                load_denoiser=False,
                optimize=VOXCPM_OPTIMIZE,
                lora_config=target_config,
            )
            if _voxcpm_model.tts_model.__class__.__name__ != "VoxCPM2Model":
                actual_model_type = _voxcpm_model.tts_model.__class__.__name__
                _voxcpm_model = None
                raise ValueError(f"仅支持 VoxCPM2，当前加载的模型类型为：{actual_model_type}")

        if lora_dir is None:
            _voxcpm_model.set_lora_enabled(False)
            _active_lora_dir = None
        elif _active_lora_dir != lora_dir or needs_rebuild:
            # Clear the prior adapter before copying a new adapter into the
            # already-created LoRA layers.  This avoids rebuilding VoxCPM2 for
            # ordinary compatible adapter switches.
            _voxcpm_model.unload_lora()
            _voxcpm_model.load_lora(str(lora_dir))
            _voxcpm_model.set_lora_enabled(True)
            _active_lora_dir = lora_dir
        else:
            _voxcpm_model.set_lora_enabled(True)
    return _voxcpm_model


def acquire_voxcpm_model(lora_dir: Optional[Path] = None):
    """Reserve the shared model so idle cleanup cannot unload it mid-request."""
    global _voxcpm_active_users
    with _voxcpm_model_lock:
        model = get_voxcpm_model(lora_dir)
        _voxcpm_active_users += 1
        return model


def release_voxcpm_model() -> None:
    """Mark one completed inference request and begin its idle timeout."""
    global _voxcpm_active_users, _voxcpm_last_released_at
    with _voxcpm_model_lock:
        _voxcpm_active_users = max(0, _voxcpm_active_users - 1)
        if _voxcpm_active_users == 0:
            _voxcpm_last_released_at = time.monotonic()


def unload_voxcpm_model_if_idle() -> bool:
    """Free base-model and LoRA memory when no request has used them recently."""
    global _voxcpm_model, _active_lora_dir
    if VOXCPM_IDLE_UNLOAD_SECONDS <= 0:
        return False
    with _voxcpm_model_lock:
        if (
            _voxcpm_model is None
            or _voxcpm_active_users > 0
            or time.monotonic() - _voxcpm_last_released_at < VOXCPM_IDLE_UNLOAD_SECONDS
        ):
            return False
        _voxcpm_model = None
        _active_lora_dir = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:
        print(f"VoxCPM CUDA cache cleanup skipped: {exc}")
    print("VoxCPM2 unloaded after idle timeout")
    return True


def _voxcpm_idle_unloader() -> None:
    """Background worker that makes model unloading independent of new requests."""
    while True:
        time.sleep(60)
        try:
            unload_voxcpm_model_if_idle()
        except Exception as exc:
            print(f"VoxCPM idle cleanup failed: {exc}")


def get_language_lora_models(language: str) -> dict[str, Path]:
    """Return complete LoRA checkpoints from lora_models/<language>/<name>."""
    models: dict[str, Path] = {}
    for root in VOXCPM_LORA_DIRS:
        language_dir = root / language
        if not language_dir.is_dir():
            continue
        for checkpoint in language_dir.iterdir():
            if not checkpoint.is_dir() or not (checkpoint / "lora_config.json").is_file():
                continue
            if any((checkpoint / name).is_file() for name in ("lora_weights.safetensors", "lora_weights.ckpt", "lora_weights.pth")):
                models.setdefault(checkpoint.name, checkpoint)
    return models


def seed_voxcpm(seed: int) -> None:
    """Seed runtimes without relying on a version-specific VoxCPM API argument."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_audio_stem(value: str, fallback: str) -> str:
    """Return a portable filename stem while retaining readable Unicode names."""
    stem = re.sub(r"[<>:\"/\\\\|?*\x00-\x1f]", "_", (value or "").strip())
    stem = re.sub(r"\s+", "_", stem).strip(". _")
    if not stem:
        stem = fallback
    # Windows device names cannot be used as filenames, even with an extension.
    if stem.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        stem = f"_{stem}"
    return stem[:120]


def format_file_created_at(stat_result: os.stat_result) -> str:
    """Format a file timestamp on Windows and Unix filesystems.

    ``st_birthtime`` is available on Windows/macOS but not on most Linux
    filesystems.  Linux therefore falls back to the file modification time.
    """
    timestamp = getattr(stat_result, "st_birthtime", stat_result.st_mtime)
    return datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")


def split_generation_line(line: str, index: int) -> tuple[str, str]:
    """Extract an optional legacy line label and its TTS text.

    A source line such as ``001- 你好`` becomes label ``001-`` and text ``你好``.
    Lines without a prefix receive their stable 3-digit generation index.
    """
    text = line.strip()
    match = re.match(r"^[\d_-]+", text)
    if not match:
        return f"{index:03d}", text
    label = match.group(0)
    remaining = text[len(label):].lstrip(" \t._-")
    # Do not turn a line consisting only of a label into an empty synthesis input.
    return safe_audio_stem(label, f"{index:03d}"), remaining or text


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AudioGenerationRequest(BaseModel):
    """音频生成请求模型"""
    ref_audio_orig: str = Field(..., description="参考音频原始文件")
    ref_text: str = Field(..., description="参考文本")
    gen_texts: str = Field(..., description="要生成的文本")
    language: str = Field(..., description="语言")
    model_name: str = Field(..., description="模型名称")
    gen_title: str = Field(..., description="生成音频的标题")
    seed: int = Field(default=-1, description="随机种子")
    cross_fade_duration: float = Field(default=0.15, ge=0, le=10, description="相邻分段间的静音长度（秒）")
    nfe_step: int = Field(default=10, ge=1, le=100, description="VoxCPM扩散推理步数")
    style_prompt: str = Field(default="", max_length=500, description="VoxCPM2 风格控制提示词")
    save_line_audio: bool = Field(default=False, description="保存分行音频")
    enable_svc: bool = Field(default=True, description="启用SVC")
    svc_type: str = Field(default="", description="SVC类型")
    svc_model: str = Field(default="", description="SVC模型")
    tone_shift: int = Field(default=0, description="音调偏移")
    rvc_index_rate: float = Field(default=0.8, description="RVC索引率")


class GlobalTaskStatus(BaseModel) :
    """全局任务状态模型"""
    status: str
    task_name: str
    completed_step: str
    progress: int


class TaskResponse(BaseModel):
    """任务响应模型"""
    session_id: str
    status: str
    message: str
    seed: int


class TaskProgress(BaseModel):
    """任务进度模型"""
    session_id: str
    status: str
    progress: int  # 0-100
    created_at: str
    updated_at: str
    result_files: List[str] = []
    last_file: str
    error_message: Optional[str] = None


class AudioFile(BaseModel):
    """音频文件信息"""
    filename: str
    session_id: str
    size: int
    created_at: str


class AudioFileList(BaseModel):
    """音频文件列表"""
    session_id: Optional[str] = None
    files: List[AudioFile]
    last_file: Optional[AudioFile] = None
    total_count: int


class AudioZipRequestItem(BaseModel):
    """A file selected from the audio history list for ZIP download."""
    filename: str
    session_id: str


class AvailableOptionLanguage(BaseModel):
    code: str
    name: str


class AvailableOptionModel(BaseModel):
    name: str
    display_name: str
    description: str
    languages: List[str]


class AvailableOptionSVCModel(BaseModel):
    name: str
    language: str
    type: str
    display_name: str


class AvailableOptionReference(BaseModel):
    name: str
    language: str
    text: str


class AvailableOptions(BaseModel):
    """可用选项模型"""
    languages: List[AvailableOptionLanguage]
    models: List[AvailableOptionModel]
    svc_types: List[str]
    svc_models: List[AvailableOptionSVCModel]
    references: List[AvailableOptionReference]


class AudioGenerationTask:
    """音频生成任务类"""
    
    def __init__(self, session_id: str, request_data: AudioGenerationRequest):
        self.session_id = session_id
        self.request_data = request_data
        self.status = TaskStatus.PENDING
        self.progress = 0
        self.completed_step = ""
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.heartbeat_at = self.updated_at
        self.result_files: List[str] = list()
        self.last_file = ""
        self.used_seed = -1
        self.error_message = None
        self.thread = None
        self.heartbeat_thread = None
        self.used_seed_ready = threading.Event()
        self._stop_event = threading.Event()
        self._heartbeat_stop_event = threading.Event()
        self._save_lock = threading.Lock()
        self._voxcpm_model_acquired = False

        self.task_instance = Task(
            session_id=self.session_id,
            status=self.status.value,
            progress=self.progress,
            result_files=self.result_files,
            last_file = self.last_file,
            used_seed=self.used_seed,
            error_message=self.error_message,
            heartbeat_at=self.heartbeat_at,
            request_data=self.request_data.model_dump(),
        )
    
    @contextmanager
    def _get_db(self) -> Generator[Session, Any, None] :
        db = SessionLocal()
        try :
            yield db
        finally :
            db.close()

    def save(self) :
        with self._save_lock:
            with self._get_db() as db :
                task_to_save = db.execute(select(Task).where(Task.session_id == self.session_id)).scalars().first()
                try :
                    if task_to_save and task_to_save.session_id == self.task_instance.session_id :
                        db.execute(update(Task).where(Task.session_id == self.session_id).values(
                            status=self.status.value,
                            progress=self.progress,
                            result_files=self.result_files,
                            last_file=self.last_file,
                            used_seed=self.used_seed,
                            error_message=self.error_message,
                            heartbeat_at=self.heartbeat_at,
                            updated_at=self.updated_at,
                        ))
                        db.commit()
                    else :
                        db.add(self.task_instance)
                        db.commit()
                        db.refresh(self.task_instance)
                except Exception :
                    db.rollback()
                    raise

    def start(self):
        """启动任务"""
        self.status = TaskStatus.RUNNING
        self.updated_at = datetime.now(timezone.utc)
        self.heartbeat_at = self.updated_at
        self.save()
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        self.thread = threading.Thread(target=self._generate_audio)
        self.thread.start()

    def _heartbeat_loop(self):
        """Persist liveness while a long model call is still executing."""
        while not self._heartbeat_stop_event.wait(15):
            if self.status != TaskStatus.RUNNING:
                return
            self.updated_at = datetime.now(timezone.utc)
            self.heartbeat_at = self.updated_at
            try:
                self.save()
            except Exception as exc:
                print(f"任务心跳保存失败 {self.session_id}: {exc}")
    
    def stop(self):
        """停止任务"""
        self._stop_event.set()
        self._heartbeat_stop_event.set()
        self.thread.join(timeout=30)
        if self.thread and self.thread.is_alive():
            raise OSError('Timeout when stopping task')
        self.status = TaskStatus.CANCELLED
        self.updated_at = datetime.now(timezone.utc)
        self.heartbeat_at = self.updated_at
        self.save()
    
    def get_used_seed(self, timeout=30) :
        """等待并获取 used_seed"""
        if self.used_seed_ready.wait(timeout=timeout) :
            return self.used_seed
        else :
            raise TimeoutError('无法获取 seed')

    def _generate_audio(self):
        """模拟音频生成过程"""
        """
            ref_audio_orig: str = Field(..., description="参考音频原始文件")
            ref_text: str = Field(..., description="参考文本")
            gen_texts: str = Field(..., description="要生成的文本")
            language: str = Field(..., description="语言")
            model_name: str = Field(..., description="模型名称")
            gen_title: str = Field(..., description="生成文件名前缀")
            seed: int = Field(default=-1, description="随机种子")
            cross_fade_duration: float = Field(default=0.15, description="交叉淡入淡出持续时间")
            nfe_step: int = Field(default=32, description="NFE步数")
            speed: float = Field(default=0.8, description="语速")
            save_line_audio: bool = Field(default=False, description="保存分行音频")
            insert_punct_in_space: bool = Field(default=False, description="在空格处插入标点")
            enable_svc: bool = Field(default=True, description="启用SVC")
            svc_type: str = Field(default="", description="SVC类型")
            svc_model: str = Field(default="", description="SVC模型")
            tone_shift: int = Field(default=0, description="音调偏移")
            rvc_index_rate: float = Field(default=0.75, description="RVC索引率")
            pre_gen_text: str = Field(..., description="生成文本前置文本")
            cut_length: float = Field(default=0.0, description="音频前面切除长度")
        """

        try:
            last_gen_audio, generated_files = self._infer()
            if last_gen_audio != "":
                self.result_files = generated_files
                self.last_file = str(last_gen_audio)
                self.status = TaskStatus.COMPLETED
                self.progress = 100
                self.completed_step = ""

        except Exception as e:
            self.status = TaskStatus.FAILED
            self.error_message = traceback.format_exc() if DEBUG else 'Task failed!'
            print("生成失败----", e)
        finally:
            self._heartbeat_stop_event.set()
            if self._voxcpm_model_acquired:
                release_voxcpm_model()
                self._voxcpm_model_acquired = False
            self.updated_at = datetime.now(timezone.utc)
            self.heartbeat_at = self.updated_at
            self.save()
    
    def get_progress(self) -> TaskProgress:
        """获取任务进度"""
        return TaskProgress(
            session_id=self.session_id,
            status=self.status.value,
            progress=self.progress,
            completed_step=self.completed_step,
            created_at=self.created_at.isoformat(),
            updated_at=self.updated_at.isoformat(),
            result_files=[os.path.basename(f) for f in self.result_files],
            last_file=os.path.basename(self.last_file),
            error_message=self.error_message
        )
    
    def get_file_list(self) -> AudioFileList:
        """获取文件列表"""
        files = []
        for file_path in self.result_files:
            if os.path.exists(file_path):
                stat = os.stat(file_path)
                files.append(AudioFile(
                    filename=os.path.basename(file_path),
                    session_id=self.session_id,
                    size=stat.st_size,
                    created_at=format_file_created_at(stat),
                ))

        last_audio = None
        if os.path.exists(self.last_file):
            stat = os.stat(self.last_file)
            last_audio = AudioFile(
                        filename=os.path.basename(self.last_file),
                        session_id=self.session_id,
                        size=stat.st_size,
                        created_at=format_file_created_at(stat),
                    )
        
        return AudioFileList(
            session_id=self.session_id,
            files=files,
            last_file=last_audio,
            total_count=len(files)
        )

    def is_stopped(self) :
        return self.status in (
            TaskStatus.COMPLETED, 
            TaskStatus.FAILED, 
            TaskStatus.CANCELLED
        )

    def is_expired(self) :
        updated_at = self.updated_at
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - updated_at).total_seconds() > EXPIRED_SECONDS

    def _infer(self):
        """Generate VoxCPM ultimate-clone segments, then optionally apply SVC/RVC."""
        from voxcpm_postprocess import concatenate_with_silence, rvc_convert_audio, sovits_convert_audio

        request = self.request_data
        ref_path = BASE_DIR / "refs" / request.language / request.ref_audio_orig
        if not ref_path.is_file():
            raise FileNotFoundError("缺少参考音频")
        if not request.ref_text.strip():
            raise ValueError("极致克隆必须提供与参考音频完全一致的参考文本")

        seed = request.seed if 0 <= request.seed <= 2**31 - 1 else int(np.random.randint(0, 2**31 - 1))
        self.used_seed = seed
        self.used_seed_ready.set()
        seed_voxcpm(seed)

        source_lines = [line for line in request.gen_texts.splitlines() if line.strip()]
        if not source_lines:
            raise ValueError("缺少要生成的文本")

        # Keep legacy numbered-line behavior: a leading ``001-`` / ``001_`` is
        # used as the individual file label and is not read aloud.
        texts = [split_generation_line(line, index) for index, line in enumerate(source_lines, start=1)]

        output_dir = BASE_DIR / "last_audio" / self.session_id
        temp_dir = output_dir / "tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        enable_svc, model_path, aux_path = get_svc_model(request.enable_svc, request.svc_type, request.svc_model, request.language)
        if request.enable_svc and not enable_svc:
            raise FileNotFoundError("选择的 SVC/RVC 模型不存在或未选择")

        lora_dir = None if request.model_name == "None" else get_language_lora_models(request.language).get(request.model_name)
        if request.model_name != "None" and lora_dir is None:
            raise FileNotFoundError(f"LoRA 模型不存在：{request.model_name}")
        model = acquire_voxcpm_model(lora_dir)
        self._voxcpm_model_acquired = True
        sample_rate = model.tts_model.sample_rate
        generated_waves, line_labels = [], []
        for index, (line_label, text) in enumerate(texts, start=1):
            if self._stop_event.is_set():
                self.status = TaskStatus.CANCELLED
                return "", []
            self.completed_step = f"{index}/{len(texts)}"
            
            styled_text = f"({request.style_prompt.strip()}){text}" if request.style_prompt.strip() else text
            generate_args = dict(
                text=styled_text,
                prompt_wav_path=str(ref_path),
                prompt_text=request.ref_text,
                inference_timesteps=request.nfe_step,
                denoise=False,
            )
            generate_args["reference_wav_path"] = str(ref_path)
            wave = model.generate(**generate_args)
            wave = np.asarray(wave, dtype=np.float32)
            if enable_svc:
                intermediate = temp_dir / f"segment-{index}.wav"
                sf.write(intermediate, wave, sample_rate, "PCM_24")
                if request.svc_type.lower() == "sovits":
                    sample_rate, wave = sovits_convert_audio(str(intermediate), model_path, aux_path, request.tone_shift)
                else:
                    sample_rate, wave = rvc_convert_audio(str(intermediate), model_path, aux_path, request.rvc_index_rate, request.tone_shift)
                if wave is None:
                    raise RuntimeError("RVC 音频转换失败")
            generated_waves.append(np.asarray(wave, dtype=np.float32))
            line_labels.append(line_label)
            self.progress = int(index / len(texts) * 100)
            self.updated_at = datetime.now(timezone.utc)
            self.heartbeat_at = self.updated_at
            self.save()

        output_audio_list = []
        pre_name = "orgi_audio"
        svc_type_str = ""
        s_version_str = ""
        s_version = request.svc_model.rpartition('-')[-1]
        if request.model_name != "None":
            f_version = request.model_name.rpartition('-')[-1]
        else:
            f_version = "0"
        if enable_svc:
            pre_name = "svc_audio"
            svc_type_str = request.svc_type.lower().lower()[0]
            s_version_str = str(s_version) + "-"

        gen_title = request.gen_title
        if request.save_line_audio:
            # 按行保存
            for i, audio_wave in enumerate(generated_waves):
                if gen_title != "":
                    audio_filepath = output_dir / f"{gen_title}-{i}.wav"
                else:
                    audio_filepath = output_dir / f"{pre_name}-{request.language }-f{f_version}-{svc_type_str}{s_version_str}_{i}.wav"
                sf.write(audio_filepath, audio_wave, sample_rate, 'PCM_24')
                output_audio_list.append(str(audio_filepath))
            final_waves = concatenate_with_silence(generated_waves, sample_rate, request.cross_fade_duration)
            if gen_title != "":
                last_gen_audio_path = output_dir / f"{gen_title}.mp3"
            else:
                last_gen_audio_path = output_dir / f"{pre_name}-{request.language }-f{f_version}-{svc_type_str}{s_version_str}.mp3"
            
            print(final_waves)
            # 如果是 float32/float64，范围通常是 [-1, 1]
            audio_int16 = (final_waves * 32767).astype(np.int16)
            audio = AudioSegment(
                audio_int16.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,  # int16 = 2 bytes
                channels=1
            )
            audio.export(last_gen_audio_path, format="mp3", bitrate="192k")
            
        else:
            # 导出转换后音频
            if gen_title != "":
                last_gen_audio_path = output_dir / f"{gen_title}.mp3"
                gen_audio_path = output_dir / f"{gen_title}.wav"
            else:
                last_gen_audio_path = output_dir / f"{pre_name}-{request.language }-f{f_version}-{svc_type_str}{s_version_str}.mp3"
                gen_audio_path = output_dir / f"{pre_name}-{request.language }-f{f_version}-{svc_type_str}{s_version_str}.wav"
            final_waves = None
            if len(generated_waves) > 0:
                final_waves = concatenate_with_silence(generated_waves, sample_rate, request.cross_fade_duration)
                sf.write(gen_audio_path, final_waves, sample_rate, 'PCM_24')
                output_audio_list.append(str(gen_audio_path))

                audio_int16 = (final_waves * 32767).astype(np.int16)
                audio = AudioSegment(
                    audio_int16.tobytes(),
                    frame_rate=sample_rate,
                    sample_width=2,  # int16 = 2 bytes
                    channels=1
                )
                audio.export(last_gen_audio_path, format="mp3", bitrate="192k")

        return str(last_gen_audio_path), output_audio_list


class TaskManager:
    """任务管理器"""
    __task_class__ = AudioGenerationTask
    
    def __init__(self):
        self.current_task: Optional[AudioGenerationTask] = None
        self._lock = threading.Lock()

    @property
    def task_history(self) -> List[AudioGenerationTask] :
        with self._get_db() as db :
            # 先更新当前任务的状态
            if self.current_task and not self.current_task.is_stopped() :
                self.current_task.save()
            # 直接从数据库返回所有任务列表
            all_history = db.execute(select(Task)).scalars().fetchall()
            if all_history :
                return all_history
            else :
                return []
    
    @contextmanager
    def _get_db(self) -> Generator[Session, Any, None] :
        db = SessionLocal()
        try :
            yield db
        finally :
            db.close()

    def create_task(self, request_data: AudioGenerationRequest) -> str:
        """创建新任务"""
        with self._lock:
            # 检查是否有正在运行的任务
            if self.current_task and self.current_task.status == TaskStatus.RUNNING:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="使用中，请稍后重试",
                )
            
            # 创建新任务
            session_id = str(uuid.uuid4())
            task = self.__task_class__(session_id, request_data)
            
            # 设置为当前任务
            self.current_task = task
            
            # 启动任务
            task.start()
            current_seed = task.get_used_seed(30)
            task.save()

            return session_id, current_seed
    
    def get_task(self, session_id: str) -> Optional[AudioGenerationTask]:
        """获取任务"""

        if self.current_task and self.current_task.session_id == session_id :
            return self.current_task
        else :
            with self._get_db() as db :
                history_one = db.execute(select(Task).where(Task.session_id == session_id)).scalars().first()
                if history_one:
                    task_one = AudioGenerationTask(
                        session_id=history_one.session_id,
                        request_data=AudioGenerationRequest.model_validate(history_one.request_data),
                    )
                    task_one.status = TaskStatus(history_one.status)
                    task_one.progress = history_one.progress
                    task_one.created_at=history_one.created_at
                    task_one.updated_at=history_one.updated_at
                    task_one.result_files=history_one.result_files
                    task_one.last_file=history_one.last_file
                    task_one.used_seed=history_one.used_seed
                    task_one.error_message=history_one.error_message
                    task_one.heartbeat_at=history_one.heartbeat_at
                    return task_one
                else:
                    return None
    
    def cancel_task(self, session_id: str) -> bool:
        """取消任务"""
        with self._lock:
            task = self.get_task(session_id)
            if not task:
                return False
            
            # 只能取消正在运行的任务
            if task.status == TaskStatus.RUNNING:
                try :
                    task.stop()
                except OSError as e :
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=str(e),
                    )
                
                # 如果是当前任务，清除引用
                if self.current_task and self.current_task.session_id == session_id:
                    self.current_task = None
                
                return True
            
            return False
    
    def clean_completed_tasks(self):
        """清理已完成的任务"""
        with self._lock:
            # 检查当前任务是否已完成
            if self.current_task and self.current_task.status in [
                TaskStatus.COMPLETED, 
                TaskStatus.FAILED, 
                TaskStatus.CANCELLED
            ]:
                self.current_task = None
            
            # 清理过期的已完成任务
            tasks_to_remove = []
            for task in self.task_history :
                if task.is_stopped() and task.is_expired() :
                    tasks_to_remove.append(task.session_id)
            self.remove_tasks(tasks_to_remove)

    def remove_tasks(self, session_id_list: List) :
        """从历史记录中移除任务"""
        try :
            with self._get_db() as db :
                db.execute(delete(Task).where(Task.session_id in session_id_list))
                db.commit()
            threading.Thread(target=self.delete_folders, args=(session_id_list,), daemon=True).start()
        except Exception :
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail="移除历史记录失败"
            )

    def delete_folders(self, session_id_list):
        for session_id in session_id_list:
            try:
                folder = BASE_DIR / 'last_audio' / session_id
                shutil.rmtree(folder)
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"Delete failed {folder}: {e}")

def find_first(path_: Path|str, start: str='', end: str='') :
    """
    Find first file or folder in `path_` according to starting or ending pattern
    
    :param path: folder to search for files & folders
    :type path: Path | str
    :param start: filename starts with this
    :type start: str
    :param end: filename ends with this
    :type end: str
    """
    search_path = Path(path_).resolve()

    for i in search_path.iterdir() :
        if i.name.startswith(start) and i.name.endswith(end) :
            result = i
            break
    else :
        result = None
    
    return result


def get_sovits_model(svc_model_name: str, lang_alone: str) :
    svc_models_list = [i for i in _get_available_options()['svc_models'] if i['type'] == 'sovits' and i['language'] == lang_alone]
    # svc_models_list = [
    #     {
    #         "name": model_path.name,
    #         "language": language_path.name,
    #         "type": "sovits",
    #         "display_name": f"sovits 音色转换和音质提升({language_path.name}/{model_path.name})",
    #     },
    #     ...
    # ]
    # lang_alone 尼泊尔语

    model_path = None
    speaker_path = None

    for speaker in svc_models_list :
        if speaker["name"] == svc_model_name :
            model_path = find_first(BASE_DIR / 'sovits-models' / speaker['language'] / speaker['name'], end='.pth')
            speaker_path = find_first(BASE_DIR / 'sovits-models' / speaker['language'] / speaker['name'], end='.npy')
            break
    if model_path is None or (not model_path.exists()) :
        print("sovits模型不存在" + svc_model_name)
        return False, "", ""
    else:
        return True, str(model_path), str(speaker_path)


def get_rvc_model(svc_model_name: str, lang_alone: str):
    svc_models_list = [i for i in _get_available_options()['svc_models'] if i['type'] == 'rvc' and i['language'] == lang_alone]
    # svc_models_list = [
    #     {
    #         "name": model_path.name,
    #         "language": language_path.name,
    #         "type": "rvc",
    #         "display_name": f"rvc 音色转换和音质提升({language_path.name}/{model_path.name})",
    #     },
    #     ...
    # ]

    model_path = None
    index_path = None

    for speaker in svc_models_list:
        if speaker["name"] == svc_model_name :
            model_path = find_first(BASE_DIR / 'rvc-models' / speaker['language'] / speaker['name'], end='.pth')
            index_path = find_first(BASE_DIR / 'rvc-models' / speaker['language'] / speaker['name'], end='.index')
            break
    if model_path is None or (not model_path.exists()) :
        print("RVC模型不存在" + svc_model_name)
        return False, "", ""
    else:
        return True, str(model_path), str(index_path)


def get_svc_model(enable_svc, svc_type, svc_model, lang_alone):
    if enable_svc:
        if svc_type is None or svc_model is None or svc_type == "" or svc_model == "":
            return False, "", ""
        else:
            if svc_type.lower() == "sovits":
                return get_sovits_model(svc_model, lang_alone)
            elif svc_type.lower() == "rvc":
                return get_rvc_model(svc_model, lang_alone)
    else:
        return False, "", ""


# 创建FastAPI应用
Base.metadata.create_all(bind=engine)


def recover_interrupted_tasks() -> None:
    """Migrate task columns and close work abandoned by a previous process."""
    now = datetime.now(timezone.utc)
    with engine.begin() as connection:
        columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(tasks)")}
        if "error_message" not in columns:
            connection.exec_driver_sql("ALTER TABLE tasks ADD COLUMN error_message TEXT")
        if "heartbeat_at" not in columns:
            connection.exec_driver_sql("ALTER TABLE tasks ADD COLUMN heartbeat_at DATETIME")
        result = connection.execute(
            update(Task)
            .where(Task.status.in_((TaskStatus.PENDING.value, TaskStatus.RUNNING.value)))
            .values(
                status=TaskStatus.FAILED.value,
                error_message="服务重启，任务已中断。",
                updated_at=now,
                heartbeat_at=now,
            )
        )
    if result.rowcount:
        print(f"已将 {result.rowcount} 个服务重启前未完成的任务标记为失败")


recover_interrupted_tasks()
if VOXCPM_IDLE_UNLOAD_SECONDS > 0:
    threading.Thread(target=_voxcpm_idle_unloader, name="voxcpm-idle-unloader", daemon=True).start()
app = FastAPI(title="音频生成服务API", version=VERSION)
last_cleanup_time = time.time()
cleanup_lock = threading.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建任务管理器
task_manager = TaskManager()

def _get_available_options() :
    """
    从文件目录获取模型信息
    """
    _AVAILABLE_OPTIONS = {
        'languages': [],
        'models': [],
        'svc_types': [],
        'svc_models': [],
        'references': [],
    }
    # Languages are defined solely by reference-audio folders.
    reference_path = BASE_DIR / 'refs'
    language_names = set()
    if reference_path.is_dir():
        language_names.update(path.name for path in reference_path.iterdir() if path.is_dir())
    
    # sovits_model_path = BASE_DIR / 'sovits-models' / language_name / model_name
    sovits_model_path = BASE_DIR / 'sovits-models'
    try :
        for language_path in sovits_model_path.iterdir() :
            if not language_path.is_dir() :
                continue

            for model_path in language_path.iterdir() :
                if not model_path.is_dir() :
                    continue

                _AVAILABLE_OPTIONS['svc_models'].append({
                    "name": model_path.name,
                    "language": language_path.name,
                    "type": "sovits",
                    "display_name": f"{model_path.name}",
                })
    except FileNotFoundError :
        print('sovits models not found')
    else :
        _AVAILABLE_OPTIONS['svc_types'].append("sovits")

    # rvc_model_path = BASE_DIR / 'rvc-models' / language_name / model_name
    rvc_model_path = BASE_DIR / 'rvc-models'
    try :
        for language_path in rvc_model_path.iterdir() :
            if not language_path.is_dir() :
                continue

            for model_path in language_path.iterdir() :
                if not model_path.is_dir() :
                    continue

                _AVAILABLE_OPTIONS['svc_models'].append({
                    "name": model_path.name,
                    "language": language_path.name,
                    "type": "rvc",
                    "display_name": f"{model_path.name}",
                })
    except FileNotFoundError :
        print('rvc models not found')
    else :
        _AVAILABLE_OPTIONS['svc_types'].append("rvc")

    if not reference_path.is_dir() :
        raise HTTPException(
            status_code=500,
            detail='参考音频不存在'
        )

    references = []
    for language_path in reference_path.iterdir() :
        if not language_path.is_dir() :
            continue

        for audio_file in language_path.iterdir() :
            if audio_file.suffix == '.wav' and audio_file.with_suffix('.txt').is_file() :
                references.append({
                    'name': audio_file.name,
                    'language': language_path.name,
                    'text': audio_file.with_suffix('.txt').read_text(encoding='utf-8'),
                })
    _AVAILABLE_OPTIONS['references'] = sorted(references, key=lambda x: x['name'])
    _AVAILABLE_OPTIONS['languages'] = [
        {'code': name, 'name': name} for name in sorted(language_names)
    ]
    for language in sorted(language_names):
        lora_models = get_language_lora_models(language)
        # Always expose the base model.  Selecting None deliberately disables LoRA.
        _AVAILABLE_OPTIONS['models'].append({
            'name': 'None',
            'display_name': 'None',
            'description': f'使用基础 VoxCPM2，不启用 LoRA（{language}）',
            'languages': [language],
        })
        _AVAILABLE_OPTIONS['models'].extend({
            'name': name,
            'display_name': name,
            'description': f'{language} 的 VoxCPM2 LoRA 微调模型',
            'languages': [language],
        } for name in sorted(lora_models))

    return _AVAILABLE_OPTIONS


@app.get("/api/options", response_model=AvailableOptions)
async def get_available_options():
    """
    获取可用的语言、模型等选项
    """
    AVAILABLE_OPTIONS = _get_available_options()
    return AvailableOptions(**AVAILABLE_OPTIONS)


@app.get("/api/global-status", response_model=GlobalTaskStatus)
async def get_global_status():
    """
    获取当前是否有任务在执行
    """
    task = task_manager.current_task
    if task and task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
        current_status = {
            'status': task.status.value,
            'task_name': task.request_data.language,
            'completed_step': task.completed_step,
            'progress': task.progress,
        }
    else:
        # Database history represents past tasks only; it never makes this
        # process busy after a restart.
        current_status = {
            'status': 'idle',
            'task_name': '',
            'completed_step': "",
            'progress': 0,
        }

    return GlobalTaskStatus(**current_status)


@app.post("/api/generate-audio", response_model=TaskResponse)
async def generate_audio(request: AudioGenerationRequest):
    """
    发起音频生成任务
    """
    try:
        
        AVAILABLE_OPTIONS = _get_available_options()
        # 验证语言和模型是否支持
        if request.language not in [lang["name"] for lang in AVAILABLE_OPTIONS["languages"]]:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的语言: {request.language}"
            )
        
        model_names = [
            model["name"] for model in AVAILABLE_OPTIONS["models"]
            if request.language in model["languages"]
        ]
        if request.model_name not in model_names:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的模型: {request.model_name}"
            )
        
        # 创建新任务
        session_id, used_seed = task_manager.create_task(request)

        # 清理已完成的任务，一天清理一次，减少遍历历史的次数
        global last_cleanup_time
        now = time.time()
        with cleanup_lock:
            if (now - last_cleanup_time) > 86400:
                last_cleanup_time = now
                task_manager.clean_completed_tasks()
        
        return TaskResponse(
            session_id=session_id,
            status="started",
            message="音频生成任务已启动",
            seed=used_seed
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/task/{session_id}/progress", response_model=TaskProgress)
async def get_task_progress(session_id: str):
    """
    查询任务进度
    """
    task = task_manager.get_task(session_id)
    
    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"任务 {session_id} 不存在"
        )
    
    return task.get_progress()


@app.get("/api/task/{session_id}/files", response_model=AudioFileList)
async def get_file_list(session_id: str):
    """
    获取任务生成的文件列表
    """
    task = task_manager.get_task(session_id)
    
    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"任务 {session_id} 不存在"
        )
    
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"任务尚未完成，当前状态: {task.status.value}"
        )
    
    return task.get_file_list()


@app.get("/api/task/{session_id}/download")
async def download_audio(
    session_id: str,
    filename: str = Query(..., description="要下载的文件名")
):
    """
    下载指定的音频文件
    """
    task = task_manager.get_task(session_id)
    
    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"任务 {session_id} 不存在"
        )
    
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"任务尚未完成，当前状态: {task.status.value}"
        )
    
    # 查找文件
    file_path = None
    for path in task.result_files:
        if os.path.basename(path) == filename:
            file_path = path
            break

    if not file_path and os.path.basename(task.last_file) == filename:
        file_path = task.last_file
    
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"文件 {filename} 不存在"
        )
    
    return FileResponse(
        file_path,
        media_type="audio/mpeg",
        filename=filename
    )


@app.post("/api/audio-files/download-zip")
async def download_audio_zip(files: List[AudioZipRequestItem]):
    """Package the audio files currently shown in the history list into a ZIP."""
    if not files:
        raise HTTPException(status_code=400, detail="请先获取至少一个音频文件")

    archive = io.BytesIO()
    used_names = set()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for item in files:
            task = task_manager.get_task(item.session_id)
            if not task or task.status != TaskStatus.COMPLETED:
                continue

            file_path = next(
                (path for path in task.result_files if os.path.basename(path) == item.filename),
                None,
            )
            if not file_path and task.last_file and os.path.basename(task.last_file) == item.filename:
                file_path = task.last_file
            if not file_path or not os.path.isfile(file_path):
                continue

            archive_name = os.path.basename(file_path)
            stem, extension = os.path.splitext(archive_name)
            suffix = 2
            while archive_name in used_names:
                archive_name = f"{stem}_{suffix}{extension}"
                suffix += 1
            used_names.add(archive_name)
            zip_file.write(file_path, arcname=archive_name)

    if not used_names:
        raise HTTPException(status_code=404, detail="所选音频文件已不存在或暂不可下载")

    archive.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return StreamingResponse(
        archive,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="recent_audios_{timestamp}.zip"'},
    )


@app.get("/api/task/{session_id}/cancel")
async def cancel_task(session_id: str):
    """
    取消任务
    """
    success = task_manager.cancel_task(session_id)
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail="无法取消任务，任务不存在或已完成"
        )
    
    return JSONResponse(
        content={
            "session_id": session_id,
            "status": "cancelled",
            "message": "任务已取消"
        }
    )


@app.get("/api/files-history", response_model=AudioFileList)
async def get_files_history(num: int):
    """
    获取服务状态（仅调试用）
    num: 要返回的历史记录数量
    最大只返回50个历史记录
    """
    
    # is_busy = False
    current_session_id = None

    if task_manager.current_task:
        current_session_id = task_manager.current_task.session_id
        # is_busy = task_manager.current_task.status == TaskStatus.RUNNING
    
    
    all_history = sorted(task_manager.task_history, key=lambda t: t.updated_at, reverse=True)
    if num < 1 or num >= len(all_history):
        num = 50
        if num >= len(all_history):
            num = len(all_history)

    files = []
    for task in all_history[:num]:
        for file_path in task.result_files:
            if os.path.exists(file_path):
                stat = os.stat(file_path)
                files.append(AudioFile(
                    filename=os.path.basename(file_path),
                    session_id=task.session_id,
                    size=stat.st_size,
                    created_at=format_file_created_at(stat),
                ))

    return AudioFileList(
            session_id=current_session_id,
            files=files,
            last_file=None,
            total_count=len(files)
        )

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # 返回一个 204 状态码，告诉浏览器：“我收到了，但我没有图标提供给你”
    return FileResponse("favicon.ico")

@app.get("/")
async def info():
    """
    API根路径
    """

    ## 否则，一律返回 index.html
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    else:
        return {
            "service": "音频生成服务",
            "version": VERSION,
            "endpoints": {
                "options": "/api/options",
                "generate": "/api/generate-audio",
                "progress": "/api/task/{session_id}/progress",
                "files": "/api/task/{session_id}/files",
                "download": "/api/task/{session_id}/download?filename=xxx",
                "cancel": "/api/task/{session_id}/cancel",
                "status": "/api/global-status"
            }
    }


if __name__ == "__main__":
    import uvicorn
    
    # 运行服务
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
