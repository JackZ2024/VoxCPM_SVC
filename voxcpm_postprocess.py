"""SVC/RVC post-processing used by the VoxCPM API.

Kept separate from the legacy Gradio/F5 application so importing this service
does not import F5-TTS or initialise its vocoder.
"""
import os
import sys
import shutil
import threading
import types
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SOVITS_ROOT = ROOT / "sovits_svc"
RVC_ROOT = ROOT / "RVC"
RVC_ASSETS_ROOT = RVC_ROOT / "assets"
_model_download_lock = threading.Lock()
for path in (SOVITS_ROOT, RVC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def concatenate_with_silence(waves: list[np.ndarray], sample_rate: int, silence_seconds: float) -> np.ndarray:
    """Join generated lines with explicit silence, rather than F5 cross-fading."""
    if not waves:
        raise ValueError("没有可拼接的音频")
    silence = np.zeros(int(silence_seconds * sample_rate), dtype=np.float32)
    parts: list[np.ndarray] = []
    for index, wave in enumerate(waves):
        if index:
            parts.append(silence)
        parts.append(np.asarray(wave, dtype=np.float32).reshape(-1))
    return np.concatenate(parts)


def _install_rvc_infer_compatibility() -> None:
    """Expose this repository's flattened ``RVC`` tree as ``infer``."""
    if "infer" in sys.modules:
        return
    infer_package = types.ModuleType("infer")
    infer_package.__path__ = [str(RVC_ROOT)]
    infer_package.__package__ = "infer"
    sys.modules["infer"] = infer_package


def _read_custom_whisper(model_path: str) -> str | None:
    import yaml
    config_path = Path(model_path).parent / "config.yaml"
    if not config_path.is_file():
        return None
    with config_path.open("r", encoding="utf-8") as config_file:
        return (yaml.safe_load(config_file) or {}).get("custom_whisper")


def _ensure_sovits_prerequisites() -> None:
    """Download legacy So-VITS prerequisites only if either file is absent."""
    from huggingface_hub import snapshot_download
    whisper_path = ROOT / "whisper_pretrain" / "large-v2.pt"
    rmvpe_path = ROOT / "rmvpe_pretrain" / "rmvpe2.pt"
    with _model_download_lock:
        if not whisper_path.is_file():
            snapshot_download("Jack202410/sovits-pretrain", local_dir=str(ROOT), local_dir_use_symlinks=False)
        if not rmvpe_path.is_file():
            archive_path = ROOT / "rmvpe.zip"
            try:
                urllib.request.urlretrieve("https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip", archive_path)
                rmvpe_path.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(archive_path, "r") as archive:
                    archive.extractall(rmvpe_path.parent)
                extracted = rmvpe_path.parent / "model.pt"
                if not extracted.is_file():
                    raise FileNotFoundError("RMVPE 下载包中未找到 model.pt")
                shutil.move(str(extracted), str(rmvpe_path))
            finally:
                archive_path.unlink(missing_ok=True)


def sovits_convert_audio(audio_path: str, model_path: str, speaker_path: str, pitch: int = 0):
    """Run the existing So-VITS-SVC inference stack on a generated WAV."""
    import torch
    from omegaconf import OmegaConf
    from sovits_svc import svc_inference
    from sovits_svc.hubert.inference import hubert_infer
    from sovits_svc.pitch import load_csv_pitch
    from sovits_svc.pitch.inference import pitch_infer
    from sovits_svc.vits.models import SynthesizerInfer
    from sovits_svc.whisper.inference import whisper_infer

    custom_whisper = _read_custom_whisper(model_path)
    _ensure_sovits_prerequisites()
    work_dir = os.path.dirname(audio_path)
    ppg_path, vec_path, pit_path = (os.path.join(work_dir, name) for name in ("svc_tmp.ppg.npy", "svc_tmp.vec.npy", "svc_tmp.pit.csv"))
    whisper_infer(audio_path, ppg_path, custom_whisper)
    hubert_infer(audio_path, vec_path)
    pitch_infer(audio_path, pit_path, "rmvpe")
    hp = OmegaConf.load(str(SOVITS_ROOT / "configs" / "base.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SynthesizerInfer(hp.data.filter_length // 2 + 1, hp.data.segment_size // hp.data.hop_length, hp).eval().to(device)
    svc_inference.load_svc_model(model_path, model)
    args = type("Args", (), {
        "model": model_path, "spk": speaker_path, "wave": audio_path,
        "enable_retrieval": False, "retrieval_index_prefix": "",
        "retrieval_ratio": 0.5, "n_retrieval_vectors": 3,
        "hubert_index_path": "", "whisper_index_path": "",
    })()
    retrieval = svc_inference.create_retrival(args)
    ppg = torch.FloatTensor(np.repeat(np.load(ppg_path), 2, 0))
    vec = torch.FloatTensor(np.repeat(np.load(vec_path), 2, 0))
    pit = np.asarray(load_csv_pitch(pit_path), dtype=np.float32)
    if pitch:
        pit[pit > 0] *= 2 ** (pitch / 12)
    audio = svc_inference.svc_infer(model, retrieval, torch.FloatTensor(np.load(speaker_path)), torch.FloatTensor(pit), ppg, vec, hp, device, work_dir)
    return hp.data.sampling_rate, audio


_rvc_vc = None
_rvc_model_path = ""


def rvc_convert_audio(audio_path: str, model_path: str, index_path: str, index_rate: float, pitch: int = 0):
    """Run RVC and cache its loaded voice model between requests."""
    global _rvc_vc, _rvc_model_path
    if not RVC_ROOT.is_dir():
        raise FileNotFoundError(f"RVC runtime directory is missing: {RVC_ROOT}")
    _install_rvc_infer_compatibility()
    from configs.config import Config
    from modules.vc.modules import VC
    rmvpe_path = RVC_ASSETS_ROOT / "rmvpe" / "rmvpe.pt"
    if not rmvpe_path.is_file():
        from huggingface_hub import snapshot_download
        with _model_download_lock:
            if not rmvpe_path.is_file():
                snapshot_download(
                    repo_id="Jack202410/rvc_pretrain2",
                    local_dir=str(RVC_ASSETS_ROOT),
                    local_dir_use_symlinks=False,
                )
    if not rmvpe_path.is_file():
        raise FileNotFoundError(f"RVC RMVPE 预训练模型缺失：{rmvpe_path}")
    if _rvc_vc is None:
        _rvc_vc = VC(Config())
    if _rvc_model_path != model_path:
        _rvc_vc.get_vc(model_path, 0.33)
        _rvc_model_path = model_path
    return _rvc_vc.vc_single(0, audio_path, pitch, "", "rmvpe", index_path, "", index_rate, 3, 0, 0.25, 0.33, False)
