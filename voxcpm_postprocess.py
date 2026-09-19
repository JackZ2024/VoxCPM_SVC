"""SVC/RVC post-processing used by the VoxCPM API.

Kept separate from the legacy Gradio/F5 application so importing this service
does not import F5-TTS or initialise its vocoder.
"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
for path in (ROOT / "sovits_svc", ROOT / "RVC"):
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


def remove_silence(wave: np.ndarray, sample_rate: int, threshold: float = 0.005) -> np.ndarray:
    """Trim leading/trailing quiet samples without removing requested inter-line gaps."""
    samples = np.asarray(wave, dtype=np.float32).reshape(-1)
    active = np.flatnonzero(np.abs(samples) > threshold)
    return samples[active[0] : active[-1] + 1] if active.size else samples


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

    work_dir = os.path.dirname(audio_path)
    ppg_path, vec_path, pit_path = (os.path.join(work_dir, name) for name in ("svc_tmp.ppg.npy", "svc_tmp.vec.npy", "svc_tmp.pit.csv"))
    whisper_infer(audio_path, ppg_path, None)
    hubert_infer(audio_path, vec_path)
    pitch_infer(audio_path, pit_path, "rmvpe")
    hp = OmegaConf.load(str(ROOT / "sovits_svc" / "configs" / "base.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SynthesizerInfer(hp.data.filter_length // 2 + 1, hp.data.segment_size // hp.data.hop_length, hp).eval().to(device)
    svc_inference.load_svc_model(model_path, model)
    args = type("Args", (), {"model": model_path, "wave": audio_path, "enable_retrieval": False, "retrieval_index_prefix": "", "retrieval_ratio": 0.5, "n_retrieval_vectors": 3})()
    retrieval = svc_inference.create_retrival(args)
    ppg = torch.FloatTensor(np.repeat(np.load(ppg_path), 2, 0))
    vec = torch.FloatTensor(np.repeat(np.load(vec_path), 2, 0))
    pit = np.asarray(load_csv_pitch(pit_path), dtype=np.float32)
    if pitch:
        pit *= 2 ** (pitch / 12)
    audio = svc_inference.svc_infer(model, retrieval, torch.FloatTensor(np.load(speaker_path)), torch.FloatTensor(pit), ppg, vec, hp, device, work_dir)
    return hp.data.sampling_rate, audio


_rvc_vc = None
_rvc_model_path = ""


def rvc_convert_audio(audio_path: str, model_path: str, index_path: str, index_rate: float, pitch: int = 0):
    """Run RVC and cache its loaded voice model between requests."""
    global _rvc_vc, _rvc_model_path
    from configs.config import Config
    from modules.vc.modules import VC
    if _rvc_vc is None:
        _rvc_vc = VC(Config())
    if _rvc_model_path != model_path:
        _rvc_vc.get_vc(model_path, 0.33)
        _rvc_model_path = model_path
    return _rvc_vc.vc_single(0, audio_path, pitch, "", "rmvpe", index_path, "", index_rate, 3, 0, 0.25, 0.33, False)
