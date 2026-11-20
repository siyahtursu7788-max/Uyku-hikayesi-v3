"""Bölüm 2.8 — AudioLDM ve çok profilli ambiyans üretim / zaman çizelgesi karıştırma motoru."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

from ..core.file_manager import sha256_file
from ..core.resource_monitor import release_gpu_vram

try:
    import soundfile as _sf
except Exception:
    _sf = None


def _read_wav(path: Path) -> Tuple[np.ndarray, int]:
    if _sf is not None:
        arr, sr = _sf.read(str(path), dtype="float32")
        return np.asarray(arr, dtype=np.float32), int(sr)
    sr, data = wavfile.read(str(path))
    if np.issubdtype(data.dtype, np.integer):
        arr = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    else:
        arr = data.astype(np.float32)
    return arr, int(sr)


def _write_wav(path: Path, arr: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(arr, dtype=np.float32), -1.0, 1.0)
    if _sf is not None:
        _sf.write(str(path), clipped, int(sr), subtype="PCM_16")
        return
    wavfile.write(str(path), int(sr), (clipped * 32767.0).astype(np.int16))


@dataclass(frozen=True)
class AmbienceProfile:
    key: str
    prompt: str
    lpf_hz: float
    hpf_hz: float
    target_db_under_voice: float
    clip_sec: float = 16.0


AMBIENCE_CATALOG: Dict[str, AmbienceProfile] = {
    "yagmur": AmbienceProfile(
        "yagmur",
        "Steady soft night rain pattering gently on a wooden roof from inside a cozy room, warm pink noise, no thunder, muted highs",
        1400.0,
        55.0,
        -35.0,
    ),
    "ruzgar": AmbienceProfile(
        "ruzgar",
        "Very soft muffled night breeze through distant pine trees and warm quiet indoor room tone, deep soothing sleep ambience",
        1200.0,
        55.0,
        -35.5,
    ),
    "deniz": AmbienceProfile(
        "deniz",
        "Calm distant slow ocean waves rolling gently onto a quiet sandy shore at midnight, warm low-pass sleep soundscape",
        1300.0,
        50.0,
        -35.0,
    ),
    "dalga": AmbienceProfile(
        "dalga",
        "Gentle slow rhythmic water waves lapping softly against smooth stones at night, tranquil sleep atmosphere",
        1300.0,
        55.0,
        -35.5,
    ),
    "somine": AmbienceProfile(
        "somine",
        "Soft warm cozy fireplace embers crackling very gently in a quiet wooden room at night, muted highs, no sharp pops",
        1500.0,
        60.0,
        -34.5,
    ),
    "orman": AmbienceProfile(
        "orman",
        "Tranquil midnight forest canopy soft breeze through leaves, calm natural ambient bed for deep sleep",
        1350.0,
        55.0,
        -36.0,
    ),
    "gece_bocekleri": AmbienceProfile(
        "gece_bocekleri",
        "Very distant soft warm summer night crickets in a quiet meadow, gentle low volume, soothing sleep background",
        1600.0,
        80.0,
        -37.0,
    ),
    "uzak_gok_gurultusu": AmbienceProfile(
        "uzak_gok_gurultusu",
        "Very distant soft low-frequency rolling thunder rumble behind gentle rain, muffled warm non-startling sleep storm",
        950.0,
        45.0,
        -36.5,
    ),
    "kutuphane": AmbienceProfile(
        "kutuphane",
        "Deep warm quiet ancient wooden library room tone, subtle low resonant warmth, no clock ticking, tranquil sanctuary",
        1100.0,
        55.0,
        -36.5,
    ),
    "dere_bahce": AmbienceProfile(
        "dere_bahce",
        "Calm quiet stream gently flowing over smooth round river stones at night with soft willow leaves breeze",
        1400.0,
        55.0,
        -35.5,
    ),
}


class AudioLDMAmbienceEngine:
    """
    AudioLDM tabanlı veya yerel lisanslı kaynak destekli ambiyans üretim ve döngü motoru.
    Anlatıcı sesini maskelemeyecek şekilde LPF/HPF, kesintisiz crossfade döngü ve seviye kontrolü uygular.
    """

    def __init__(
        self,
        output_dir: Path,
        sample_rate: int = 24000,
        model_name: str = "cvssp/audioldm-s-full-v2",
        licensed_assets_dir: Optional[Path] = None,
        generator_fn: Optional[Callable[[str, float], Tuple[np.ndarray, int]]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.model_name = model_name
        self.licensed_assets_dir = Path(licensed_assets_dir) if licensed_assets_dir else None
        self.generator_fn = generator_fn
        self._pipe = None

    def load_audioldm(self, device: str = "cuda") -> None:
        if self._pipe is not None or self.generator_fn is not None:
            return
        import torch
        from diffusers import AudioLDMPipeline

        dtype = torch.float16 if device == "cuda" else torch.float32
        self._pipe = AudioLDMPipeline.from_pretrained(self.model_name, torch_dtype=dtype)
        if device == "cuda":
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe = self._pipe.to(device)

    def unload(self) -> None:
        self._pipe = None
        release_gpu_vram()

    @staticmethod
    def _filter_sleep_safe(arr: np.ndarray, sr: int, hpf_hz: float, lpf_hz: float) -> np.ndarray:
        x = np.asarray(arr, dtype=np.float32)
        if x.ndim > 1:
            x = np.mean(x, axis=1).astype(np.float32)
        x = x - float(np.mean(x))
        sos_h = butter(2, max(hpf_hz / (0.5 * sr), 0.001), btype="high", output="sos")
        sos_l = butter(2, min(lpf_hz / (0.5 * sr), 0.99), btype="low", output="sos")
        x = sosfilt(sos_h, x)
        x = sosfilt(sos_l, x)
        return x.astype(np.float32)

    @staticmethod
    def make_seamless_loop(arr: np.ndarray, sr: int, crossfade_sec: float = 2.0) -> Tuple[np.ndarray, float]:
        """Baş/son bölgesini eşit-güçlü crossfade ile birleştirip dikiş sıçramasını ölçer."""
        n_cf = min(int(sr * crossfade_sec), len(arr) // 4)
        if n_cf < 16:
            return arr.astype(np.float32), 0.0
        body = arr[:-n_cf].copy()
        tail = arr[-n_cf:]
        t = np.linspace(0.0, np.pi / 2.0, n_cf, dtype=np.float32)
        body[:n_cf] = body[:n_cf] * np.sin(t) + tail * np.cos(t)
        seam_delta = float(abs(float(body[-1]) - float(body[0])))
        return body.astype(np.float32), round(seam_delta, 6)

    @staticmethod
    def extend_loop_to_duration(loop_arr: np.ndarray, sr: int, target_sec: float) -> np.ndarray:
        target_samples = max(1, int(round(sr * target_sec)))
        if len(loop_arr) >= target_samples:
            return loop_arr[:target_samples].astype(np.float32)
        reps = int(np.ceil(target_samples / max(1, len(loop_arr))))
        tiled = np.tile(loop_arr, reps)[:target_samples]
        return tiled.astype(np.float32)

    @staticmethod
    def _synthesize_deterministic_bed(profile: AmbienceProfile, sr: int) -> np.ndarray:
        """Yalnızca CPU/test modu veya lisanslı dosya yokken simüle ambiyans yatağı üretir."""
        rng = np.random.default_rng(abs(hash(profile.key)) % (2 ** 32))
        n = int(sr * profile.clip_sec)
        white = rng.standard_normal(n).astype(np.float32)
        pink = np.cumsum(white)
        pink = pink / (np.max(np.abs(pink)) + 1e-6) * 0.15
        return pink.astype(np.float32)

    def ensure_ambience_clip(
        self,
        profile_key: str,
        device: str = "cuda",
        allow_synthetic_test_fallback: bool = False,
    ) -> Dict[str, Any]:
        profile = AMBIENCE_CATALOG.get(profile_key, AMBIENCE_CATALOG["ruzgar"])
        out_path = self.output_dir / f"amb_{profile.key}.wav"
        source_mode = "CACHED"

        if not out_path.exists() or out_path.stat().st_size == 0:
            # 1. Lisanslı yerel ses kütüphanesi kontrolü
            licensed_cand = (
                self.licensed_assets_dir / f"{profile.key}.wav"
                if self.licensed_assets_dir
                else None
            )
            if licensed_cand and licensed_cand.exists():
                raw, sr_in = _read_wav(licensed_cand)
                source_mode = "LICENSED_ASSET"
            elif self.generator_fn is not None:
                raw, sr_in = self.generator_fn(profile.prompt, profile.clip_sec)
                source_mode = "CUSTOM_GENERATOR"
            elif allow_synthetic_test_fallback:
                raw = self._synthesize_deterministic_bed(profile, self.sample_rate)
                sr_in = self.sample_rate
                source_mode = "SYNTHETIC_CPU_TEST_FALLBACK"
            else:
                self.load_audioldm(device=device)
                assert self._pipe is not None
                out = self._pipe(
                    profile.prompt,
                    num_inference_steps=25,
                    audio_length_in_s=profile.clip_sec,
                )
                raw = np.asarray(out.audios[0], dtype=np.float32)
                sr_in = 16000
                source_mode = "AUDIOLDM_REAL_GPU"

            filtered = self._filter_sleep_safe(raw, sr_in, profile.hpf_hz, profile.lpf_hz)
            seamless, seam_delta = self.make_seamless_loop(filtered, sr_in, crossfade_sec=2.0)
            rms = float(np.sqrt(np.mean(seamless ** 2)) + 1e-9)
            target_lin = 10.0 ** (profile.target_db_under_voice / 20.0)
            normalized = np.clip(seamless * (target_lin / rms), -0.95, 0.95).astype(np.float32)
            _write_wav(out_path, normalized, sr_in)
        else:
            existing, sr_in = _read_wav(out_path)
            seam_delta = round(float(abs(float(existing[-1]) - float(existing[0]))), 6)

        return {
            "profile": asdict(profile),
            "path": str(out_path),
            "sha256": sha256_file(out_path),
            "source_mode": source_mode,
            "loop_seam_delta": seam_delta,
        }

    def render_timeline_ambience(
        self,
        segments: List[Dict[str, Any]],
        out_wav: Path,
        allow_synthetic_test_fallback: bool = False,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """Sahne zaman çizelgesine göre ambiyansları kesintisiz crossfade ile birleştirir."""
        pieces: List[np.ndarray] = []
        used_profiles: List[str] = []
        sr_ref = self.sample_rate
        cf_samples = int(sr_ref * 1.5)

        for seg in segments:
            key = str(seg.get("ambience_key", "ruzgar"))
            dur = max(0.5, float(seg.get("duration_sec", 5.0)))
            boost_db = float(seg.get("boost_db", 0.0))
            info = self.ensure_ambience_clip(
                key,
                device=device,
                allow_synthetic_test_fallback=allow_synthetic_test_fallback,
            )
            used_profiles.append(key)
            clip, sr_c = _read_wav(Path(info["path"]))
            sr_ref = sr_c
            ext = self.extend_loop_to_duration(clip, sr_c, dur)
            if boost_db != 0.0:
                ext = np.clip(ext * (10.0 ** (boost_db / 20.0)), -0.95, 0.95).astype(np.float32)
            pieces.append(ext)

        if not pieces:
            combined = np.zeros(sr_ref, dtype=np.float32)
        else:
            combined = pieces[0]
            for nxt in pieces[1:]:
                n_cf = min(cf_samples, len(combined) // 4, len(nxt) // 4)
                if n_cf > 16:
                    t = np.linspace(0.0, np.pi / 2.0, n_cf, dtype=np.float32)
                    blend = combined[-n_cf:] * np.cos(t) + nxt[:n_cf] * np.sin(t)
                    combined = np.concatenate([combined[:-n_cf], blend, nxt[n_cf:]])
                else:
                    combined = np.concatenate([combined, nxt])

        _write_wav(out_wav, combined, sr_ref)
        rms_db = round(float(20.0 * np.log10(np.sqrt(np.mean(combined ** 2)) + 1e-9)), 2)
        return {
            "output_path": str(out_wav),
            "sha256": sha256_file(out_wav),
            "duration_sec": round(len(combined) / float(sr_ref), 3),
            "rms_db": rms_db,
            "narrator_safe": rms_db <= -24.0,
            "profiles_used": sorted(set(used_profiles)),
        }
