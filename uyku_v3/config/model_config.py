"""
Test edilmiş ve sabitlenmiş model yapılandırmaları (XTTS-v2, Whisper Turbo, SDXL 1.0, SVD-XT, CLIP, AudioLDM).
Çalışan XTTS-v2 doğal insan tınısı parametreleri korunmuş, kontrollü yeniden deneme profilleri tanımlanmıştır.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class XTTSConfig:
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    model_version: str = "xtts_v2_coqui_0.25"
    language: str = "tr"
    sample_rate: int = 24000
    gpt_cond_len: int = 16
    gpt_cond_chunk_len: int = 6
    max_ref_len: int = 18
    speed: float = 0.84
    temperature: float = 0.22
    repetition_penalty: float = 10.0
    top_k: int = 20
    top_p: float = 0.65
    split_sentences: bool = False
    # Her denemede kontrollü parametre değişimi: (temperature, speed, top_p)
    retry_profiles: Tuple[Tuple[float, float, float], ...] = (
        (0.22, 0.84, 0.65),
        (0.27, 0.83, 0.68),
        (0.18, 0.84, 0.60),
        (0.31, 0.82, 0.72),
    )

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "gpt_cond_len": self.gpt_cond_len,
            "gpt_cond_chunk_len": self.gpt_cond_chunk_len,
            "max_ref_len": self.max_ref_len,
            "speed": self.speed,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "split_sentences": self.split_sentences,
        }


@dataclass(frozen=True)
class WhisperConfig:
    model_name: str = "turbo"
    language: str = "tr"
    word_timestamps: bool = True
    condition_on_previous_text: bool = False
    no_speech_threshold: float = 0.6


@dataclass(frozen=True)
class SDXLConfig:
    model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    variant: str = "fp16"
    gen_width: int = 1024
    gen_height: int = 576
    final_width: int = 1920
    final_height: int = 1080
    num_inference_steps: int = 28
    guidance_scale: float = 7.0
    use_imagen3_if_api_key: bool = True
    negative_prompt: str = (
        "bright daylight, harsh sun, neon, oversaturated, modern plastic, text, watermark, "
        "logo, signature, human, person, face, hands, silhouette of person, blurry, deformed, "
        "cartoon, anime, CGI look"
    )


@dataclass(frozen=True)
class SVDConfig:
    model_name: str = "stabilityai/stable-video-diffusion-img2vid-xt"
    variant: str = "fp16"
    width: int = 1024
    height: int = 576
    num_frames: int = 25
    decode_chunk_size: int = 4
    fps: int = 7
    num_inference_steps: int = 22
    default_noise_aug_strength: float = 0.02
    max_retries: int = 3
    enable_cpu_offload: bool = True


@dataclass(frozen=True)
class AudioLDMConfig:
    model_name: str = "cvssp/audioldm-s-full-v2"
    num_inference_steps: int = 25
    sfx_lpf_hz: float = 2400.0
    ambience_lpf_hz: float = 1400.0
    hpf_hz: float = 55.0


@dataclass(frozen=True)
class MotionProfile:
    name: str
    motion_bucket_id: int
    noise_aug_strength: float
    recommended_loop: str  # "ping_pong" | "forward_crossfade" | "optical_blend"
    directional: bool


# Sahne hareket türüne göre optimize edilmiş profiller (test sonuçlarına göre ayarlanabilir)
MOTION_PROFILES: Dict[str, MotionProfile] = {
    "indoor_calm": MotionProfile("indoor_calm", 38, 0.018, "ping_pong", False),
    "still_object": MotionProfile("still_object", 34, 0.015, "ping_pong", False),
    "candle_lamp": MotionProfile("candle_lamp", 44, 0.020, "forward_crossfade", True),
    "fireplace": MotionProfile("fireplace", 56, 0.020, "forward_crossfade", True),
    "steam_tea": MotionProfile("steam_tea", 50, 0.018, "forward_crossfade", True),
    "rain_window": MotionProfile("rain_window", 54, 0.020, "forward_crossfade", True),
    "ocean_waves": MotionProfile("ocean_waves", 60, 0.020, "optical_blend", True),
}

# K1-K6 kademeli anlatıcı ve atmosfer standartları
STEPPED_AUDIO_PROFILES: Dict[str, Dict[str, float]] = {
    "KAYIT_01": {"lufs_target": -14.0, "foley_db": -25.0, "amb_db": -35.0, "sentence_gap": 1.10, "tempo_start": 0.93, "tempo_end": 0.91},
    "KAYIT_02": {"lufs_target": -14.0, "foley_db": -25.0, "amb_db": -36.0, "sentence_gap": 1.20, "tempo_start": 0.91, "tempo_end": 0.89},
    "KAYIT_03": {"lufs_target": -14.0, "foley_db": -27.0, "amb_db": -36.5, "sentence_gap": 1.30, "tempo_start": 0.89, "tempo_end": 0.87},
    "KAYIT_04": {"lufs_target": -15.0, "foley_db": -28.0, "amb_db": -37.0, "sentence_gap": 1.45, "tempo_start": 0.86, "tempo_end": 0.84},
    "KAYIT_05": {"lufs_target": -16.5, "foley_db": -29.0, "amb_db": -36.0, "sentence_gap": 1.65, "tempo_start": 0.83, "tempo_end": 0.81},
    "KAYIT_06": {"lufs_target": -18.0, "foley_db": -30.5, "amb_db": -38.0, "sentence_gap": 2.00, "tempo_start": 0.80, "tempo_end": 0.78},
}
