"""
V2.2 Cümle Bazlı Kalıcı WAV + SHA-256 Önbelleği.
- Yalnızca APPROVED cümleleri atomik olarak diske kaydeder.
- Kesinti sonrası yeniden çalıştırmada SHA-256 + metin hash + parametre hash doğrulaması yapar.
- Bozuk, eksik veya hash uyuşmayan önbellek dosyası ASLA kullanılmaz (fail-closed).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
try:
    import soundfile as _sf
except ImportError:
    _sf = None
from scipy.io import wavfile as _wavfile


def _read_wav(path: Path):
    if _sf is not None:
        return _sf.read(path, dtype="float32")
    sr, d = _wavfile.read(path)
    if d.dtype.kind in ("i", "u"):
        d = d.astype(np.float32) / 32768.0
    else:
        d = d.astype(np.float32)
    return d, sr


def _write_wav(path: Path, data: np.ndarray, sr: int):
    if _sf is not None:
        _sf.write(path, data.astype(np.float32), sr)
    else:
        pcm = np.clip(data, -1.0, 1.0)
        _wavfile.write(path, sr, (pcm * 32767.0).astype(np.int16))

from ..core.file_manager import (
    atomic_copy_file,
    atomic_write_json,
    safe_read_json,
    sha256_dict,
    sha256_file,
    sha256_text,
)


@dataclass
class SentenceCacheEntry:
    paragraph_id: str
    sentence_index: int
    expected_text: str
    text_sha256: str
    params_sha256: str
    wav_path: str
    wav_sha256: str
    status: str
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SentenceAudioCache:
    def __init__(self, cache_dir: Path, sample_rate: int = 24000) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.hits = 0
        self.misses = 0

    def _paths(self, paragraph_id: str, sentence_index: int) -> Tuple[Path, Path]:
        stem = f"{paragraph_id}_C{sentence_index:02d}"
        return self.cache_dir / f"{stem}.wav", self.cache_dir / f"{stem}.json"

    def get_verified(
        self,
        paragraph_id: str,
        sentence_index: int,
        expected_text: str,
        params: Dict[str, Any],
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        wav_p, meta_p = self._paths(paragraph_id, sentence_index)
        if not wav_p.exists() or not meta_p.exists() or wav_p.stat().st_size == 0:
            self.misses += 1
            return None

        meta = safe_read_json(meta_p, default=None)
        if not isinstance(meta, dict) or meta.get("status") != "APPROVED":
            self.misses += 1
            return None

        if meta.get("text_sha256") != sha256_text(expected_text):
            self.misses += 1
            return None

        if meta.get("params_sha256") != sha256_dict(params):
            self.misses += 1
            return None

        try:
            actual_wav_sha = sha256_file(wav_p)
        except OSError:
            self.misses += 1
            return None

        if actual_wav_sha != meta.get("wav_sha256"):
            self.misses += 1
            return None

        try:
            audio, sr_in = _read_wav(wav_p)
            if sr_in != self.sample_rate or len(audio) == 0:
                self.misses += 1
                return None
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
        except Exception:
            self.misses += 1
            return None

        self.hits += 1
        return audio.astype(np.float32), meta

    def store_approved(
        self,
        paragraph_id: str,
        sentence_index: int,
        expected_text: str,
        params: Dict[str, Any],
        audio: np.ndarray,
        metrics: Dict[str, Any],
        local_tmp_dir: Path,
    ) -> SentenceCacheEntry:
        wav_p, meta_p = self._paths(paragraph_id, sentence_index)
        local_tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_wav = local_tmp_dir / f"cache_{paragraph_id}_C{sentence_index:02d}.wav"
        _write_wav(tmp_wav, audio, self.sample_rate)
        wav_sha = atomic_copy_file(tmp_wav, wav_p, verify_hash=True)
        tmp_wav.unlink(missing_ok=True)

        entry = SentenceCacheEntry(
            paragraph_id=paragraph_id,
            sentence_index=sentence_index,
            expected_text=expected_text,
            text_sha256=sha256_text(expected_text),
            params_sha256=sha256_dict(params),
            wav_path=str(wav_p),
            wav_sha256=wav_sha,
            status="APPROVED",
            metrics=metrics,
        )
        atomic_write_json(meta_p, entry.to_dict())
        return entry
