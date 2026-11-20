"""
Whisper Turbo 16 kHz Kelime Zaman Damgalı Transkripsiyon Modülü (Bölüm 12).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..audio.text_normalizer import tr_norm_tokens


@dataclass
class WordTimestamp:
    word: str
    norm: str
    start: float
    end: float


def extract_16k_mono_wav(media_path: Path, out_wav: Path, ffmpeg_threads: int = 2) -> Path:
    subprocess.check_call([
        "ffmpeg", "-y", "-loglevel", "error", "-threads", str(ffmpeg_threads),
        "-i", str(media_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(out_wav),
    ])
    return out_wav


def transcribe_audio_16k(
    media_path: Path,
    local_tmp_dir: Path,
    outro_start_sec: Optional[float] = None,
    device: str = "cuda",
    model_name: str = "turbo",
) -> List[WordTimestamp]:
    """Final videodan 16 kHz ses çıkarıp Whisper Turbo ile kelime zaman damgalarını döndürür."""
    import whisper  # type: ignore

    wav_16k = Path(local_tmp_dir) / f"{media_path.stem}_16k.wav"
    extract_16k_mono_wav(media_path, wav_16k)
    try:
        asr = whisper.load_model(model_name, device=device)
        res = asr.transcribe(
            str(wav_16k),
            language="tr",
            fp16=(device == "cuda"),
            word_timestamps=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )
    finally:
        wav_16k.unlink(missing_ok=True)

    words: List[WordTimestamp] = []
    for seg in res.get("segments", []):
        for w in seg.get("words", []):
            raw_w = str(w.get("word", "")).strip()
            toks = tr_norm_tokens(raw_w)
            if not toks:
                continue
            st = float(w.get("start", seg.get("start", 0.0)))
            en = float(w.get("end", seg.get("end", st + 0.2)))
            if outro_start_sec is not None and st >= outro_start_sec:
                continue
            words.append(WordTimestamp(word=raw_w, norm=toks[0], start=st, end=max(en, st + 0.05)))
    return words
