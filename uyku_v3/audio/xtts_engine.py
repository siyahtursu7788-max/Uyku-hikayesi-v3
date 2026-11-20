"""
XTTS-v2 Profesyonel Ses Üretim Motoru (Bölüm 5 & 6).
- GPU zorunluluğu denetlenir (sessizce CPU'ya düşmez).
- Çalışan doğal tını parametreleri korunur; her denemede kontrollü parametre değişimi yapılır.
- P0-02 DÜZELTMESİ: Tüm denemeler başarısız olursa çıktı ASLA APPROVED yapılmaz;
  MANUAL_REVIEW durumuna alınır ve orijinal başarısız ses inceleme için saklanır.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

from ..config.model_config import XTTSConfig
from ..config.quality_thresholds import AudioThresholds
from ..core.state_manager import ArtifactStatus
from .audio_quality import check_splice_boundary, safe_trim_trailing_ghost
from .whisper_validator import AudioValidationResult, validate_sentence_audio


@dataclass
class SentenceSynthesisResult:
    sentence_text: str
    status: ArtifactStatus
    audio: Optional[np.ndarray]
    attempts: int
    used_params: Dict[str, Any]
    validation: Optional[AudioValidationResult]
    history: List[Dict[str, Any]] = field(default_factory=list)
    raw_audio: Optional[np.ndarray] = None
    profile_used: str = ""
    atempo_applied: Optional[float] = None


def gentle_voice_cleanup(w: np.ndarray, sr: int = 24000) -> np.ndarray:
    """
    Doğal insan tınısını ve konuşma dinamiklerini koruyan hafif temizlik:
    - DC offset temizliği + 70 Hz sıfır-fazlı HPF
    - Geniş pencereli (240ms) hafif tepe yumuşatma (agresif cümle bazlı normalizasyon YOK!)
    - Yumuşak giriş/çıkış fade (click önleme)
    """
    if len(w) == 0:
        return w.astype(np.float32)
    x = np.asarray(w, dtype=np.float32)
    x = x - float(np.mean(x))

    if len(x) > 32:
        sos = butter(2, max(70.0 / (0.5 * sr), 0.001), btype="high", output="sos")
        x = sosfiltfilt(sos, x).astype(np.float32)

    # Geniş pencereli tepe yumuşatıcı (dinamikleri ezmeden yalnızca aşırı patlamaları yumuşatır)
    win = max(1, int(sr * 0.24))
    env = np.sqrt(uniform_filter1d(x ** 2, size=win) + 1e-12)
    speech_mask = env > 0.012
    if np.any(speech_mask):
        med = float(np.percentile(env[speech_mask], 55))
        ceiling = med * 1.38
        gain = np.where(env > ceiling, (ceiling / (env + 1e-6)) ** 0.50, 1.0)
        gain = uniform_filter1d(gain.astype(np.float32), size=max(1, int(sr * 0.16)))
        x = x * gain

    # Yumuşak tıklama önleyici kenarlar
    fi = min(int(sr * 0.020), len(x) // 10)
    fo = min(int(sr * 0.035), len(x) // 10)
    if fi > 0:
        x[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    if fo > 0:
        x[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    return x.astype(np.float32)



def is_pure_fast_wpm_only(val: AudioValidationResult, max_compensable_wpm: float = 185.0) -> bool:
    """
    V2.2 Kontrollü Hız Telafisi Kapısı:
    - YALNIZCA metin %100 doğruysa (WER=0, CER=0, Align=1.0, son kelime/hece sağlam,
      eksik/fazla kelime yok) ve tek sorun uygulanmış WPM sınırının hafif aşılmasıysa True döner.
    - Telaffuz hatalarını ASLA gizlemez!
    """
    if val.wer != 0.0 or val.cer != 0.0 or val.word_alignment_ratio < 1.0:
        return False
    if not (val.last_word_ok and val.last_syllable_ok and val.second_last_word_ok):
        return False
    if val.missing_words or val.extra_words:
        return False
    ac = val.acoustic if isinstance(val.acoustic, dict) else {}
    wpm = float(ac.get("wpm", 0.0))
    applied_max = float(ac.get("applied_max_wpm", 145.0))
    if not (applied_max < wpm <= max_compensable_wpm):
        return False
    # Tek sorun hızlı konuşma olmalı (clipping, RMS, abrupt_tail_cut vb. olmamalı)
    if ac.get("clipped_samples", 0) > 0 or ac.get("abrupt_tail_cut", False):
        return False
    return all("hızlı" in iss.lower() or "wpm" in iss.lower() for iss in val.issues) and len(val.issues) > 0


def apply_controlled_atempo(
    audio: np.ndarray,
    sr: int,
    factor: float,
    temp_dir: Path,
) -> np.ndarray:
    """FFmpeg atempo filtresiyle kontrollü hız telafisi uygular (orijinal diziyi değiştirmez)."""
    import subprocess
    import soundfile as sf
    factor = max(0.80, min(0.96, float(factor)))
    temp_dir.mkdir(parents=True, exist_ok=True)
    in_p = temp_dir / "atempo_in.wav"
    out_p = temp_dir / "atempo_out.wav"
    sf.write(in_p, audio.astype(np.float32), sr)
    subprocess.check_call([
        "ffmpeg", "-y", "-loglevel", "error", "-threads", "2",
        "-i", str(in_p), "-filter:a", f"atempo={factor:.3f}",
        "-ar", str(sr), "-ac", "1", str(out_p),
    ])
    slowed, _ = sf.read(out_p, dtype="float32")
    if slowed.ndim > 1:
        slowed = np.mean(slowed, axis=1)
    return slowed.astype(np.float32)


ADAPTIVE_SUFFIX_PROFILES = [
    ("adaptive_low_rep_1", 0.15, 0.80, 0.55, 5.0, 12),
    ("adaptive_low_rep_2", 0.25, 0.78, 0.65, 4.0, 12),
    ("adaptive_low_rep_3", 0.20, 0.76, 0.60, 3.0, 12),
]

STANDARD_RETRY_PROFILES = [
    ("std_primary", 0.22, 0.84, 0.65, 10.0, 16),
    ("std_fallback_1", 0.28, 0.83, 0.65, 10.0, 16),
    ("std_fallback_2", 0.18, 0.84, 0.65, 10.0, 16),
    ("std_fallback_3", 0.32, 0.82, 0.65, 10.0, 16),
]


class XTTSEngine:
    """
    XTTS-v2 üretim ve doğrulama orkestratörü.
    Testlerde synthesizer_fn ve transcriber_fn enjekte edilerek GPU olmadan da
    tüm yeniden deneme ve durum geçiş mantığı deterministik olarak test edilebilir.
    """

    def __init__(
        self,
        config: Optional[XTTSConfig] = None,
        thresholds: Optional[AudioThresholds] = None,
        synthesizer_fn: Optional[Callable[[str, float, float, float], np.ndarray]] = None,
        transcriber_fn: Optional[Callable[[np.ndarray, int], Tuple[str, List[Dict[str, Any]]]]] = None,
        known_proper_nouns: Optional[List[str]] = None,
    ) -> None:
        self.config = config or XTTSConfig()
        self.thresholds = thresholds or AudioThresholds()
        self._synthesizer_fn = synthesizer_fn
        self._transcriber_fn = transcriber_fn
        self.known_proper_nouns = known_proper_nouns or []
        self._tts_model: Any = None

    def load_models(self, device: str = "cuda") -> None:
        if self._synthesizer_fn is not None:
            return
        if device != "cuda":
            raise RuntimeError("XTTS-v2 üretimi için CUDA/T4 GPU gereklidir; CPU üzerinde sessiz çalıştırma engellendi.")
        import os
        os.environ["COQUI_TOS_AGREED"] = "1"
        from TTS.api import TTS  # type: ignore

        self._tts_model = TTS(self.config.model_name).to(device)
        try:
            args = self._tts_model.synthesizer.tts_config.model_args
            args.gpt_cond_len = self.config.gpt_cond_len
            args.gpt_cond_chunk_len = self.config.gpt_cond_chunk_len
            args.max_ref_len = self.config.max_ref_len
        except Exception:
            pass

    def unload_models(self) -> None:
        if self._tts_model is not None:
            del self._tts_model
            self._tts_model = None
        gc.collect()
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def synthesize_sentence_with_qc(
        self,
        sentence_text: str,
        reference_wav: Optional[str] = None,
        temp_dir: Optional[Path] = None,
    ) -> SentenceSynthesisResult:
        """
        Cümleyi kontrollü parametre profilleriyle üretir ve her denemeyi bağımsız QC'den geçirir.
        - Kesim (tail trim) uygulanırsa kesim öncesi ve sonrası doğrulanır.
        - Maksimum denemede onay alınamazsa MANUAL_REVIEW döner (ASLA APPROVED dönmez!).
        """
        sr = self.config.sample_rate
        history: List[Dict[str, Any]] = []
        best_audio: Optional[np.ndarray] = None
        best_raw_audio: Optional[np.ndarray] = None
        best_val: Optional[AudioValidationResult] = None
        best_params: Dict[str, Any] = {}
        best_score = -1e9

        for attempt_idx, (temp, speed, top_p) in enumerate(self.config.retry_profiles, 1):
            params = {
                "temperature": temp,
                "speed": speed,
                "top_p": top_p,
                "top_k": self.config.top_k,
                "repetition_penalty": self.config.repetition_penalty,
                "split_sentences": self.config.split_sentences,
            }
            if self._synthesizer_fn is not None:
                raw_audio = self._synthesizer_fn(sentence_text, temp, speed, top_p)
            else:
                if self._tts_model is None or not reference_wav:
                    raise RuntimeError("XTTS modeli yüklenmedi veya referans ses yolu eksik.")
                import soundfile as sf  # type: ignore
                out_raw = str((temp_dir or Path("/tmp")) / f"xtts_raw_a{attempt_idx}.wav")
                self._tts_model.tts_to_file(
                    text=sentence_text,
                    speaker_wav=reference_wav,
                    language=self.config.language,
                    file_path=out_raw,
                    speed=speed,
                    temperature=temp,
                    repetition_penalty=self.config.repetition_penalty,
                    top_k=self.config.top_k,
                    top_p=top_p,
                    split_sentences=self.config.split_sentences,
                )
                raw_audio, _ = sf.read(out_raw, dtype="float32")
                if raw_audio.ndim > 1:
                    raw_audio = np.mean(raw_audio, axis=1)

            # 1. Ham ses üzerinde ön doğrulama (son kelime zaman damgası için)
            if self._transcriber_fn is None:
                raise RuntimeError("Whisper doğrulayıcı fonksiyonu tanımlı değil.")
            raw_text, raw_words = self._transcriber_fn(raw_audio, sr)
            last_word_end = max((float(w.get("end", 0.0)) for w in raw_words), default=None)

            # 2. Güvenli hayalet temizliği ve doğal temizlik
            trimmed_audio, was_trimmed = safe_trim_trailing_ghost(raw_audio, sr, last_word_end_sec=last_word_end)
            proc_audio = gentle_voice_cleanup(trimmed_audio, sr)

            # 3. İşlenmiş nihai ses üzerinde bağımsız Whisper + Akustik QC
            proc_text, proc_words = self._transcriber_fn(proc_audio, sr)
            val = validate_sentence_audio(
                expected_text=sentence_text,
                transcribed_text=proc_text,
                audio_signal=proc_audio,
                sr=sr,
                word_timestamps=proc_words,
                thresholds=self.thresholds,
                known_proper_nouns=self.known_proper_nouns,
            )

            # Kesim sonrası son kelime bozulduysa kesilmemiş versiyona geri dön ve tekrar doğrula
            if was_trimmed and (not val.last_word_ok or not val.last_syllable_ok):
                fallback_audio = gentle_voice_cleanup(raw_audio, sr)
                fb_text, fb_words = self._transcriber_fn(fallback_audio, sr)
                fb_val = validate_sentence_audio(
                    expected_text=sentence_text,
                    transcribed_text=fb_text,
                    audio_signal=fallback_audio,
                    sr=sr,
                    word_timestamps=fb_words,
                    thresholds=self.thresholds,
                    known_proper_nouns=self.known_proper_nouns,
                )
                if fb_val.word_alignment_ratio >= val.word_alignment_ratio:
                    proc_audio, val = fallback_audio, fb_val

            rank_score = val.word_alignment_ratio - val.wer - (0.5 if not val.last_word_ok else 0.0)
            history.append({
                "attempt": attempt_idx,
                "params": params,
                "passed": bool(val.passed),
                    "needs_manual_review": bool(val.needs_manual_review),
                "needs_manual_review": val.needs_manual_review,
                "wer": val.wer,
                "cer": val.cer,
                "issues": val.issues,
            })

            if rank_score > best_score:
                best_score = rank_score
                best_audio = proc_audio
                best_raw_audio = raw_audio
                best_val = val
                best_params = params

            if val.passed and not val.needs_manual_review:
                return SentenceSynthesisResult(
                    sentence_text=sentence_text,
                    status=ArtifactStatus.APPROVED,
                    audio=proc_audio,
                    attempts=attempt_idx,
                    used_params=params,
                    validation=val,
                    history=history,
                    raw_audio=raw_audio,
                )

        # Maksimum deneme bitti ve onay alınamadı -> KESİNLİKLE MANUAL_REVIEW (ASLA APPROVED DEĞİL!)
        return SentenceSynthesisResult(
            sentence_text=sentence_text,
            status=ArtifactStatus.MANUAL_REVIEW,
            audio=best_audio,
            attempts=len(self.config.retry_profiles),
            used_params=best_params,
            validation=best_val,
            history=history,
            raw_audio=best_raw_audio,
        )

    @staticmethod
    def concat_sentences_safely(
        segments: List[np.ndarray],
        sr: int,
        gap_sec: float = 1.10,
    ) -> Tuple[np.ndarray, List[str]]:
        """Cümleleri birleştirir ve her birleşim noktasında tıklama/seviye sıçraması denetler."""
        splice_warnings: List[str] = []
        if not segments:
            return np.zeros(0, dtype=np.float32), splice_warnings
        if len(segments) == 1:
            return segments[0].astype(np.float32), splice_warnings

        gap = np.zeros(int(gap_sec * sr), dtype=np.float32)
        parts: List[np.ndarray] = []
        for idx, seg in enumerate(segments):
            if idx > 0:
                ok, warns = check_splice_boundary(segments[idx - 1], seg, sr)
                if not ok:
                    splice_warnings.extend([f"Cümle {idx}->{idx+1}: {w}" for w in warns])
                parts.append(gap)
            parts.append(seg.astype(np.float32))
        return np.concatenate(parts), splice_warnings

    def synthesize_sentence_v22(
        self,
        sentence_text: str,
        raw_sentence_for_split: Optional[str] = None,
        pronunciation_dict: Any = None,
        reference_wav: Optional[str] = None,
        temp_dir: Optional[Path] = None,
        enable_atempo_compensation: bool = True,
        enable_adaptive_profiles: bool = True,
        atempo_fn: Optional[Callable[[np.ndarray, int, float, Path], np.ndarray]] = None,
    ) -> SentenceSynthesisResult:
        """
        V2.2 Kontrollü Üretim Orkestratörü (fail-closed):
        1. Uzun virgüllü cümlelerde doğal nefes grubu bölmesi (orijinal metin/SRT değişmez).
        2. İlk denemede son hece/kelime hatası görülürse uyarlanabilir XTTS profillerine geçer.
        3. Metin %100 doğruysa (WER=0, CER=0) ama yalnızca WPM hafif hızlıysa kontrollü atempo
           uygular ve sonucu YENİDEN Whisper + akustik QC'den geçirir.
        4. Onay almayan hiçbir çıktı APPROVED olmaz -> MANUAL_REVIEW.
        """
        from .text_normalizer import split_long_sentence_for_tts
        from .mastering import apply_true_peak_limiter

        sr = self.config.sample_rate
        t_dir = temp_dir or Path("/tmp")
        atempo_impl = atempo_fn or apply_controlled_atempo

        # Nefes grupları (yalnızca uzun virgüllü cümlelerde bölünür)
        if raw_sentence_for_split:
            breath_groups = split_long_sentence_for_tts(raw_sentence_for_split, pronunciation_dict=pronunciation_dict)
        else:
            breath_groups = [sentence_text]
        if not breath_groups:
            breath_groups = [sentence_text]

        history: List[Dict[str, Any]] = []
        best_audio: Optional[np.ndarray] = None
        best_raw_audio: Optional[np.ndarray] = None
        best_val: Optional[AudioValidationResult] = None
        best_params: Dict[str, Any] = {}
        best_profile = ""
        best_score = -1e9

        queue = list(STANDARD_RETRY_PROFILES[:1])
        switched_adaptive = False
        attempt_idx = 0
        max_attempts = 4

        while attempt_idx < max_attempts:
            if not queue:
                break
            attempt_idx += 1
            prof_name, temp, speed, top_p, rep_pen, gpt_cond = queue.pop(0)
            params = {
                "profile_name": prof_name,
                "temperature": temp,
                "speed": speed,
                "top_p": top_p,
                "top_k": self.config.top_k,
                "repetition_penalty": rep_pen,
                "gpt_cond_len": gpt_cond,
                "breath_groups": len(breath_groups),
            }

            if self._synthesizer_fn is not None:
                if len(breath_groups) == 1:
                    raw_audio = self._synthesizer_fn(breath_groups[0], temp, speed, top_p)
                else:
                    parts = [self._synthesizer_fn(g, temp, speed, top_p) for g in breath_groups]
                    gap = np.zeros(int(0.25 * sr), dtype=np.float32)
                    joined: List[np.ndarray] = []
                    for gi, p_arr in enumerate(parts):
                        if gi > 0:
                            joined.append(gap)
                        joined.append(p_arr)
                    raw_audio = np.concatenate(joined)
            else:
                if self._tts_model is None or not reference_wav:
                    raise RuntimeError("XTTS modeli yüklenmedi veya referans ses yolu eksik.")
                xtts_model = self._tts_model.synthesizer.tts_model
                cond_lat, spk_emb = xtts_model.get_conditioning_latents(
                    audio_path=[str(reference_wav)], gpt_cond_len=gpt_cond
                )
                parts = []
                for g in breath_groups:
                    out = xtts_model.inference(
                        text=g, language=self.config.language,
                        gpt_cond_latent=cond_lat, speaker_embedding=spk_emb,
                        temperature=temp, top_p=top_p, top_k=self.config.top_k,
                        repetition_penalty=rep_pen, speed=speed,
                    )
                    rw = out["wav"]
                    w_arr = rw.detach().cpu().numpy().astype(np.float32) if hasattr(rw, "detach") else np.asarray(rw, dtype=np.float32)
                    parts.append(np.squeeze(w_arr))
                if len(parts) == 1:
                    raw_audio = parts[0]
                else:
                    gap = np.zeros(int(0.25 * sr), dtype=np.float32)
                    joined = []
                    for gi, p_arr in enumerate(parts):
                        if gi > 0:
                            joined.append(gap)
                        joined.append(gentle_voice_cleanup(p_arr, sr))
                    raw_audio = np.concatenate(joined)

            if self._transcriber_fn is None:
                raise RuntimeError("Whisper doğrulayıcı fonksiyonu tanımlı değil.")

            raw_text, raw_words = self._transcriber_fn(raw_audio, sr)
            last_word_end = max((float(w.get("end", 0.0)) for w in raw_words), default=None)
            trimmed_audio, was_trimmed = safe_trim_trailing_ghost(raw_audio, sr, last_word_end_sec=last_word_end)
            proc_audio = apply_true_peak_limiter(gentle_voice_cleanup(trimmed_audio, sr), sr=sr, ceiling_dbtp=-1.5)

            proc_text, proc_words = self._transcriber_fn(proc_audio, sr)
            val = validate_sentence_audio(
                expected_text=sentence_text,
                transcribed_text=proc_text,
                audio_signal=proc_audio,
                sr=sr,
                word_timestamps=proc_words,
                thresholds=self.thresholds,
                known_proper_nouns=self.known_proper_nouns,
            )

            atempo_used: Optional[float] = None
            # V2.2 Kontrollü Otomatik Hız Telafisi (yalnızca WER=0, CER=0 ve tek sorun hızlı WPM ise)
            if enable_atempo_compensation and (not val.passed) and is_pure_fast_wpm_only(val):
                ac = val.acoustic if isinstance(val.acoustic, dict) else {}
                wpm_now = float(ac.get("wpm", 160.0))
                target_max = float(ac.get("applied_max_wpm", 145.0))
                factor = max(0.82, min(0.95, (target_max - 10.0) / max(1.0, wpm_now)))
                slowed_audio = apply_true_peak_limiter(atempo_impl(proc_audio, sr, factor, t_dir), sr=sr, ceiling_dbtp=-1.5)
                s_text, s_words = self._transcriber_fn(slowed_audio, sr)
                s_val = validate_sentence_audio(
                    expected_text=sentence_text,
                    transcribed_text=s_text,
                    audio_signal=slowed_audio,
                    sr=sr,
                    word_timestamps=s_words,
                    thresholds=self.thresholds,
                    known_proper_nouns=self.known_proper_nouns,
                )
                if s_val.passed and not s_val.needs_manual_review:
                    proc_audio, val, atempo_used = slowed_audio, s_val, round(factor, 3)

            rank_score = val.word_alignment_ratio - val.wer - (0.5 if not val.last_word_ok else 0.0)
            history.append({
                "attempt": attempt_idx,
                "profile_name": prof_name,
                "params": params,
                "passed": bool(val.passed),
                    "needs_manual_review": bool(val.needs_manual_review),
                "wer": val.wer,
                "cer": val.cer,
                "last_syllable_ok": val.last_syllable_ok,
                "atempo_applied": atempo_used,
                "issues": val.issues,
            })

            if rank_score > best_score:
                best_score = rank_score
                best_audio = proc_audio
                best_raw_audio = raw_audio
                best_val = val
                best_params = params
                best_profile = prof_name

            if val.passed and not val.needs_manual_review:
                return SentenceSynthesisResult(
                    sentence_text=sentence_text,
                    status=ArtifactStatus.APPROVED,
                    audio=proc_audio,
                    attempts=attempt_idx,
                    used_params=params,
                    validation=val,
                    history=history,
                    raw_audio=raw_audio,
                    profile_used=prof_name,
                    atempo_applied=atempo_used,
                )

            # Uyarlanabilir yeniden deneme profili seçimi (yalnızca 1. başarısız denemeden sonra)
            if attempt_idx == 1:
                if enable_adaptive_profiles and (not val.last_syllable_ok or not val.last_word_ok):
                    queue.extend(ADAPTIVE_SUFFIX_PROFILES)
                    switched_adaptive = True
                else:
                    queue.extend(STANDARD_RETRY_PROFILES[1:])

        return SentenceSynthesisResult(
            sentence_text=sentence_text,
            status=ArtifactStatus.MANUAL_REVIEW,
            audio=best_audio,
            attempts=attempt_idx,
            used_params=best_params,
            validation=best_val,
            history=history,
            raw_audio=best_raw_audio,
            profile_used=best_profile,
            atempo_applied=None,
        )
