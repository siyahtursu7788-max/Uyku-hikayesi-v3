#!/usr/bin/env python3
"""
UYKU_HIKAYESI_V3_PATCHED — BÜTÜNLEŞİK TEK ANA ÜRETİM ORKESTRATÖRÜ (v3_runner.py)
================================================================================
Düzeltilen 2 kritik sorun:
  1) ZIP otomatik bulma, SHA-256 doğrulama, atomik çıkarma ve sys.path kurulumu (idempotent).
  2) Tüm mevcut V3 modüllerinin tek akışta gerçek girdi/çıktı, fail-closed manifest
     kapıları ve VRAM boşaltma ile birbirine bağlanması:
     XTTS-v2 -> Whisper QC -> AudioLDM Ambiyans -> Ses Mastering ->
     SDXL/Imagen Görseller -> SVD-XT Image-to-Video + CLIP/OpenCV/Loop ->
     Whisper Turbo SRT -> FFmpeg 1080p/24fps Kurgu ve Nihai MP4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

EXPECTED_V3_PATCHED_ZIP_SHA256 = "__SYNCED_ON_BUILD__"
DEFAULT_V3_PATCHED_DRIVE_ROOT = Path("/content/drive/MyDrive/V3_PATCHED")
DEFAULT_V3_LOCAL_PKG_ROOT = Path("/content/uyku_v3_patched_pkg")


def ensure_v3_package_extracted(
    zip_path: Optional[Path] = None,
    extract_root: Path = DEFAULT_V3_LOCAL_PKG_ROOT,
    expected_sha256: Optional[str] = None,
) -> Path:
    """
    ZIP paketini bulur, SHA-256 doğrulaması yapar, idempotent şekilde çıkarır ve
    uyku_v3 paketini sys.path'e ekler. Yerel kaynak klasörü zaten yanındaysa onu kullanır.
    """
    here = Path(__file__).resolve().parent
    if (here / "uyku_v3" / "__init__.py").exists() and zip_path is None:
        if str(here) not in sys.path:
            sys.path.insert(0, str(here))
        return here

    candidate_zips = [
        p
        for p in [
            Path(zip_path) if zip_path else None,
            here / "UYKU_HIKAYESI_V3_PATCHED.zip",
            DEFAULT_V3_PATCHED_DRIVE_ROOT / "UYKU_HIKAYESI_V3_PATCHED.zip",
        ]
        if p is not None and p.exists()
    ]
    if not candidate_zips:
        raise FileNotFoundError(
            "UYKU_HIKAYESI_V3_PATCHED.zip bulunamadı ve yerel uyku_v3/ dizini yok."
        )
    zpath = candidate_zips[0]
    actual_sha = hashlib.sha256(zpath.read_bytes()).hexdigest()

    exp = expected_sha256
    if not exp and EXPECTED_V3_PATCHED_ZIP_SHA256 != "__SYNCED_ON_BUILD__":
        exp = EXPECTED_V3_PATCHED_ZIP_SHA256
    sha_sidecar = zpath.with_suffix(".zip.sha256")
    if not exp and sha_sidecar.exists():
        exp = sha_sidecar.read_text(encoding="utf-8").strip().split()[0]

    if exp and actual_sha != exp:
        raise RuntimeError(
            f"FAIL-CLOSED: V3_PATCHED ZIP SHA-256 uyuşmuyor! beklenen={exp}, bulunan={actual_sha}"
        )

    extract_root = Path(extract_root)
    marker = extract_root / ".v3_patched_zip_sha256"
    pkg_root = extract_root / "V3_PATCHED"
    ready = (pkg_root / "uyku_v3" / "__init__.py").exists()
    cached_sha = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""

    if not (ready and cached_sha == actual_sha):
        extract_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix=".extract_v3_", dir=str(extract_root)))
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(tmp_dir)
            extracted_sub = tmp_dir / "V3_PATCHED"
            if not (extracted_sub / "uyku_v3" / "__init__.py").exists():
                raise RuntimeError(f"ZIP içinde V3_PATCHED/uyku_v3 bulunamadı: {zpath}")
            if pkg_root.exists():
                shutil.rmtree(pkg_root)
            shutil.move(str(extracted_sub), str(pkg_root))
            marker.write_text(actual_sha, encoding="utf-8")
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    return pkg_root


BOLUM_CUMLE_ARASI_SN: Dict[str, float] = {
    "GIRIS": 1.00,
    "BOLUM_01": 1.00,
    "BOLUM_02": 1.05,
    "BOLUM_03": 1.15,
    "BOLUM_04": 1.20,
    "BOLUM_05": 1.35,
    "BOLUM_06": 1.45,
    "BOLUM_07": 1.50,
    "BOLUM_08": 1.75,
    "BOLUM_09": 2.00,
}

BOLUM_AMBIYANS: Dict[str, str] = {
    "GIRIS": "ruzgar",
    "BOLUM_01": "ruzgar",
    "BOLUM_02": "kutuphane",
    "BOLUM_03": "ruzgar",
    "BOLUM_04": "dere_bahce",
    "BOLUM_05": "kutuphane",
    "BOLUM_06": "yagmur",
    "BOLUM_07": "somine",
    "BOLUM_08": "yagmur",
    "BOLUM_09": "deniz",
}

BOLUM_MOTION_PROFILE: Dict[str, str] = {
    "GIRIS": "indoor_calm",
    "BOLUM_01": "indoor_calm",
    "BOLUM_02": "candle_lamp",
    "BOLUM_03": "still_object",
    "BOLUM_04": "ocean_waves",
    "BOLUM_05": "candle_lamp",
    "BOLUM_06": "rain_window",
    "BOLUM_07": "fireplace",
    "BOLUM_08": "rain_window",
    "BOLUM_09": "ocean_waves",
}


def parse_script_v3(script_path: Path, pron_dict: Any) -> List[Dict[str, Any]]:
    from uyku_v3.audio.text_normalizer import group_short_sentences, normalize_for_tts
    from uyku_v3.core.file_manager import sha256_text

    lines = script_path.read_text(encoding="utf-8").splitlines()
    paragraphs: List[Dict[str, Any]] = []
    active_section = "GIRIS"
    has_header = any(
        ln.strip().startswith("GİRİŞ") or re.match(r"^BÖLÜM\s+\d+", ln.strip()) for ln in lines
    )
    started = not has_header
    p_idx = 0

    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        if s.startswith("GİRİŞ"):
            started = True
            active_section = "GIRIS"
            continue
        m_b = re.match(r"^BÖLÜM\s+(\d+)", s)
        if m_b:
            started = True
            active_section = f"BOLUM_{int(m_b.group(1)):02d}"
            continue
        if s in ("SON", "[SON]"):
            break
        if not started or s.startswith("====") or (s.startswith("[") and s.endswith("]")):
            continue

        raw_sents = [p.strip() for p in re.split(r"(?<=[.!?])\s+|;\s*", s) if p.strip()]
        norm_sents = [normalize_for_tts(rs, pronunciation_dict=pron_dict) for rs in raw_sents]
        norm_sents = [x for x in norm_sents if x]
        grouped_norm = group_short_sentences(norm_sents)
        raw_aligned = raw_sents if len(raw_sents) == len(grouped_norm) else grouped_norm

        p_id = f"paragraf_{p_idx:03d}"
        paragraphs.append(
            {
                "p_id": p_id,
                "p_idx": p_idx,
                "section": active_section,
                "ambience_key": BOLUM_AMBIYANS.get(active_section, "ruzgar"),
                "motion_profile": BOLUM_MOTION_PROFILE.get(active_section, "indoor_calm"),
                "gap_sec": BOLUM_CUMLE_ARASI_SN.get(active_section, 1.25),
                "raw_paragraph": s,
                "norm_sentences": grouped_norm,
                "raw_sentences": raw_aligned,
                "source_hash": sha256_text(s),
            }
        )
        p_idx += 1

    return paragraphs


def safe_begin_generation(manifest: Any, artifact_id: str) -> None:
    from uyku_v3.core.state_manager import ArtifactStatus

    rec = manifest.get(artifact_id)
    if rec is None:
        raise KeyError(f"Manifest kaydı eksik: {artifact_id}")
    st = rec.get_status()
    if st in (ArtifactStatus.GENERATING, ArtifactStatus.VALIDATING):
        manifest.set_status(artifact_id, ArtifactStatus.INTERRUPTED, error_message="Yarım kalan oturum kapatıldı")
        manifest.set_status(artifact_id, ArtifactStatus.PENDING, error_message="Yeniden üretim kuyruğu")
    elif st in (
        ArtifactStatus.INTERRUPTED,
        ArtifactStatus.REJECTED,
        ArtifactStatus.MANUAL_REVIEW,
        ArtifactStatus.FAILED,
    ):
        manifest.set_status(artifact_id, ArtifactStatus.PENDING, error_message="Yeniden deneme hazırlığı")
    manifest.set_status(artifact_id, ArtifactStatus.GENERATING)


def build_default_protected_map(studio_dir: Path) -> Dict[str, Path]:
    protected: Dict[str, Path] = {
        "live_v2_manifest": studio_dir / "bolum_yedekleri/uretim_manifesti_v2.json",
        "orig_script": studio_dir / "Unutulmus_Zamanlar_Arsivi_Final_v2.txt",
        "referans_ses": studio_dir / "referans_ses_15sn.wav",
        "v1_ses_py": studio_dir / "unutulmus_zamanlar_ses_ve_master.py",
        "v1_gorsel_py": studio_dir / "unutulmus_zamanlar_gorsel_video_srt.py",
        "v21_runner": studio_dir / "unutulmus_zamanlar_v2_runner.py",
        "v21_zip": Path("/content/drive/MyDrive/UYKU_BORU_HATTI_PROFESYONEL_V2/02_GUNCELLENMIS_KODLAR/UYKU_BORU_HATTI_V2_PAKET.zip"),
        "v22_zip": Path("/content/drive/MyDrive/UYKU_BORU_HATTI_PROFESYONEL_V2/02_GUNCELLENMIS_KODLAR/UYKU_BORU_HATTI_V2_2_PAKET.zip"),
        "orig_v3_zip": Path("/content/drive/MyDrive/UYKU_HIKAYESI_V3_FINAL_PRODUCTION/UYKU_HIKAYESI_V3_FINAL_PRODUCTION.zip"),
    }
    for aid in ["paragraf_001", "paragraf_003", "paragraf_007", "paragraf_009", "paragraf_017", "paragraf_018", "paragraf_021"]:
        protected[f"v2_approved_{aid}"] = studio_dir / "bolum_yedekleri/paragraflar_v2" / f"{aid}.wav"
    return protected


def _read_wav_f32(path: Path) -> Tuple[Any, int]:
    import numpy as np
    from scipy.io import wavfile

    try:
        import soundfile as sf
        arr, sr = sf.read(str(path), dtype="float32")
        return np.asarray(arr, dtype=np.float32), int(sr)
    except Exception:
        sr, data = wavfile.read(str(path))
        if np.issubdtype(data.dtype, np.integer):
            arr = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
        else:
            arr = data.astype(np.float32)
        return arr, int(sr)


def _write_wav_pcm16(path: Path, audio: Any, sr: int) -> None:
    import numpy as np
    from scipy.io import wavfile

    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    try:
        import soundfile as sf
        sf.write(str(path), clipped, int(sr), subtype="PCM_16")
    except Exception:
        wavfile.write(str(path), int(sr), (clipped * 32767.0).astype(np.int16))


def _build_scene_specs(paragraphs: List[Dict[str, Any]], continuity_mgr: Any) -> List[Any]:
    from uyku_v3.visuals.scene_manager import SceneSpec

    specs: List[SceneSpec] = []
    # Her 8 paragrafta veya bölüm değişiminde yeni sahne oluştur
    if not paragraphs:
        return specs
    groups: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    for par in paragraphs:
        if cur and (par["section"] != cur[0]["section"] or len(cur) >= 8):
            groups.append(cur)
            cur = [par]
        else:
            cur.append(par)
    if cur:
        groups.append(cur)

    for idx, grp in enumerate(groups, start=1):
        first_p, last_p = grp[0], grp[-1]
        p_code_s = f"P{first_p['p_idx']:03d}"
        p_code_e = f"P{last_p['p_idx']:03d}"
        loc_desc = f"Nocturnal atmospheric scene for {first_p['section']}: {first_p['raw_paragraph'][:140]}"
        prompt_obj = continuity_mgr.build_scene_prompt(
            scene_id=f"sahne_{idx:02d}",
            location_desc=loc_desc,
            allow_characters=False,
        )
        specs.append(
            SceneSpec(
                scene_index=idx,
                image_filename=f"sahne_{idx:02d}_{first_p['section'].lower()}.jpg",
                start_paragraph=p_code_s,
                end_paragraph=p_code_e,
                motion_profile_name=first_p["motion_profile"],
                clip_query=f"dark nocturnal calm sleep scene {first_p['section'].lower()}",
                sdxl_prompt=prompt_obj["positive_prompt"],
            )
        )
    return specs


def run_v3_pipeline(
    cfg: Dict[str, Any],
    script_path: Path,
    ref_audio: Path,
    v3_out_dir: Path,
    local_work: Path,
    stages_req: set[str],
    max_paragraphs: int = 0,
    stop_after: int = 0,
    dry_run: bool = False,
    protected_override: Optional[Dict[str, Path]] = None,
    backends: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Tüm V3 modüllerini tek bütünleşik üretim akışında yürütür.
    `backends` yalnızca test amaçlı sahte GPU kancaları için kullanılır; üretimde None'dır.
    """
    import numpy as np
    import scipy.signal as sps

    from uyku_v3.ambience.audioldm_engine import AudioLDMAmbienceEngine
    from uyku_v3.assembly.ffmpeg_engine import FFmpegAssemblyEngine
    from uyku_v3.assembly.final_export import export_verified_final_video
    from uyku_v3.assembly.timeline import format_youtube_chapters
    from uyku_v3.audio.mastering import MasteringEngine
    from uyku_v3.audio.pronunciation_dictionary import PronunciationDictionary
    from uyku_v3.audio.sentence_cache import SentenceAudioCache
    from uyku_v3.audio.xtts_engine import XTTSEngine
    from uyku_v3.config.model_config import SDXLConfig, SVDConfig, XTTSConfig
    from uyku_v3.config.quality_thresholds import (
        AssemblyThresholds,
        AudioThresholds,
        ImageThresholds,
        SRTThresholds,
        VideoThresholds,
    )
    from uyku_v3.core.file_manager import atomic_copy_file, atomic_write_json, sha256_file, sha256_text
    from uyku_v3.core.logging_system import PipelineLogger
    from uyku_v3.core.manifest import ManifestManager
    from uyku_v3.core.protection import LegacyProtectionGuard
    from uyku_v3.core.recovery import RecoveryManager
    from uyku_v3.core.resource_monitor import release_gpu_vram, snapshot_system_resources
    from uyku_v3.core.state_manager import ArtifactStatus
    from uyku_v3.subtitles.alignment import align_sentences_to_words
    from uyku_v3.subtitles.srt_generator import generate_srt_cues, write_srt_file
    from uyku_v3.subtitles.srt_validator import validate_srt_file
    from uyku_v3.subtitles.transcription import transcribe_audio_16k
    from uyku_v3.video.clip_validator import CLIPValidator
    from uyku_v3.video.svd_engine import SVDEngine
    from uyku_v3.visuals.continuity_manager import VisualContinuityManager
    from uyku_v3.visuals.image_generator import ImageGenerator

    backends = backends or {}
    studio_dir = Path(cfg["paths"]["drive_studio_dir"])
    for d in (
        v3_out_dir / "paragraflar",
        v3_out_dir / "cumle_onbellek",
        v3_out_dir / "karantina",
        v3_out_dir / "gorseller",
        v3_out_dir / "video_klipler_svd",
        v3_out_dir / "sahne_uzatilmis",
        v3_out_dir / "ambiyans",
        v3_out_dir / "final",
        v3_out_dir / "raporlar",
        v3_out_dir / "loglar",
        local_work,
    ):
        d.mkdir(parents=True, exist_ok=True)

    logger = PipelineLogger(v3_out_dir / "loglar", "uyku_v3_patched", echo=True)
    guard = LegacyProtectionGuard(
        protected_override if protected_override is not None else build_default_protected_map(studio_dir)
    )
    guard.snapshot_before(strict_missing=False)

    t0 = time.time()
    manifest = ManifestManager(v3_out_dir / "v3_production_manifest.json", "uyku_v3_patched")
    recovery = RecoveryManager(manifest, v3_out_dir / "karantina")
    recovery_report = recovery.recover_after_interruption([v3_out_dir / "paragraflar", local_work])
    logger.info("RECOVERY", f"Kesinti kurtarma tamamlandı: {recovery_report}")

    assert script_path.exists(), f"Senaryo bulunamadı: {script_path}"
    pron_path = studio_dir / "telaffuz_sozlugu.json"
    pron_dict = PronunciationDictionary(path=pron_path if pron_path.exists() else None)
    paragraphs = parse_script_v3(script_path, pron_dict)
    if max_paragraphs > 0:
        paragraphs = paragraphs[:max_paragraphs]

    script_sha = sha256_file(script_path)
    ref_sha = sha256_file(ref_audio) if ref_audio.exists() else "MISSING_REF"
    cfg_sha = sha256_text(json.dumps(cfg, sort_keys=True))
    run_all = "all" in stages_req

    report: Dict[str, Any] = {
        "schema_version": "3.0.1-patched",
        "platform": "UYKU_HIKAYESI_V3_PATCHED",
        "mode": "DRY_RUN" if dry_run else ("INJECTED_TEST_BACKENDS" if backends else "LIVE_GPU_PRODUCTION"),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "script_path": str(script_path),
        "script_sha256": script_sha,
        "ref_audio_sha256": ref_sha,
        "config_sha256": cfg_sha,
        "paragraph_count": len(paragraphs),
        "sentence_count": sum(len(p["norm_sentences"]) for p in paragraphs),
        "recovery": recovery_report,
        "resources_start": snapshot_system_resources(local_work),
        "stages": {},
    }

    if dry_run:
        continuity_mgr = VisualContinuityManager()
        planned_scenes = _build_scene_specs(paragraphs, continuity_mgr)
        report["stages"]["dry_run_plan"] = {
            "status": "OK",
            "planned_paragraphs": len(paragraphs),
            "planned_scenes": len(planned_scenes),
            "stage_order": ["voice", "ambience", "mastering", "visuals", "video", "subtitles", "assembly"],
        }
        report["protection_sha256"] = guard.verify_after()
        report["resources_end"] = snapshot_system_resources(local_work)
        report["total_duration_sec"] = round(time.time() - t0, 2)
        rep_p = v3_out_dir / "raporlar" / f"v3_patched_dry_run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
        atomic_write_json(rep_p, report)
        report["report_path"] = str(rep_p)
        return report

    # Gerçek üretimde NVIDIA T4 CUDA zorunlu (test backends verilmediyse)
    device = "cpu" if backends else "cuda"
    if not backends:
        import torch
        assert bool(getattr(torch.cuda, "is_available")()), "HATA: Canlı üretim için CUDA GPU gereklidir!"
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("GPU", f"Aktif GPU doğrulandı: {gpu_name}")
        report["gpu_device_name"] = gpu_name

    assert ref_audio.exists(), f"Referans ses bulunamadı: {ref_audio}"

    xtts_cfg = XTTSConfig()
    audio_th = AudioThresholds()
    outro_sec = float(cfg.get("ambience", {}).get("outro_duration_sec", 160.0))
    outro_fade_sec = float(cfg.get("ambience", {}).get("outro_fade_out_sec", 40.0))

    # =========================================================================
    # AŞAMA 1: XTTS-v2 SESLENDİRME + WHISPER TURBO QC + ÖNBELLEK
    # =========================================================================
    if run_all or "voice" in stages_req:
        t_s = time.time()
        sent_cache = SentenceAudioCache(v3_out_dir / "cumle_onbellek", sample_rate=xtts_cfg.sample_rate)
        w_ref, sr_r = _read_wav_f32(ref_audio)
        if w_ref.ndim > 1:
            w_ref = np.mean(w_ref, axis=1)
        if sr_r != xtts_cfg.sample_rate:
            w_ref = sps.resample(w_ref, int(len(w_ref) * xtts_cfg.sample_rate / sr_r)).astype(np.float32)
        ref_local = local_work / "ref_24k.wav"
        _write_wav_pcm16(ref_local, (w_ref / (np.max(np.abs(w_ref)) + 1e-9) * 0.84), xtts_cfg.sample_rate)

        asr_model = None
        if "whisper_fn" in backends:
            whisper_fn = backends["whisper_fn"]
        else:
            import whisper
            asr_model = whisper.load_model(cfg["whisper"]["model_name"], device="cuda")

            def whisper_fn(audio_arr: np.ndarray, sr_in: int) -> Tuple[str, List[Dict[str, Any]]]:
                tmp_w = local_work / "v3_whisper_qc.wav"
                _write_wav_pcm16(tmp_w, audio_arr, sr_in)
                out_w = asr_model.transcribe(str(tmp_w), language="tr", fp16=True, word_timestamps=True)
                words = [
                    {"word": w.get("word", ""), "start": float(w.get("start", 0.0)), "end": float(w.get("end", 0.0))}
                    for seg in out_w.get("segments", []) for w in seg.get("words", [])
                ]
                return str(out_w.get("text", "")), words

        engine = XTTSEngine(
            config=xtts_cfg,
            thresholds=audio_th,
            synthesizer_fn=backends.get("xtts_synth_fn"),
            transcriber_fn=whisper_fn,
            known_proper_nouns=cfg["voice"]["known_proper_nouns"],
        )
        if "xtts_synth_fn" not in backends:
            engine.load_models(device="cuda")

        profile_stats: Dict[str, Dict[str, int]] = {}
        reused_parags: List[str] = []
        reused_par_sents = 0
        reused_active_sents = 0
        synthesized_sents = 0
        atempo_rescued = 0
        processed_count = 0

        base_cache_params = {
            "v": "3.0.1-patched",
            "ref_sha": ref_sha[:16],
            "script_sha": script_sha[:16],
            "xtts_config": asdict(xtts_cfg),
            "audio_thresholds": asdict(audio_th),
        }

        try:
            for par in paragraphs:
                if stop_after > 0 and processed_count >= stop_after:
                    logger.info("VOICE", f"--stop-after {stop_after} kesinti noktasına ulaşıldı.")
                    break
                p_id = par["p_id"]
                wav_out = v3_out_dir / "paragraflar" / f"{p_id}.wav"
                rec = manifest.register_or_validate_cache(
                    artifact_id=p_id,
                    stage="voice_v3",
                    output_path=str(wav_out),
                    source_hash=par["source_hash"],
                    model_name="xtts_v2",
                    model_version="v3.0.1",
                    parameters={"gap": par["gap_sec"], "cfg_sha": cfg_sha[:12]},
                )
                if rec.get_status() == ArtifactStatus.APPROVED:
                    reused_parags.append(p_id)
                    reused_par_sents += len(par["norm_sentences"])
                    processed_count += 1
                    continue

                safe_begin_generation(manifest, p_id)
                wav_parts: List[np.ndarray] = []
                failed_sents: List[Dict[str, Any]] = []
                sent_records: List[Dict[str, Any]] = []

                for c_idx, (norm_t, raw_t) in enumerate(zip(par["norm_sentences"], par["raw_sentences"]), start=1):
                    c_params = {**base_cache_params, "c_idx": c_idx}
                    hit = sent_cache.get_verified(p_id, c_idx, norm_t, c_params)
                    if hit is not None:
                        wav_parts.append(hit[0])
                        reused_active_sents += 1
                        sent_records.append({"c_idx": c_idx, "status": "APPROVED", "from_cache": True})
                        continue

                    res = engine.synthesize_sentence_v22(
                        sentence_text=norm_t,
                        raw_sentence_for_split=raw_t,
                        pronunciation_dict=pron_dict,
                        reference_wav=str(ref_local),
                        temp_dir=local_work,
                        enable_atempo_compensation=True,
                        enable_adaptive_profiles=True,
                    )
                    synthesized_sents += 1
                    for h in res.history:
                        pn = h.get("profile_name", "unknown")
                        profile_stats.setdefault(pn, {"tries": 0, "passes": 0})
                        profile_stats[pn]["tries"] += 1
                        if bool(h.get("passed")) and not bool(h.get("needs_manual_review", False)):
                            profile_stats[pn]["passes"] += 1

                    if res.status == ArtifactStatus.APPROVED and res.audio is not None:
                        if res.atempo_applied is not None:
                            atempo_rescued += 1
                        sent_cache.store_approved(
                            paragraph_id=p_id,
                            sentence_index=c_idx,
                            expected_text=norm_t,
                            params=c_params,
                            audio=res.audio,
                            local_tmp_dir=local_work,
                            metrics={"profile_used": res.profile_used, "attempts": res.attempts},
                        )
                        wav_parts.append(res.audio)
                        sent_records.append({
                            "c_idx": c_idx, "status": "APPROVED", "from_cache": False,
                            "profile_used": res.profile_used, "attempts": res.attempts,
                        })
                    else:
                        vd = res.validation.to_dict() if res.validation else {}
                        ac = vd.get("acoustic") or {}
                        raw_q_path = ""
                        if res.audio is not None and len(res.audio) > 0:
                            ts_q = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
                            tmp_q = local_work / f"{p_id}_C{c_idx:02d}_{ts_q}_raw.wav"
                            dst_q = v3_out_dir / "karantina" / tmp_q.name
                            _write_wav_pcm16(tmp_q, res.audio, xtts_cfg.sample_rate)
                            atomic_copy_file(tmp_q, dst_q)
                            raw_q_path = str(dst_q)
                        fail_entry = {
                            "c_idx": c_idx,
                            "status": res.status.value,
                            "from_cache": False,
                            "text": norm_t,
                            "expected_text": vd.get("expected_text", norm_t),
                            "transcribed_text": vd.get("transcribed_text", ""),
                            "wer": vd.get("wer"),
                            "cer": vd.get("cer"),
                            "word_alignment_ratio": vd.get("word_alignment_ratio"),
                            "last_word_ok": vd.get("last_word_ok"),
                            "last_syllable_ok": vd.get("last_syllable_ok"),
                            "wpm": ac.get("wpm"),
                            "rms_db": ac.get("rms_db"),
                            "true_peak_dbtp": ac.get("true_peak_dbtp"),
                            "profile_used": res.profile_used,
                            "attempts": res.attempts,
                            "issues": vd.get("issues", []),
                            "raw_wav_path": raw_q_path,
                        }
                        failed_sents.append(fail_entry)
                        sent_records.append(fail_entry)

                manifest.set_status(p_id, ArtifactStatus.VALIDATING)
                if failed_sents:
                    manifest.set_status(
                        p_id, ArtifactStatus.MANUAL_REVIEW,
                        quality_scores={"sentences": sent_records, "failed": failed_sents},
                        error_message=f"{len(failed_sents)} cümle MANUAL_REVIEW",
                    )
                else:
                    merged, splice_warns = XTTSEngine.concat_sentences_safely(
                        wav_parts, sr=xtts_cfg.sample_rate, gap_sec=par["gap_sec"]
                    )
                    tmp_wav = local_work / f"{p_id}.wav"
                    _write_wav_pcm16(tmp_wav, merged, xtts_cfg.sample_rate)
                    out_sha = atomic_copy_file(tmp_wav, wav_out)
                    manifest.set_status(
                        p_id, ArtifactStatus.APPROVED,
                        compute_output_hash=True, quality_scores={"sentences": sent_records, "splice_warnings": splice_warns},
                    )
                processed_count += 1
        finally:
            engine.unload_models()
            del asr_model
            release_gpu_vram()

        report["stages"]["voice"] = {
            "duration_sec": round(time.time() - t_s, 2),
            "processed_paragraphs": processed_count,
            "reused_approved_paragraphs": reused_parags,
            "reused_approved_paragraph_sentences": reused_par_sents,
            "reused_cached_sentences_in_active_paragraphs": reused_active_sents,
            "synthesized_sentences": synthesized_sents,
            "atempo_rescued_sentences": atempo_rescued,
            "profile_success_stats": profile_stats,
            "paragraph_status_counts": {st.value: sum(1 for a in manifest.data["artifacts"].values() if a.get("status") == st.value) for st in ArtifactStatus},
        }
        if stop_after > 0:
            report["stopped_early_at"] = stop_after
            report["protection_sha256"] = guard.verify_after()
            report["resources_end"] = snapshot_system_resources(local_work)
            report["total_duration_sec"] = round(time.time() - t0, 2)
            rep_p = v3_out_dir / "raporlar" / f"v3_patched_stop_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
            atomic_write_json(rep_p, report)
            report["report_path"] = str(rep_p)
            return report

    # FAIL-CLOSED KAPI: Tüm hedef paragraflar APPROVED olmadan downstream aşamalara geçilemez
    unapproved_parags = [
        par["p_id"]
        for par in paragraphs
        if (manifest.get(par["p_id"]) is None)
        or (manifest.get(par["p_id"]).get_status() != ArtifactStatus.APPROVED)
    ]
    if unapproved_parags and (run_all or any(s in stages_req for s in ("ambience", "mastering", "assembly", "subtitles"))):
        logger.warning("GATE", f"FAIL-CLOSED: {len(unapproved_parags)} paragraf APPROVED değil, downstream engellendi.")
        report["stages"]["downstream_gate"] = {
            "blocked": True,
            "reason": "APPROVED olmayan paragraflar final mastering/kurgu aşamasına geçemez.",
            "unapproved_paragraphs": unapproved_parags,
        }
        report["protection_sha256"] = guard.verify_after()
        report["resources_end"] = snapshot_system_resources(local_work)
        report["total_duration_sec"] = round(time.time() - t0, 2)
        rep_p = v3_out_dir / "raporlar" / f"v3_patched_blocked_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
        atomic_write_json(rep_p, report)
        report["report_path"] = str(rep_p)
        return report

    # Onaylı paragraf WAV'larını ve gerçek sürelerini yalnızca downstream aşamalar isteniyorsa oku
    paragraph_durations: Dict[str, float] = {}
    chapter_vocals_map: Dict[str, List[np.ndarray]] = {}
    amb_segments: List[Dict[str, Any]] = []
    if run_all or any(s in stages_req for s in ("ambience", "mastering", "assembly", "subtitles")):
        for par in paragraphs:
            p_wav = v3_out_dir / "paragraflar" / f"{par['p_id']}.wav"
            arr, sr_p = _read_wav_f32(p_wav)
            dur = len(arr) / float(sr_p)
            paragraph_durations[par["p_id"]] = dur
            chapter_vocals_map.setdefault(par["section"], []).append(arr)
            amb_segments.append({"ambience_key": par["ambience_key"], "duration_sec": max(1.0, dur)})

    # =========================================================================
    # AŞAMA 2: AUDIOLDM AMBİYANS ÜRETİMİ VE ZAMAN ÇİZELGESİ
    # =========================================================================
    amb_wav_path = v3_out_dir / "ambiyans" / "timeline_ambiyans.wav"
    if run_all or "ambience" in stages_req:
        t_s = time.time()
        amb_engine = AudioLDMAmbienceEngine(
            output_dir=v3_out_dir / "ambiyans",
            sample_rate=xtts_cfg.sample_rate,
            model_name=cfg["ambience"]["model_name"],
            licensed_assets_dir=studio_dir / "sfx_ambiyans_audioldm",
            generator_fn=backends.get("audioldm_fn"),
        )
        try:
            amb_rec = manifest.register_or_validate_cache(
                artifact_id="ambience_timeline",
                stage="ambience_v3",
                output_path=str(amb_wav_path),
                source_hash=script_sha,
                model_name=cfg["ambience"]["model_name"],
                model_version="v3.0.1",
                parameters={"sections": sorted(set(p["ambience_key"] for p in paragraphs))},
            )
            if amb_rec.get_status() != ArtifactStatus.APPROVED:
                safe_begin_generation(manifest, "ambience_timeline")
                amb_res = amb_engine.render_timeline_ambience(
                    segments=amb_segments,
                    out_wav=amb_wav_path,
                    allow_synthetic_test_fallback=bool(backends.get("allow_synthetic_ambience", False)),
                    device=device,
                )
                manifest.set_status("ambience_timeline", ArtifactStatus.VALIDATING)
                manifest.set_status(
                    "ambience_timeline", ArtifactStatus.APPROVED,
                    compute_output_hash=True, quality_scores=amb_res,
                )
            else:
                amb_res = {"output_path": str(amb_wav_path), "reused_from_cache": True}
        finally:
            amb_engine.unload()
        report["stages"]["ambience"] = {"duration_sec": round(time.time() - t_s, 2), **amb_res}

    # =========================================================================
    # AŞAMA 3: PROFESYONEL SES MASTERING (LUFS / TRUE PEAK / OUTRO)
    # =========================================================================
    master_wav_path = v3_out_dir / "final" / "master_audio.wav"
    master_m4a_path = v3_out_dir / "final" / "master_audio.m4a"
    outro_start_sec = 0.0
    if run_all or "mastering" in stages_req:
        t_s = time.time()
        m_rec = manifest.register_or_validate_cache(
            artifact_id="master_audio",
            stage="mastering_v3",
            output_path=str(master_wav_path),
            source_hash=script_sha,
            model_name="mastering_v3",
            model_version="v3.0.1",
            parameters={"outro_sec": outro_sec, "outro_fade_sec": outro_fade_sec},
        )
        chapter_vocals: List[Tuple[str, np.ndarray]] = []
        for sec_name, arrs in chapter_vocals_map.items():
            chapter_vocals.append((sec_name, np.concatenate(arrs)))
        outro_start_sec = sum(len(a) for _, a in chapter_vocals) / float(xtts_cfg.sample_rate)

        if m_rec.get_status() != ArtifactStatus.APPROVED or not master_m4a_path.exists():
            safe_begin_generation(manifest, "master_audio")
            amb_bed = None
            if amb_wav_path.exists():
                amb_bed, sr_a = _read_wav_f32(amb_wav_path)
                if sr_a != xtts_cfg.sample_rate:
                    amb_bed = sps.resample(amb_bed, int(len(amb_bed) * xtts_cfg.sample_rate / sr_a)).astype(np.float32)

            m_eng = MasteringEngine(sr=xtts_cfg.sample_rate, thresholds=audio_th)
            mastered, m_rep = m_eng.mix_and_master(
                chapter_vocals=chapter_vocals,
                ambience_bed=amb_bed,
                outro_sec=outro_sec,
                outro_fade_sec=outro_fade_sec,
            )
            tmp_m = local_work / "master_audio.wav"
            _write_wav_pcm16(tmp_m, mastered, xtts_cfg.sample_rate)
            out_sha = atomic_copy_file(tmp_m, master_wav_path)
            subprocess.check_call(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(master_wav_path), "-c:a", "aac", "-b:a", "192k", str(master_m4a_path),
                ]
            )
            manifest.set_status("master_audio", ArtifactStatus.VALIDATING)
            m_dict = m_rep.to_dict()
            m_dict["outro_start_sec"] = round(outro_start_sec, 3)
            if m_rep.passed:
                manifest.set_status("master_audio", ArtifactStatus.APPROVED, compute_output_hash=True, quality_scores=m_dict)
            else:
                manifest.set_status("master_audio", ArtifactStatus.MANUAL_REVIEW, quality_scores=m_dict, error_message="; ".join(m_rep.issues))
            report["stages"]["mastering"] = {"duration_sec": round(time.time() - t_s, 2), **m_dict}
        else:
            report["stages"]["mastering"] = {"duration_sec": round(time.time() - t_s, 2), "reused_from_cache": True, "outro_start_sec": round(outro_start_sec, 3)}

    # FAIL-CLOSED KAPI: Mastering başarısızsa görsel/video/altyazı/kurguya geçilmez
    m_gate_rec = manifest.get("master_audio")
    if (run_all or "mastering" in stages_req) and (
        m_gate_rec is None or m_gate_rec.get_status() != ArtifactStatus.APPROVED
    ) and (run_all or any(s in stages_req for s in ("visuals", "video", "subtitles", "assembly"))):
        report["stages"]["mastering_gate"] = {
            "blocked": True,
            "reason": "master_audio APPROVED olmadan görsel/video/altyazı/kurgu aşamasına geçilemez.",
            "master_status": m_gate_rec.status if m_gate_rec else "MISSING",
        }
        report["protection_sha256"] = guard.verify_after()
        report["resources_end"] = snapshot_system_resources(local_work)
        report["total_duration_sec"] = round(time.time() - t0, 2)
        rep_p = v3_out_dir / "raporlar" / f"v3_patched_blocked_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
        atomic_write_json(rep_p, report)
        report["report_path"] = str(rep_p)
        return report

    # =========================================================================
    # AŞAMA 4: SDXL / IMAGEN GÖRSEL ÜRETİMİ VE DEVAMLILIK DENETİMİ
    # =========================================================================
    continuity_mgr = VisualContinuityManager()
    scene_specs = _build_scene_specs(paragraphs, continuity_mgr)
    approved_scene_images: Dict[int, Path] = {}

    if run_all or "visuals" in stages_req:
        t_s = time.time()
        img_gen = ImageGenerator(
            config=SDXLConfig(),
            thresholds=ImageThresholds(),
            generator_fn=backends.get("image_gen_fn"),
        )
        img_statuses: Dict[str, str] = {}
        try:
            for spec in scene_specs:
                aid = f"visual_sahne_{spec.scene_index:02d}"
                img_path = v3_out_dir / "gorseller" / spec.image_filename
                rec = manifest.register_or_validate_cache(
                    artifact_id=aid,
                    stage="visuals_v3",
                    output_path=str(img_path),
                    source_hash=sha256_text(spec.sdxl_prompt),
                    model_name=cfg["visuals"]["primary_model"],
                    model_version="v3.0.1",
                    parameters={"w": 1920, "h": 1080},
                )
                if rec.get_status() == ArtifactStatus.APPROVED:
                    approved_scene_images[spec.scene_index] = img_path
                    img_statuses[aid] = "APPROVED_CACHED"
                    continue

                safe_begin_generation(manifest, aid)
                if "image_gen_fn" not in backends and img_gen._pipe is None:
                    img_gen.load_sdxl(device="cuda")
                ok, val_res, _ = img_gen.ensure_valid_scene_image(
                    target_path=img_path,
                    prompt=spec.sdxl_prompt,
                    local_tmp_dir=local_work,
                    max_retries=2,
                )
                manifest.set_status(aid, ArtifactStatus.VALIDATING)
                if ok and img_path.exists():
                    manifest.set_status(aid, ArtifactStatus.APPROVED, compute_output_hash=True, quality_scores=val_res.to_dict())
                    approved_scene_images[spec.scene_index] = img_path
                    img_statuses[aid] = "APPROVED"
                else:
                    manifest.set_status(aid, ArtifactStatus.MANUAL_REVIEW, quality_scores=val_res.to_dict(), error_message="; ".join(val_res.issues))
                    img_statuses[aid] = "MANUAL_REVIEW"
        finally:
            img_gen.unload()
            release_gpu_vram()

        cont_rep = continuity_mgr.evaluate_cross_scene_continuity(
            [approved_scene_images[k] for k in sorted(approved_scene_images.keys())]
        )
        report["stages"]["visuals"] = {
            "duration_sec": round(time.time() - t_s, 2),
            "scene_count": len(scene_specs),
            "approved_images": len(approved_scene_images),
            "statuses": img_statuses,
            "continuity": cont_rep,
        }

    # =========================================================================
    # AŞAMA 5: SVD-XT IMAGE-TO-VIDEO + CLIP + OPENCV + PING-PONG/LOOP
    # =========================================================================
    approved_video_clips: Dict[int, Path] = {}
    if (run_all or "visuals" in stages_req) and len(approved_scene_images) != len(scene_specs) and (run_all or any(s in stages_req for s in ("video", "assembly"))):
        report["stages"]["visuals_gate"] = {
            "blocked": True,
            "reason": "Tüm sahne görselleri APPROVED olmadan SVD-XT video ve final kurguya geçilemez.",
            "approved_images": len(approved_scene_images),
            "required_images": len(scene_specs),
        }
        report["protection_sha256"] = guard.verify_after()
        report["resources_end"] = snapshot_system_resources(local_work)
        report["total_duration_sec"] = round(time.time() - t0, 2)
        rep_p = v3_out_dir / "raporlar" / f"v3_patched_blocked_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
        atomic_write_json(rep_p, report)
        report["report_path"] = str(rep_p)
        return report

    if run_all or "video" in stages_req:
        t_s = time.time()
        clip_val = CLIPValidator(
            thresholds=VideoThresholds(),
            score_fn=backends.get("clip_score_fn"),
        )
        svd_eng = SVDEngine(
            config=SVDConfig(),
            thresholds=VideoThresholds(),
            clip_validator=clip_val,
            frame_generator_fn=backends.get("svd_frame_fn"),
        )

        vid_statuses: Dict[str, str] = {}
        try:
            for spec in scene_specs:
                aid = f"video_clip_sahne_{spec.scene_index:02d}"
                img_path = approved_scene_images.get(spec.scene_index) or (v3_out_dir / "gorseller" / spec.image_filename)
                if not img_path.exists():
                    vid_statuses[aid] = "BLOCKED_MISSING_IMAGE"
                    continue
                clip_path = v3_out_dir / "video_klipler_svd" / f"klip_{spec.scene_index:02d}.mp4"
                rec = manifest.register_or_validate_cache(
                    artifact_id=aid,
                    stage="video_i2v_v3",
                    output_path=str(clip_path),
                    source_hash=sha256_file(img_path),
                    model_name=cfg["video_i2v"]["model_name"],
                    model_version="v3.0.1",
                    parameters={"profile": spec.motion_profile_name},
                )
                if rec.get_status() == ArtifactStatus.APPROVED:
                    approved_video_clips[spec.scene_index] = clip_path
                    vid_statuses[aid] = "APPROVED_CACHED"
                    continue

                safe_begin_generation(manifest, aid)
                if "svd_frame_fn" not in backends and svd_eng._pipe is None:
                    svd_eng.load_models(device="cuda")
                c_res = svd_eng.generate_and_validate_clip(
                    scene=spec,
                    source_image_path=img_path,
                    target_clip_path=clip_path,
                    local_tmp_dir=local_work,
                )
                manifest.set_status(aid, ArtifactStatus.VALIDATING)
                if c_res.status == ArtifactStatus.APPROVED and clip_path.exists():
                    manifest.set_status(aid, ArtifactStatus.APPROVED, compute_output_hash=True, quality_scores={"attempts": c_res.attempts})
                    approved_video_clips[spec.scene_index] = clip_path
                    vid_statuses[aid] = "APPROVED"
                else:
                    manifest.set_status(aid, ArtifactStatus.MANUAL_REVIEW, quality_scores={"attempts": c_res.attempts, "error_summary": getattr(c_res, "error_summary", ""), "history": getattr(c_res, "history", [])}, error_message=getattr(c_res, "error_summary", "") or "SVD/CLIP/OpenCV kalite eşiği geçilemedi")
                    vid_statuses[aid] = "MANUAL_REVIEW"
        finally:
            svd_eng.unload_models()
            release_gpu_vram()

        report["stages"]["video_i2v"] = {
            "duration_sec": round(time.time() - t_s, 2),
            "scene_count": len(scene_specs),
            "approved_clips": len(approved_video_clips),
            "statuses": vid_statuses,
        }

    # =========================================================================
    # AŞAMA 6: WHISPER TURBO SRT ALTYAZI VE DOĞRULAMA
    # =========================================================================
    srt_path = v3_out_dir / "final" / "YOUTUBE_FINAL_TR.srt"
    if (run_all or "subtitles" in stages_req) and master_wav_path.exists():
        t_s = time.time()
        all_sentences = [s for par in paragraphs for s in par["norm_sentences"]]
        if "srt_words_fn" in backends:
            words = backends["srt_words_fn"](master_wav_path, outro_start_sec)
        else:
            words = transcribe_audio_16k(
                media_path=master_wav_path,
                local_tmp_dir=local_work,
                outro_start_sec=outro_start_sec,
                device="cuda",
                model_name=cfg["whisper"]["model_name"],
            )
            release_gpu_vram()

        aligned = align_sentences_to_words(all_sentences, words, outro_start_sec=outro_start_sec, thresholds=SRTThresholds())
        cues = generate_srt_cues(aligned, outro_start_sec=outro_start_sec, thresholds=SRTThresholds())
        write_srt_file(cues, srt_path)
        srt_val = validate_srt_file(srt_path, expected_sentences=all_sentences, outro_start_sec=outro_start_sec, thresholds=SRTThresholds())
        report["stages"]["subtitles"] = {
            "duration_sec": round(time.time() - t_s, 2),
            "srt_path": str(srt_path),
            "cue_count": len(cues),
            "passed": srt_val.passed,
            "issues": srt_val.issues,
            "sha256": sha256_file(srt_path),
        }

    # =========================================================================
    # AŞAMA 7: FFMPEG 1080p/24fps KURGU, SAF SİYAH OUTRO VE NİHAİ DOĞRULAMA
    # =========================================================================
    final_mp4_path = v3_out_dir / "final" / "YOUTUBE_FINAL_1080P.mp4"
    if run_all or "assembly" in stages_req:
        t_s = time.time()
        m_rec = manifest.get("master_audio")
        all_clips_ok = len(approved_video_clips) == len(scene_specs) and len(scene_specs) > 0
        if (m_rec is None) or (m_rec.get_status() != ArtifactStatus.APPROVED) or (not all_clips_ok):
            report["stages"]["assembly"] = {
                "blocked": True,
                "reason": "FAIL-CLOSED: Mastering veya tüm sahne video klipleri APPROVED olmadan final MP4 üretilemez.",
                "master_status": m_rec.status if m_rec else "MISSING",
                "approved_clips": len(approved_video_clips),
                "required_clips": len(scene_specs),
            }
        else:
            asm = FFmpegAssemblyEngine(local_work_dir=local_work, ffmpeg_threads=int(cfg["assembly"]["ffmpeg_threads"]))
            extended_files: List[Path] = []
            chapters: List[Tuple[str, float]] = []
            cursor = 0.0
            for spec in scene_specs:
                s_idx = int(spec.start_paragraph[1:])
                e_idx = int(spec.end_paragraph[1:])
                s_dur = sum(
                    paragraph_durations.get(f"paragraf_{i:03d}", 0.0)
                    for i in range(s_idx, e_idx + 1)
                )
                s_dur = max(2.0, s_dur)
                chapters.append((f"Sahne {spec.scene_index:02d}", cursor))
                cursor += s_dur
                ext_p = v3_out_dir / "sahne_uzatilmis" / f"sahne_{spec.scene_index:02d}_ext.mp4"
                asm.render_extended_scene(approved_video_clips[spec.scene_index], s_dur, ext_p)
                extended_files.append(ext_p)

            outro_mp4 = v3_out_dir / "sahne_uzatilmis" / "outro_siyah.mp4"
            asm.render_black_outro(outro_sec, outro_mp4)
            extended_files.append(outro_mp4)

            exp_res = export_verified_final_video(
                engine=asm,
                scene_files=extended_files,
                master_audio_path=master_m4a_path,
                final_output_path=final_mp4_path,
                outro_start_sec=outro_start_sec,
                thresholds=AssemblyThresholds(),
            )
            chap_txt = format_youtube_chapters("Unutulmuş Zamanlar Arşivi (V3)", chapters)
            (v3_out_dir / "final" / "YOUTUBE_BOLUMLER.txt").write_text(chap_txt, encoding="utf-8")
            report["stages"]["assembly"] = {
                "duration_sec": round(time.time() - t_s, 2),
                **exp_res.to_dict(),
            }

    report["protection_sha256"] = guard.verify_after()
    report["resources_end"] = snapshot_system_resources(local_work)
    report["total_duration_sec"] = round(time.time() - t0, 2)
    rep_path = v3_out_dir / "raporlar" / f"v3_patched_uretim_raporu_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    atomic_write_json(rep_path, report)
    report["report_path"] = str(rep_path)
    assert report["protection_sha256"]["all_protected_ok"], "KRİTİK: Korunan eski dosyalarda SHA-256 uyuşmazlığı!"
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="UYKU_HIKAYESI_V3_PATCHED — Bütünleşik Tek Ana Çalıştırıcı")
    parser.add_argument("--zip-path", default=None, help="UYKU_HIKAYESI_V3_PATCHED.zip yolu")
    parser.add_argument("--extract-root", default=str(DEFAULT_V3_LOCAL_PKG_ROOT))
    parser.add_argument("--config", default=None)
    parser.add_argument("--script", default=None)
    parser.add_argument("--ref-audio", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--local-work", default="/content/uyku_v3_patched_work")
    parser.add_argument("--stages", default="all")
    parser.add_argument("--max-paragraphs", type=int, default=0)
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    pkg_root = ensure_v3_package_extracted(
        zip_path=Path(args.zip_path) if args.zip_path else None,
        extract_root=Path(args.extract_root),
    )
    from uyku_v3.config.settings import enable_line_buffering
    enable_line_buffering()

    cfg_path = Path(args.config) if args.config else (pkg_root / "config_v3_production.json")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    script_path = Path(args.script or cfg["paths"]["default_script"])
    ref_audio = Path(args.ref_audio or cfg["paths"]["default_ref_audio"])
    out_dir = Path(args.output_dir or (DEFAULT_V3_PATCHED_DRIVE_ROOT / "uretim_ciktilari"))

    try:
        rep = run_v3_pipeline(
            cfg=cfg,
            script_path=script_path,
            ref_audio=ref_audio,
            v3_out_dir=out_dir,
            local_work=Path(args.local_work),
            stages_req={s.strip().lower() for s in args.stages.split(",")},
            max_paragraphs=args.max_paragraphs,
            stop_after=args.stop_after,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        err_dir = out_dir / "loglar"
        err_dir.mkdir(parents=True, exist_ok=True)
        (err_dir / "last_traceback.txt").write_text(tb, encoding="utf-8")
        print(f"[CRITICAL][PIPELINE] Beklenmeyen hata: {exc}\n{tb}", file=sys.stderr, flush=True)
        return 1

    st = rep.get("stages", {})
    if (
        st.get("downstream_gate", {}).get("blocked")
        or st.get("mastering_gate", {}).get("blocked")
        or st.get("visuals_gate", {}).get("blocked")
        or st.get("assembly", {}).get("blocked")
    ):
        return 2
    if "subtitles" in st and not st["subtitles"].get("passed", False):
        return 2
    if "assembly" in st and st["assembly"].get("status") != "APPROVED":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
