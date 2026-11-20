"""
SVD-XT Hareketli Video Üretim Motoru (Bölüm 9).
P0-03 DÜZELTMESİ:
- 3 denemenin tamamı kalite kontrolünden geçemezse en yüksek puanlı başarısız videoyu
  ONAYLI klip olarak ASLA kaydetmez!
- Başarısız klip MANUAL_REVIEW durumuna alınır, inceleme kopyası karantina klasörüne
  alınır ve final montaja girmesi kesin olarak engellenir.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from ..config.model_config import SVDConfig
from ..config.quality_thresholds import VideoThresholds
from ..core.state_manager import ArtifactStatus
from ..visuals.scene_manager import SceneSpec
from .clip_validator import CLIPValidationResult, CLIPValidator
from .loop_engine import LoopResult, build_seamless_loop
from .motion_validator import MotionValidationResult, analyze_video_frames
from .video_validator import validate_video_file


@dataclass
class ClipGenerationResult:
    scene_index: int
    status: ArtifactStatus
    output_path: str
    attempts: int
    clip_report: Optional[CLIPValidationResult]
    motion_report: Optional[MotionValidationResult]
    loop_report: Optional[LoopResult]
    history: List[Dict[str, Any]] = field(default_factory=list)
    error_summary: str = ""


class SVDEngine:
    def __init__(
        self,
        config: Optional[SVDConfig] = None,
        thresholds: Optional[VideoThresholds] = None,
        clip_validator: Optional[CLIPValidator] = None,
        frame_generator_fn: Optional[Callable[[Image.Image, int, float, int], List[Image.Image]]] = None,
    ) -> None:
        self.config = config or SVDConfig()
        self.thresholds = thresholds or VideoThresholds()
        self.clip_validator = clip_validator or CLIPValidator(self.thresholds)
        self._frame_generator_fn = frame_generator_fn
        self._pipe: Any = None

    def load_models(self, device: str = "cuda") -> None:
        if self._frame_generator_fn is not None:
            return
        if device != "cuda":
            raise RuntimeError("SVD-XT video üretimi için T4 GPU (CUDA) gereklidir; CPU üzerinde sessiz çalıştırma engellendi.")
        import torch  # type: ignore
        from diffusers import StableVideoDiffusionPipeline  # type: ignore

        self._pipe = StableVideoDiffusionPipeline.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            variant=self.config.variant,
        )
        if self.config.enable_cpu_offload:
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe.to(device)
        self.clip_validator.load(device=device)

    def unload_models(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        if getattr(self.clip_validator, "_model", None) is not None:
            self.clip_validator._model = None
            self.clip_validator._proc = None
        gc.collect()
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def generate_and_validate_clip(
        self,
        scene: SceneSpec,
        source_image_path: Path,
        target_clip_path: Path,
        local_tmp_dir: Path,
    ) -> ClipGenerationResult:
        """
        Her sahne için en fazla `max_retries` (varsayılan 3) deneme yapar.
        Yalnızca hem CLIP hem OpenCV hareket hem döngü hem FFprobe/decode doğrulamasını
        geçen klipler `target_clip_path`'e yazılır ve APPROVED olur.
        Aksi halde `target_clip_path` OLUŞTURULMAZ ve durum MANUAL_REVIEW olur!
        """
        source_img = Image.open(source_image_path).convert("RGB").resize(
            (self.config.width, self.config.height), Image.Resampling.LANCZOS
        )
        prof = scene.motion_profile
        base_mb = prof.motion_bucket_id

        retry_profiles = [
            (base_mb, prof.noise_aug_strength, 42 + scene.scene_index * 7),
            (max(24, base_mb - 12), max(0.010, prof.noise_aug_strength - 0.006), 108 + scene.scene_index * 13),
            (max(20, base_mb - 18), 0.010, 256 + scene.scene_index * 19),
        ][: self.config.max_retries]

        history: List[Dict[str, Any]] = []
        last_clip_res: Optional[CLIPValidationResult] = None
        last_motion_res: Optional[MotionValidationResult] = None

        for attempt, (mb_id, noise_aug, seed) in enumerate(retry_profiles, 1):
            if self._frame_generator_fn is not None:
                frames = self._frame_generator_fn(source_img, mb_id, noise_aug, seed)
            else:
                if self._pipe is None:
                    raise RuntimeError("SVD-XT pipeline yüklenmedi.")
                import torch  # type: ignore
                gen = torch.Generator(device="cpu").manual_seed(seed)
                frames = self._pipe(
                    source_img,
                    width=self.config.width,
                    height=self.config.height,
                    num_frames=self.config.num_frames,
                    decode_chunk_size=self.config.decode_chunk_size,
                    motion_bucket_id=mb_id,
                    noise_aug_strength=noise_aug,
                    fps=self.config.fps,
                    num_inference_steps=self.config.num_inference_steps,
                    generator=gen,
                ).frames[0]

            clip_res = self.clip_validator.validate_frames(frames, scene.clip_query)
            motion_res = analyze_video_frames(frames, source_img, self.thresholds)
            last_clip_res, last_motion_res = clip_res, motion_res

            attempt_ok = clip_res.passed and motion_res.passed
            history.append({
                "attempt": attempt,
                "motion_bucket_id": mb_id,
                "noise_aug_strength": noise_aug,
                "seed": seed,
                "clip_passed": clip_res.passed,
                "motion_passed": motion_res.passed,
                "clip_issues": clip_res.issues,
                "motion_issues": motion_res.issues,
            })

            if attempt_ok:
                loop_res = build_seamless_loop(
                    frames=frames,
                    output_mp4=target_clip_path,
                    local_tmp_dir=local_tmp_dir,
                    loop_mode=prof.recommended_loop,
                    fps_source=self.config.fps,
                    fps_target=24,
                    max_seam_delta=self.thresholds.max_loop_seam_delta,
                )
                vid_val = validate_video_file(target_clip_path, expected_width=1920, expected_height=1080, expected_fps=24.0)
                if loop_res.passed and vid_val.passed:
                    return ClipGenerationResult(
                        scene_index=scene.scene_index,
                        status=ArtifactStatus.APPROVED,
                        output_path=str(target_clip_path),
                        attempts=attempt,
                        clip_report=clip_res,
                        motion_report=motion_res,
                        loop_report=loop_res,
                        history=history,
                    )
                else:
                    # Döngü veya decode başarısızsa onaylı hedefe yazılan dosyayı kaldır
                    target_clip_path.unlink(missing_ok=True)
                    history[-1]["loop_issues"] = loop_res.issues + vid_val.issues

            gc.collect()

        # KRİTİK KURAL (Bölüm 9): Tüm denemeler başarısız -> ASLA APPROVED veya hedef MP4 KAYDETME!
        target_clip_path.unlink(missing_ok=True)
        all_issues = (last_clip_res.issues if last_clip_res else []) + (last_motion_res.issues if last_motion_res else [])
        return ClipGenerationResult(
            scene_index=scene.scene_index,
            status=ArtifactStatus.MANUAL_REVIEW,
            output_path="",
            attempts=len(retry_profiles),
            clip_report=last_clip_res,
            motion_report=last_motion_res,
            loop_report=None,
            history=history,
            error_summary="; ".join(all_issues) or "Maksimum deneme sayısında onay alınamadı.",
        )
