"""
Final Dışa Aktarım Koordinatörü (Bölüm 11).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from ..config.quality_thresholds import AssemblyThresholds
from ..core.state_manager import ArtifactStatus
from .ffmpeg_engine import FFmpegAssemblyEngine
from .sync_validator import SyncValidationResult, validate_final_assembly


@dataclass
class FinalExportResult:
    status: ArtifactStatus
    output_path: str
    validation: SyncValidationResult

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": self.status.value,
            "output_path": self.output_path,
            "validation": self.validation.to_dict(),
        }


def export_verified_final_video(
    engine: FFmpegAssemblyEngine,
    scene_files: List[Path],
    master_audio_path: Path,
    final_output_path: Path,
    outro_start_sec: Optional[float] = None,
    thresholds: Optional[AssemblyThresholds] = None,
) -> FinalExportResult:
    engine.assemble_final_mp4(scene_files, master_audio_path, final_output_path)
    val = validate_final_assembly(
        final_mp4_path=final_output_path,
        master_audio_path=master_audio_path,
        outro_start_sec=outro_start_sec,
        thresholds=thresholds,
    )
    status = ArtifactStatus.APPROVED if val.passed else ArtifactStatus.REJECTED
    return FinalExportResult(status=status, output_path=str(final_output_path), validation=val)
