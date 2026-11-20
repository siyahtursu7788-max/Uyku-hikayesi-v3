"""
Yapılandırılmış JSONL loglama + mobil Colab uyumlu canlı konsol çıktısı.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class PipelineLogger:
    def __init__(self, log_dir: Path, project_id: str, echo: bool = True) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.project_id = project_id
        self.echo = echo
        self.log_file = self.log_dir / f"{project_id}_pipeline.jsonl"
        self.error_file = self.log_dir / f"{project_id}_errors.jsonl"

    def log(
        self,
        level: str,
        stage: str,
        message: str,
        artifact_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "project_id": self.project_id,
            "level": level.upper(),
            "stage": stage,
            "artifact_id": artifact_id,
            "message": message,
            "extra": extra or {},
        }
        line = json.dumps(entry, ensure_ascii=False)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
        if level.upper() in ("ERROR", "CRITICAL", "WARNING"):
            with open(self.error_file, "a", encoding="utf-8") as ef:
                ef.write(line + "\n")
                ef.flush()
        if self.echo:
            prefix = f"[{level.upper()}][{stage}]"
            if artifact_id:
                prefix += f"[{artifact_id}]"
            print(f"{prefix} {message}", flush=True)

    def info(self, stage: str, message: str, artifact_id: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        self.log("INFO", stage, message, artifact_id, extra)

    def warning(self, stage: str, message: str, artifact_id: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        self.log("WARNING", stage, message, artifact_id, extra)

    def error(self, stage: str, message: str, artifact_id: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        self.log("ERROR", stage, message, artifact_id, extra)
