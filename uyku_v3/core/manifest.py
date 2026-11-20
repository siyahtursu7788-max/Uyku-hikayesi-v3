"""
Merkezi JSON Manifest Sistemi (Bölüm 4 & 13).
Her çıktı için proje kimliği, kaynak hash'i, üretim tarihi, model, model sürümü,
parametreler, deneme sayısı, kalite puanları, onay durumu, hata mesajları ve son işlem tarihini saklar.
Kaynak metin, kaynak görsel veya parametreler değiştiğinde eski onayları otomatik geçersiz kılar.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .file_manager import atomic_write_json, safe_read_json, sha256_dict, sha256_file
from .state_manager import ArtifactStatus, StateManager


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ArtifactRecord:
    artifact_id: str
    stage: str
    project_id: str
    output_path: str
    source_hash: str
    params_hash: str
    model_name: str
    model_version: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    attempts: int = 0
    quality_scores: Dict[str, Any] = field(default_factory=dict)
    status: str = ArtifactStatus.PENDING.value
    output_hash: str = ""
    error_messages: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    def get_status(self) -> ArtifactStatus:
        return ArtifactStatus(self.status)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactRecord":
        return cls(**data)


class ManifestManager:
    """
    Tüm üretim aşamalarının tek gerçek kaynağı (Single Source of Truth).
    Atomik yazım sayesinde Colab kesintilerinde manifest dosyası asla bozulmaz.
    """

    SCHEMA_VERSION = "2.0.0"

    def __init__(self, manifest_path: Union[str, Path], project_id: str) -> None:
        self.manifest_path = Path(manifest_path)
        self.project_id = project_id
        self.data: Dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "project_id": project_id,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "artifacts": {},
            "retry_queue": [],
            "manual_review_queue": [],
        }
        self.load()

    def load(self) -> None:
        if not self.manifest_path.exists():
            return
        loaded = safe_read_json(self.manifest_path, default=None)
        if isinstance(loaded, dict) and "artifacts" in loaded:
            self.data = loaded
        else:
            # Bozuk JSON varsa yedeğini karantinaya al
            corrupt_bak = self.manifest_path.with_suffix(".corrupt.bak")
            try:
                os.replace(self.manifest_path, corrupt_bak)
            except OSError:
                pass

    def save(self) -> str:
        self.data["updated_at"] = _utc_now()
        self._rebuild_queues()
        return atomic_write_json(self.manifest_path, self.data)

    def _rebuild_queues(self) -> None:
        retry_q: List[str] = []
        manual_q: List[str] = []
        for aid, raw in self.data["artifacts"].items():
            st = raw.get("status")
            if st in (ArtifactStatus.REJECTED.value, ArtifactStatus.INTERRUPTED.value, ArtifactStatus.FAILED.value):
                retry_q.append(aid)
            elif st == ArtifactStatus.MANUAL_REVIEW.value:
                manual_q.append(aid)
        self.data["retry_queue"] = sorted(retry_q)
        self.data["manual_review_queue"] = sorted(manual_q)

    def get(self, artifact_id: str) -> Optional[ArtifactRecord]:
        raw = self.data["artifacts"].get(artifact_id)
        return ArtifactRecord.from_dict(raw) if raw else None

    def upsert(self, record: ArtifactRecord) -> None:
        record.updated_at = _utc_now()
        self.data["artifacts"][record.artifact_id] = record.to_dict()
        self.save()

    def register_or_validate_cache(
        self,
        artifact_id: str,
        stage: str,
        output_path: str,
        source_hash: str,
        model_name: str,
        model_version: str,
        parameters: Dict[str, Any],
        dependencies: Optional[List[str]] = None,
    ) -> ArtifactRecord:
        """
        Mevcut kaydı denetler:
        - Kaynak hash'i veya üretim parametreleri değiştiyse eski APPROVED kaydını PENDING'e düşürür.
        - Dosya diskte yoksa veya dosya hash'i manifestteki output_hash ile uyuşmuyorsa geçersiz kılar.
        - Bağımlı olduğu üst çıktılardan biri APPROVED değilse geçersiz kılar.
        """
        params_hash = sha256_dict(parameters)
        deps = dependencies or []
        existing = self.get(artifact_id)

        if existing is None:
            rec = ArtifactRecord(
                artifact_id=artifact_id,
                stage=stage,
                project_id=self.project_id,
                output_path=output_path,
                source_hash=source_hash,
                params_hash=params_hash,
                model_name=model_name,
                model_version=model_version,
                parameters=parameters,
                status=ArtifactStatus.PENDING.value,
                dependencies=deps,
            )
            self.upsert(rec)
            return rec

        invalidation_reasons: List[str] = []
        if existing.source_hash != source_hash:
            invalidation_reasons.append("Kaynak dosya/metin hash'i değişti.")
        if existing.params_hash != params_hash or existing.model_name != model_name or existing.model_version != model_version:
            invalidation_reasons.append("Model veya üretim parametreleri değişti.")

        for dep_id in deps:
            dep_rec = self.get(dep_id)
            if dep_rec is None or dep_rec.get_status() != ArtifactStatus.APPROVED:
                invalidation_reasons.append(f"Bağımlı çıktı onaylı değil: {dep_id}")

        if existing.get_status() == ArtifactStatus.APPROVED:
            out_p = Path(output_path)
            if not out_p.exists() or out_p.stat().st_size == 0:
                invalidation_reasons.append("Onaylı çıktı dosyası diskte bulunamadı veya boş.")
            elif existing.output_hash:
                actual_hash = sha256_file(out_p)
                if actual_hash != existing.output_hash:
                    invalidation_reasons.append("Çıktı dosyasının SHA-256 hash'i manifest ile uyuşmuyor (bozulmuş/değiştirilmiş).")

        if invalidation_reasons:
            existing.status = ArtifactStatus.PENDING.value
            existing.source_hash = source_hash
            existing.params_hash = params_hash
            existing.model_name = model_name
            existing.model_version = model_version
            existing.parameters = parameters
            existing.dependencies = deps
            existing.output_hash = ""
            existing.attempts = 0
            existing.error_messages.extend(invalidation_reasons)
            self.upsert(existing)

        return existing

    def set_status(
        self,
        artifact_id: str,
        target_status: ArtifactStatus,
        quality_scores: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        increment_attempt: bool = False,
        compute_output_hash: bool = False,
    ) -> ArtifactRecord:
        rec = self.get(artifact_id)
        if rec is None:
            raise KeyError(f"Manifestte kayıt bulunamadı: {artifact_id}")

        current = rec.get_status()
        StateManager.transition(current, target_status, reason=error_message or "")
        rec.status = target_status.value
        if increment_attempt:
            rec.attempts += 1
        if quality_scores is not None:
            rec.quality_scores = quality_scores
        if error_message:
            rec.error_messages.append(error_message)
        if compute_output_hash and Path(rec.output_path).exists():
            rec.output_hash = sha256_file(rec.output_path)
        self.upsert(rec)
        return rec

    def all_approved(self, stage: Optional[str] = None) -> bool:
        records = [
            ArtifactRecord.from_dict(r)
            for r in self.data["artifacts"].values()
            if stage is None or r.get("stage") == stage
        ]
        return bool(records) and all(r.get_status() == ArtifactStatus.APPROVED for r in records)

    def get_by_stage(self, stage: str) -> List[ArtifactRecord]:
        return [
            ArtifactRecord.from_dict(r)
            for r in self.data["artifacts"].values()
            if r.get("stage") == stage
        ]

    def summary_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {s.value: 0 for s in ArtifactStatus}
        for r in self.data["artifacts"].values():
            st = r.get("status", ArtifactStatus.PENDING.value)
            counts[st] = counts.get(st, 0) + 1
        return counts
