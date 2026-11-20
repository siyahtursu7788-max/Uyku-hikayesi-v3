"""
Birleşik Durum Makinesi (Bölüm 4).
Yalnızca APPROVED çıktılar bir sonraki aşamaya aktarılabilir.
Başarısız bir çıktı hiçbir koşulda otomatik olarak APPROVED kabul edilemez.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Set


class ArtifactStatus(str, Enum):
    PENDING = "PENDING"
    GENERATING = "GENERATING"
    VALIDATING = "VALIDATING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


class InvalidStateTransitionError(ValueError):
    """Geçersiz durum geçişini engelleyen istisna."""


ALLOWED_TRANSITIONS: Dict[ArtifactStatus, Set[ArtifactStatus]] = {
    ArtifactStatus.PENDING: {
        ArtifactStatus.GENERATING,
        ArtifactStatus.VALIDATING,
        ArtifactStatus.FAILED,
        ArtifactStatus.INTERRUPTED,
    },
    ArtifactStatus.GENERATING: {
        ArtifactStatus.VALIDATING,
        ArtifactStatus.FAILED,
        ArtifactStatus.INTERRUPTED,
    },
    ArtifactStatus.VALIDATING: {
        ArtifactStatus.APPROVED,
        ArtifactStatus.REJECTED,
        ArtifactStatus.MANUAL_REVIEW,
        ArtifactStatus.FAILED,
        ArtifactStatus.INTERRUPTED,
    },
    ArtifactStatus.REJECTED: {
        ArtifactStatus.PENDING,
        ArtifactStatus.GENERATING,
        ArtifactStatus.MANUAL_REVIEW,
        ArtifactStatus.FAILED,
    },
    ArtifactStatus.INTERRUPTED: {
        ArtifactStatus.PENDING,
        ArtifactStatus.GENERATING,
        ArtifactStatus.VALIDATING,
        ArtifactStatus.FAILED,
    },
    ArtifactStatus.FAILED: {
        ArtifactStatus.PENDING,
        ArtifactStatus.GENERATING,
        ArtifactStatus.MANUAL_REVIEW,
    },
    ArtifactStatus.MANUAL_REVIEW: {
        ArtifactStatus.PENDING,
        ArtifactStatus.VALIDATING,
        ArtifactStatus.APPROVED,  # Yalnızca açık manuel onay veya yeniden doğrulama sonrası
        ArtifactStatus.REJECTED,
    },
    ArtifactStatus.APPROVED: {
        ArtifactStatus.PENDING,   # Kaynak hash/parametre değiştiğinde geçersiz kılma
        ArtifactStatus.VALIDATING,
    },
}


class StateManager:
    """Tüm üretim çıktılarının durum geçiş kurallarını katı biçimde denetler."""

    @staticmethod
    def can_transition(current: ArtifactStatus, target: ArtifactStatus) -> bool:
        if current == target:
            return True
        return target in ALLOWED_TRANSITIONS.get(current, set())

    @staticmethod
    def transition(current: ArtifactStatus, target: ArtifactStatus, reason: str = "") -> ArtifactStatus:
        if not StateManager.can_transition(current, target):
            raise InvalidStateTransitionError(
                f"Geçersiz durum geçişi: {current.value} -> {target.value}"
                + (f" ({reason})" if reason else "")
            )
        return target

    @staticmethod
    def is_ready_for_next_stage(status: ArtifactStatus) -> bool:
        """Katı kural: Yalnızca APPROVED çıktılar sonraki aşamaya geçebilir."""
        return status == ArtifactStatus.APPROVED

    @staticmethod
    def decide_after_validation(
        passed: bool,
        attempts: int,
        max_attempts: int,
    ) -> ArtifactStatus:
        """
        Doğrulama sonucuna göre deterministik durum kararı.
        Maksimum deneme sayısına ulaşan başarısız çıktılar MANUAL_REVIEW olur.
        """
        if passed:
            return ArtifactStatus.APPROVED
        if attempts >= max_attempts:
            return ArtifactStatus.MANUAL_REVIEW
        return ArtifactStatus.REJECTED
