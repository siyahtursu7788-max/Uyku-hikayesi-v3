"""UYKU_HIKAYESI_V3_FINAL_PRODUCTION — 6 faktörlü SHA-256 önbellek motoru."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .file_manager import atomic_copy_file, atomic_write_json, sha256_file, sha256_text


@dataclass(frozen=True)
class SixFactorCacheKey:
    """Her çıktı için 6 bağımsız hash faktörü (herhangi biri değişirse önbellek geçersizleşir)."""

    text_hash: str
    model_hash: str
    reference_audio_hash: str
    params_hash: str
    thresholds_config_hash: str
    source_file_hash: str

    def digest(self) -> str:
        payload = json.dumps(
            {
                "text_hash": self.text_hash,
                "model_hash": self.model_hash,
                "reference_audio_hash": self.reference_audio_hash,
                "params_hash": self.params_hash,
                "thresholds_config_hash": self.thresholds_config_hash,
                "source_file_hash": self.source_file_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256_text(payload)


class VerifiedV3Cache:
    """Yalnızca APPROVED çıktıları saklayan ve her okumada SHA-256 doğrulayan önbellek."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.corrupt_rejections = 0

    def _paths(self, key_digest: str, ext: str) -> tuple[Path, Path]:
        ext_norm = ext if ext.startswith(".") else f".{ext}"
        return self.cache_dir / f"{key_digest}{ext_norm}", self.cache_dir / f"{key_digest}.meta.json"

    def get(self, key: SixFactorCacheKey, ext: str = ".wav") -> Optional[Dict[str, Any]]:
        dg = key.digest()
        data_p, meta_p = self._paths(dg, ext)
        if not (data_p.exists() and meta_p.exists()):
            self.misses += 1
            return None
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
        except Exception:
            self.corrupt_rejections += 1
            self.misses += 1
            return None
        if meta.get("status") != "APPROVED" or meta.get("cache_key") != dg:
            self.corrupt_rejections += 1
            self.misses += 1
            return None
        expected_sha = meta.get("file_sha256", "")
        if not expected_sha or data_p.stat().st_size == 0 or sha256_file(data_p) != expected_sha:
            self.corrupt_rejections += 1
            self.misses += 1
            return None
        self.hits += 1
        return {"path": data_p, "meta": meta, "cache_key": dg}

    def put_approved(
        self,
        key: SixFactorCacheKey,
        source_file: Path,
        metadata: Optional[Dict[str, Any]] = None,
        status: str = "APPROVED",
    ) -> Dict[str, Any]:
        if status != "APPROVED":
            raise ValueError("FAIL-CLOSED: Önbelleğe yalnızca APPROVED çıktılar yazılabilir.")
        src = Path(source_file)
        if not src.exists() or src.stat().st_size == 0:
            raise FileNotFoundError(f"Önbelleğe alınacak dosya geçersiz: {src}")
        dg = key.digest()
        data_p, meta_p = self._paths(dg, src.suffix or ".bin")
        file_sha = atomic_copy_file(src, data_p)
        meta = {
            "cache_key": dg,
            "status": "APPROVED",
            "file_sha256": file_sha,
            "file_size_bytes": data_p.stat().st_size,
            "factors": {
                "text_hash": key.text_hash,
                "model_hash": key.model_hash,
                "reference_audio_hash": key.reference_audio_hash,
                "params_hash": key.params_hash,
                "thresholds_config_hash": key.thresholds_config_hash,
                "source_file_hash": key.source_file_hash,
            },
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        atomic_write_json(meta_p, meta)
        return {"path": data_p, "meta": meta, "cache_key": dg}
