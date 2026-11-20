"""UYKU_HIKAYESI_V3_FINAL_PRODUCTION — Eski dosya SHA-256 koruma bekçisi (tekil dosya dökümlü)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .file_manager import sha256_file


class LegacyProtectionGuard:
    """İşlem öncesi ve sonrası korunan tüm dosyaların tekil SHA-256 karşılaştırmasını yapar."""

    def __init__(self, protected_paths: Dict[str, Path]) -> None:
        self.protected_paths = {k: Path(v) for k, v in protected_paths.items()}
        self.pre_hashes: Dict[str, str] = {}

    def snapshot_before(self, strict_missing: bool = False) -> Dict[str, str]:
        self.pre_hashes = {}
        for name, path in self.protected_paths.items():
            if path.exists() and path.is_file():
                self.pre_hashes[name] = sha256_file(path)
            elif strict_missing:
                raise FileNotFoundError(f"Korunan dosya eksik: {name} -> {path}")
            else:
                self.pre_hashes[name] = "MISSING_PRE"
        return dict(self.pre_hashes)

    def verify_after(self) -> Dict[str, Any]:
        mismatches: List[Dict[str, str]] = []
        per_file: Dict[str, Dict[str, Any]] = {}
        for name, pre in self.pre_hashes.items():
            path = self.protected_paths[name]
            post = sha256_file(path) if (path.exists() and path.is_file()) else "MISSING_POST"
            ok = pre == post
            per_file[name] = {
                "path": str(path),
                "pre_sha256": pre,
                "post_sha256": post,
                "unchanged": ok,
            }
            if not ok:
                mismatches.append(per_file[name])
        return {
            "all_protected_ok": len(mismatches) == 0,
            "checked_count": len(self.pre_hashes),
            "mismatches": mismatches,
            "files": per_file,
        }
