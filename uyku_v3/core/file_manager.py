"""
Atomik dosya yazımı, SHA-256 hash doğrulaması ve Google Drive FUSE kesinti koruması.
Yarım yazılmış dosyalar asla nihai dosya adıyla görünmez.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Union


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_dict(obj: Any) -> str:
    canonical = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256_text(canonical)


def sha256_file(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # Drive FUSE üzerinde dir fsync her zaman desteklenmeyebilir
        pass


def atomic_write_bytes(target_path: Union[str, Path], data: bytes) -> str:
    """Veriyi geçici .tmp dosyasına yazıp fsync sonrası atomik olarak hedefe taşır."""
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{int(time.time() * 1000)}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
        _fsync_dir(target.parent)
        return sha256_bytes(data)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def atomic_write_text(target_path: Union[str, Path], text: str) -> str:
    return atomic_write_bytes(target_path, text.encode("utf-8"))


def atomic_write_json(target_path: Union[str, Path], payload: Dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    return atomic_write_text(target_path, serialized)


def atomic_copy_file(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
    verify_hash: bool = True,
    retries: int = 3,
    retry_delay_sec: float = 1.0,
) -> str:
    """
    Yerel SSD'den Google Drive'a (veya başka bir dizine) atomik ve hash doğrulamalı kopyalama.
    Drive bağlantısı anlık koparsa veri kaybı veya yarım dosya oluşmasını önler.
    """
    src = Path(src_path)
    dst = Path(dst_path)
    if not src.exists():
        raise FileNotFoundError(f"Kaynak dosya bulunamadı: {src}")

    src_hash = sha256_file(src)
    dst.parent.mkdir(parents=True, exist_ok=True)

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        tmp = dst.with_name(f".{dst.name}.{os.getpid()}.{attempt}.tmp")
        try:
            shutil.copy2(src, tmp)
            if verify_hash:
                tmp_hash = sha256_file(tmp)
                if tmp_hash != src_hash:
                    raise IOError(f"Kopyalama hash uyuşmazlığı: {tmp_hash} != {src_hash}")
            os.replace(tmp, dst)
            _fsync_dir(dst.parent)
            return src_hash
        except Exception as exc:
            last_err = exc
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            if attempt < retries:
                time.sleep(retry_delay_sec)

    raise IOError(f"Atomik kopyalama {retries} denemede başarısız oldu ({src} -> {dst}): {last_err}")


def safe_read_json(path: Union[str, Path], default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return {} if default is None else default
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {} if default is None else default
