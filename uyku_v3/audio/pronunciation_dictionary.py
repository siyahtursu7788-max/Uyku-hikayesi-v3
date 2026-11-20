"""
Kullanıcı tarafından düzenlenebilir Türkçe Telaffuz Sözlüğü (Bölüm 5).
Kelime sınırı (word-boundary) güvenli değiştirme uygular;
örneğin 'hala' -> 'hâlâ' kuralı 'halat' veya 'mahala' kelimelerini ASLA bozmaz.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Union

from ..core.file_manager import atomic_write_json, safe_read_json


DEFAULT_TURKISH_PRONUNCIATION_DICT: Dict[str, str] = {
    # Özel isimler (uyku hikâyesi karakterleri & yer adları)
    "Arin": "Arin",
    "Liora": "Liyora",
    "Elias": "Eliyas",
    "Artur Vens": "Artur Vens",
    # Sık karşılaşılan şapkalı / ince okunuşlu Türkçe kelimeler
    "rüzgar": "rüzgâr",
    "Rüzgar": "Rüzgâr",
    "rüzgarın": "rüzgârın",
    "rüzgara": "rüzgâra",
    "rüzgarla": "rüzgârla",
    "kağıt": "kâğıt",
    "Kağıt": "Kâğıt",
    "kağıda": "kâğıda",
    "kağıdın": "kâğıdın",
    "kağıtta": "kâğıtta",
    "kağıttaki": "kâğıttaki",
    "hala": "hâlâ",
    "Hala": "Hâlâ",
    "hikaye": "hikâye",
    "Hikaye": "Hikâye",
    "hikayesi": "hikâyesi",
    "hikayeler": "hikâyeler",
    "sakin": "sâkin",
    "Sakin": "Sâkin",
    "yadigar": "yadigâr",
    "karargah": "karargâh",
}

# Türkçe harf karakter sınıfı (kelime sınırı kontrolü için)
_TR_WORD_CHAR = r"A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû"


class PronunciationDictionary:
    """JSON tabanlı, kullanıcı tarafından düzenlenebilir telaffuz sözlüğü."""

    def __init__(self, entries: Optional[Dict[str, str]] = None, path: Optional[Union[str, Path]] = None) -> None:
        self.path = Path(path) if path else None
        self.entries: Dict[str, str] = dict(DEFAULT_TURKISH_PRONUNCIATION_DICT)
        if self.path and self.path.exists():
            loaded = safe_read_json(self.path, default={})
            if isinstance(loaded, dict):
                self.entries.update({str(k): str(v) for k, v in loaded.items()})
        if entries:
            self.entries.update(entries)

    def add_entry(self, source: str, spoken: str, save: bool = False) -> None:
        self.entries[source] = spoken
        if save and self.path:
            self.save()

    def remove_entry(self, source: str, save: bool = False) -> None:
        self.entries.pop(source, None)
        if save and self.path:
            self.save()

    def save(self, target_path: Optional[Union[str, Path]] = None) -> None:
        p = Path(target_path) if target_path else self.path
        if p is None:
            raise ValueError("Sözlük kayıt yolu belirtilmedi.")
        atomic_write_json(p, self.entries)

    def apply(self, text: str) -> str:
        """
        Orijinal metni değiştirmeden yeni bir metin üzerinde kelime-sınırı güvenli
        telaffuz dönüşümü uygular. Uzun ifadeler (çok kelimeli özel isimler) önce eşleştirilir.
        """
        result = text
        for src in sorted(self.entries.keys(), key=len, reverse=True):
            dst = self.entries[src]
            if src == dst:
                continue
            pattern = re.compile(rf"(?<![{_TR_WORD_CHAR}]){re.escape(src)}(?![{_TR_WORD_CHAR}])")
            result = pattern.sub(dst, result)
        return result
