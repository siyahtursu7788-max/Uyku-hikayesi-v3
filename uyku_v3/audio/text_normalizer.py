"""
Türkçe Metin Hazırlama ve Normalizasyon Motoru (Bölüm 5).
- P0-05 DÜZELTMESİ: Türkçe 'İ' -> 'i' ve 'I' -> 'ı' dönüşümü .lower() ÖNCESİNDE yapılır.
- 0..999.999 tam sayı, sıra sayısı (400. -> dört yüzüncü), saat (21:30), tarih ve kısaltma normalizasyonu.
- Orijinal senaryo metnini ASLA değiştirmez; seslendirme için ayrı normalize edilmiş metin döndürür.
"""

from __future__ import annotations

import re
from typing import List, Optional

from .pronunciation_dictionary import PronunciationDictionary

_ONES = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
_TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]

_ORDINAL_SUFFIX = {
    "bir": "birinci",
    "iki": "ikinci",
    "üç": "üçüncü",
    "dört": "dördüncü",
    "beş": "beşinci",
    "altı": "altıncı",
    "yedi": "yedinci",
    "sekiz": "sekizinci",
    "dokuz": "dokuzuncu",
    "on": "onuncu",
    "yirmi": "yirminci",
    "otuz": "otuzuncu",
    "kırk": "kırkıncı",
    "elli": "ellinci",
    "altmış": "altmışıncı",
    "yetmiş": "yetmişinci",
    "seksen": "sekseninci",
    "doksan": "doksanıncı",
    "yüz": "yüzüncü",
    "bin": "bininci",
    "sıfır": "sıfırıncı",
}

_ABBREVIATIONS = {
    "Dr.": "Doktor",
    "Prof.": "Profesör",
    "sn.": "saniye",
    "dk.": "dakika",
    "km.": "kilometre",
    "cm.": "santimetre",
    "mm.": "milimetre",
    "vb.": "ve benzeri",
    "vd.": "ve diğerleri",
    "yy.": "yüzyıl",
}

_MONTHS_TR = {
    1: "ocak", 2: "şubat", 3: "mart", 4: "nisan", 5: "mayıs", 6: "haziran",
    7: "temmuz", 8: "ağustos", 9: "eylül", 10: "ekim", 11: "kasım", 12: "aralık",
}


def tr_lower(text: str) -> str:
    """Türkçe kurallarına %100 uygun küçük harf dönüşümü (İ->i, I->ı)."""
    return text.replace("İ", "i").replace("I", "ı").lower().replace("\u0307", "")


def number_to_turkish_words(n: int) -> str:
    """0..999.999 arası tam sayıları doğru Türkçe yazıma çevirir."""
    if n < 0:
        return "eksi " + number_to_turkish_words(-n)
    if n == 0:
        return "sıfır"
    if n >= 1_000_000:
        millions = n // 1_000_000
        rem = n % 1_000_000
        head = f"{number_to_turkish_words(millions)} milyon"
        return f"{head} {number_to_turkish_words(rem)}".strip() if rem else head

    parts: List[str] = []
    thousands = n // 1000
    rem = n % 1000
    if thousands == 1:
        parts.append("bin")
    elif thousands > 1:
        parts.append(f"{number_to_turkish_words(thousands)} bin")

    hundreds = rem // 100
    rem2 = rem % 100
    if hundreds == 1:
        parts.append("yüz")
    elif hundreds > 1:
        parts.append(f"{_ONES[hundreds]} yüz")

    tens = rem2 // 10
    ones = rem2 % 10
    if tens:
        parts.append(_TENS[tens])
    if ones:
        parts.append(_ONES[ones])

    return " ".join(parts)


def ordinal_to_turkish_words(n: int) -> str:
    words = number_to_turkish_words(n).split()
    if not words:
        return ""
    last = words[-1]
    words[-1] = _ORDINAL_SUFFIX.get(last, last + "inci")
    return " ".join(words)


def expand_turkish_numbers_and_dates(text: str) -> str:
    """Tarihler (14.05.1924), saatler (21:30), sıra sayıları (400.) ve tam sayıları Türkçeye açar."""
    out = text

    # Tarihler: GG.AA.YYYY
    def _repl_date(m: re.Match[str]) -> str:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= day <= 31 and 1 <= month <= 12:
            return f"{number_to_turkish_words(day)} {_MONTHS_TR[month]} {number_to_turkish_words(year)}"
        return m.group(0)

    out = re.sub(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", _repl_date, out)

    # Saatler: SS:DD
    def _repl_time(m: re.Match[str]) -> str:
        hh, mm = int(m.group(1)), int(m.group(2))
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            if mm == 0:
                return f"saat {number_to_turkish_words(hh)}"
            return f"{number_to_turkish_words(hh)} {number_to_turkish_words(mm)}"
        return m.group(0)

    out = re.sub(r"\b(\d{1,2}):(\d{2})\b", _repl_time, out)

    # Ondalık sayılar: 3,5 veya 3.5 (tarih dışındaki)
    def _repl_decimal(m: re.Match[str]) -> str:
        whole, frac = int(m.group(1)), int(m.group(2))
        return f"{number_to_turkish_words(whole)} virgül {number_to_turkish_words(frac)}"

    out = re.sub(r"\b(\d+)[,.](\d+)\b", _repl_decimal, out)

    # Sıra sayıları: 400. veya 12. (ardından boşluk veya küçük harf geliyorsa)
    def _repl_ordinal(m: re.Match[str]) -> str:
        val = int(m.group(1))
        return ordinal_to_turkish_words(val)

    out = re.sub(r"\b(\d+)\.(?=\s+[a-zçğıöşüA-ZÇĞİÖŞÜ])", _repl_ordinal, out)

    # Kalan tam sayılar (kesme ekli '400'üncü' vb. dahil)
    def _repl_int(m: re.Match[str]) -> str:
        val = int(m.group(1))
        return number_to_turkish_words(val)

    out = re.sub(r"\b(\d+)\b", _repl_int, out)
    return out


def tr_norm_tokens(text: str) -> List[str]:
    """
    Whisper transkripti ile hedef senaryo metnini adil karşılaştırmak için
    Türkçe normalize kelime listesi üretir.
    - İ/I dönüşümü doğru sırada yapılır.
    - Şapkalı karakterler eşleştirme için normalize edilir.
    - Kesme işaretleri bitişik tutulur (Pasifik'in -> pasifikin).
    - Tüm sayılar Türkçe kelimelere açılır (400 -> dört yüz).
    """
    expanded = expand_turkish_numbers_and_dates(text)
    s = tr_lower(expanded)
    s = s.replace("â", "a").replace("î", "i").replace("û", "u")
    s = s.replace("'", "").replace("’", "").replace("‘", "")
    # Birleşik yazılmış sayı kalıplarını standartlaştır
    s = re.sub(r"[^a-zçğıöşü0-9\s]", " ", s)
    return [tok for tok in s.split() if tok]


def normalize_for_tts(
    raw_text: str,
    pronunciation_dict: Optional[PronunciationDictionary] = None,
    preserve_circumflex: bool = True,
) -> str:
    """
    Orijinal metni DEĞİŞTİRMEDEN XTTS-v2 için temiz, sakin uyku hikâyesi akışına uygun
    seslendirme metni oluşturur.
    - Telaffuz sözlüğü uygulanır ve şapkalı harfler (preserve_circumflex=True) korunur
      (P1-06 hatası giderildi: sözlük sonrası şapkalar silinmez!).
    - Kısaltmalar, tarihler ve sayılar Türkçeye açılır.
    - XTTS'de nefes patlamasına yol açan üç nokta (...) ve gereksiz tırnaklar temizlenir.
    """
    m = raw_text.strip()
    if not m:
        return ""

    for abbr, full in _ABBREVIATIONS.items():
        m = re.sub(rf"(?<!\w){re.escape(abbr)}", full, m)

    m = expand_turkish_numbers_and_dates(m)

    if pronunciation_dict is not None:
        m = pronunciation_dict.apply(m)

    if not preserve_circumflex:
        m = m.replace("â", "a").replace("Â", "A").replace("î", "i").replace("û", "u")

    m = m.replace("...", ",").replace("…", ",").replace(":", ",")
    m = m.replace('"', "").replace("“", "").replace("”", "").replace("—", " ")
    m = m.replace("’", "'").replace("‘", "'")
    # Virgüllerde sert nefes patlamasını önlemek için fazla virgülleri temizle
    m = re.sub(r",\s*,+", ",", m)
    m = m.replace(",", " ")
    m = re.sub(r"\s+", " ", m).strip()
    if m and not m.endswith((".", "!", "?")):
        m += "."
    return m


def split_into_breath_groups(
    line: str,
    pronunciation_dict: Optional[PronunciationDictionary] = None,
    max_words_per_group: int = 24,
) -> List[str]:
    """
    Nokta, ünlem, soru işareti ve noktalı virgüllere (';') göre doğal Türkçe nefes gruplarına ayırır.
    Çok uzun cümleleri (>max_words_per_group) doğal bağlaç/virgül sınırından böler.
    """
    raw_parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|;\s*", line.strip()) if p.strip()]
    groups: List[str] = []
    for part in raw_parts:
        words = part.split()
        if len(words) <= max_words_per_group:
            cleaned = normalize_for_tts(part, pronunciation_dict=pronunciation_dict)
            if cleaned:
                groups.append(cleaned)
        else:
            # Uzun cümleyi önce virgüllerden, yetmezse güvenli kelime sınırından böl
            sub_chunks = [c.strip() for c in part.split(",") if c.strip()]
            buf = ""
            for sc in sub_chunks:
                cand = f"{buf} {sc}".strip() if buf else sc
                if len(cand.split()) > max_words_per_group and buf:
                    cleaned = normalize_for_tts(buf, pronunciation_dict=pronunciation_dict)
                    if cleaned:
                        groups.append(cleaned)
                    buf = sc
                else:
                    buf = cand
            if buf:
                cleaned = normalize_for_tts(buf, pronunciation_dict=pronunciation_dict)
                if cleaned:
                    groups.append(cleaned)
    return groups


def group_short_sentences(sentences: List[str]) -> List[str]:
    """
    Yalnızca 1 kelimelik satırları veya arka arkaya gelen 1-2 kelimelik çok kısa satırları
    birleştirerek XTTS ton sıçramasını ve kelime yutmayı önler.
    """
    grouped: List[str] = []
    for s in sentences:
        wc = len(tr_norm_tokens(s))
        if grouped and wc <= 2 and len(tr_norm_tokens(grouped[-1])) <= 3:
            prev = grouped[-1].rstrip(".!?").strip()
            grouped[-1] = f"{prev} {tr_lower(s[0])}{s[1:]}" if s else prev
        elif grouped and wc == 1:
            prev = grouped[-1].rstrip(".!?").strip()
            grouped[-1] = f"{prev} {tr_lower(s[0])}{s[1:]}" if s else prev
        else:
            grouped.append(s)
    return grouped


def split_long_sentence_for_tts(
    raw_sentence: str,
    pronunciation_dict: Optional[PronunciationDictionary] = None,
    min_words_to_split: int = 9,
    min_side_words: int = 3,
) -> List[str]:
    """
    V2.2 Akıllı Nefes Grubu Bölme:
    - Orijinal senaryo ve SRT metnini ASLA değiştirmez; yalnızca TTS girdisi için çalışır.
    - Uzun (>=9 kelimelik) ve içinde doğal virgül bulunan cümleleri merkeze en yakın
      virgülden iki nefes grubuna ayırır (her iki tarafta en az 3 kelime şartıyla).
    """
    s = raw_sentence.strip()
    if not s:
        return []
    tokens = tr_norm_tokens(s)
    if len(tokens) < min_words_to_split or "," not in s:
        norm = normalize_for_tts(s, pronunciation_dict=pronunciation_dict)
        return [norm] if norm else []

    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) < 2:
        norm = normalize_for_tts(s, pronunciation_dict=pronunciation_dict)
        return [norm] if norm else []

    best_idx = None
    best_diff = 10**9
    for idx in range(1, len(parts)):
        left = ", ".join(parts[:idx])
        right = ", ".join(parts[idx:])
        lw = len(tr_norm_tokens(left))
        rw = len(tr_norm_tokens(right))
        if lw >= min_side_words and rw >= min_side_words:
            diff = abs(lw - rw)
            if diff < best_diff:
                best_diff = diff
                best_idx = idx

    if best_idx is None:
        norm = normalize_for_tts(s, pronunciation_dict=pronunciation_dict)
        return [norm] if norm else []

    left_norm = normalize_for_tts(", ".join(parts[:best_idx]), pronunciation_dict=pronunciation_dict)
    right_norm = normalize_for_tts(", ".join(parts[best_idx:]), pronunciation_dict=pronunciation_dict)
    return [x for x in (left_norm, right_norm) if x]
