"""
Yeniden Tasarlanmış Çoklu Sinyalli Ses Kalite Denetçisi (Bölüm 6).
- SequenceMatcher skorunu TEK BAŞINA karar ölçütü olarak KULLANMAZ.
- WER, CER, dinamik programlama (DP) kelime hizalama, eksik/fazla kelime sayısı,
  son kelime & son hece (Türkçe fiil çekimleri), sondan ikinci kelime ve
  akustik sağlık metriklerini birlikte değerlendirir.
- Whisper konuşma tanıma hatası ile gerçek ses hatasını birbirinden ayırır:
  örneğin akustik ses sağlam, kelime sayısı ve süre tutarlı ancak Whisper özel ismi
  veya tek kelimeyi fonetik olarak farklı yazdıysa (yüksek karakter/sesletim benzerliği),
  bunu gerçek ses kesilmesi olarak değil 'whisper_recognition_uncertainty' olarak sınıflandırır.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config.quality_thresholds import AudioThresholds
from .audio_quality import AudioAcousticMetrics, analyze_audio_acoustics
from .text_normalizer import tr_norm_tokens


@dataclass
class AudioValidationResult:
    passed: bool
    needs_manual_review: bool
    wer: float
    cer: float
    word_alignment_ratio: float
    sequence_ratio: float
    missing_words: List[str]
    extra_words: List[str]
    last_word_ok: bool
    last_syllable_ok: bool
    second_last_word_ok: bool
    whisper_error_suspected: bool
    acoustic: Optional[Dict[str, Any]]
    expected_text: str
    transcribed_text: str
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _levenshtein(seq_a: List[str] | str, seq_b: List[str] | str) -> int:
    n, m = len(seq_a), len(seq_b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        ai = seq_a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == seq_b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def normalize_compound_token_splits(
    expected_tokens: List[str],
    actual_tokens: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Türkçe birleşik kelime tokenizasyon farklarını (örn. 'başparmağıyla' vs 'baş parmağıyla')
    YALNIZCA bitişik iki token harf-harf birebir aynı birleşimi veriyorsa normalize eder.
    Gerçek eksik kelimeleri veya anlam/ek değişikliklerini ASLA gizlemez.
    """
    norm_exp: List[str] = []
    norm_act: List[str] = []
    notes: List[str] = []
    i, j = 0, 0
    while i < len(expected_tokens) and j < len(actual_tokens):
        e = expected_tokens[i]
        a = actual_tokens[j]
        if e == a:
            norm_exp.append(e)
            norm_act.append(a)
            i += 1
            j += 1
        elif j + 1 < len(actual_tokens) and (actual_tokens[j] + actual_tokens[j + 1]) == e:
            norm_exp.append(e)
            norm_act.append(e)
            notes.append(f"compound_merged_actual:{actual_tokens[j]}+{actual_tokens[j+1]}->{e}")
            i += 1
            j += 2
        elif i + 1 < len(expected_tokens) and (expected_tokens[i] + expected_tokens[i + 1]) == a:
            norm_exp.append(expected_tokens[i])
            norm_exp.append(expected_tokens[i + 1])
            norm_act.append(expected_tokens[i])
            norm_act.append(expected_tokens[i + 1])
            notes.append(f"compound_split_actual:{a}->{expected_tokens[i]}+{expected_tokens[i+1]}")
            i += 2
            j += 1
        else:
            norm_exp.append(e)
            norm_act.append(a)
            i += 1
            j += 1
    if i < len(expected_tokens):
        norm_exp.extend(expected_tokens[i:])
    if j < len(actual_tokens):
        norm_act.extend(actual_tokens[j:])
    return norm_exp, norm_act, notes


def compute_wer_cer(expected_tokens: List[str], actual_tokens: List[str]) -> Tuple[float, float]:
    """Kelime Hata Oranı (WER) ve Karakter Hata Oranı (CER) hesaplar."""
    if not expected_tokens:
        return (0.0 if not actual_tokens else 1.0), (0.0 if not actual_tokens else 1.0)
    word_dist = _levenshtein(expected_tokens, actual_tokens)
    wer = word_dist / float(len(expected_tokens))

    exp_str = " ".join(expected_tokens)
    act_str = " ".join(actual_tokens)
    char_dist = _levenshtein(exp_str, act_str)
    cer = char_dist / float(max(1, len(exp_str)))
    return round(wer, 4), round(cer, 4)


def align_words_dp(
    expected_tokens: List[str],
    actual_tokens: List[str],
    fuzzy_threshold: float = 0.78,
) -> Tuple[List[str], List[str], float]:
    """
    Beklenen ve okunan kelimeleri sıralı biçimde hizalar.
    Döndürür: (eksik_kelimeler, fazla_kelimeler, hizalama_orani)
    """
    sm = difflib.SequenceMatcher(None, expected_tokens, actual_tokens, autojunk=False)
    missing: List[str] = []
    extra: List[str] = []
    matched = 0

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            matched += (i2 - i1)
        elif tag == "delete":
            missing.extend(expected_tokens[i1:i2])
        elif tag == "insert":
            extra.extend(actual_tokens[j1:j2])
        elif tag == "replace":
            exp_slice = expected_tokens[i1:i2]
            act_slice = actual_tokens[j1:j2]
            pair_count = min(len(exp_slice), len(act_slice))
            for k in range(pair_count):
                sim = difflib.SequenceMatcher(None, exp_slice[k], act_slice[k]).ratio()
                if sim >= fuzzy_threshold:
                    matched += 1
                else:
                    missing.append(exp_slice[k])
                    extra.append(act_slice[k])
            if len(exp_slice) > pair_count:
                missing.extend(exp_slice[pair_count:])
            if len(act_slice) > pair_count:
                extra.extend(act_slice[pair_count:])

    ratio = matched / float(max(1, len(expected_tokens)))
    return missing, extra, round(ratio, 4)


def _check_turkish_last_syllable(expected_word: str, actual_word: str) -> bool:
    """
    Türkçe fiil çekimlerinin son hecesinin düşüp düşmediğini denetler.
    Örn: 'dinliyordu' -> 'dinliyor' (son hece 'du' düşmüş) yakalanır!
    """
    if not expected_word or not actual_word:
        return False
    if expected_word == actual_word:
        return True
    # Hedef kelime >= 5 harf ise son 2 karakterin korunması beklenir
    if len(expected_word) >= 5:
        if expected_word.startswith(actual_word) and (len(expected_word) - len(actual_word)) >= 2:
            return False
        return expected_word[-2:] == actual_word[-2:] or difflib.SequenceMatcher(None, expected_word[-3:], actual_word[-3:]).ratio() >= 0.75
    return difflib.SequenceMatcher(None, expected_word, actual_word).ratio() >= 0.78


def validate_sentence_audio(
    expected_text: str,
    transcribed_text: str,
    audio_signal: Optional[np.ndarray] = None,
    sr: int = 24000,
    word_timestamps: Optional[List[Dict[str, Any]]] = None,
    thresholds: Optional[AudioThresholds] = None,
    known_proper_nouns: Optional[List[str]] = None,
) -> AudioValidationResult:
    """
    Bağımsız çoklu kontrol ile cümle sesini doğrular:
    1. WER / CER + kelime hizalama (eksik/fazla kelime)
    2. Son kelime, son Türkçe hece ve sondan ikinci kelime kontrolü
    3. Akustik enerji, clipping, hız (WPM), iç sessizlik ve ani kesilme kontrolü
    4. Whisper yanlış tanıma vs gerçek ses hatası ayrımı
    """
    th = thresholds or AudioThresholds()
    raw_exp_tokens = tr_norm_tokens(expected_text)
    raw_act_tokens = tr_norm_tokens(transcribed_text)
    exp_tokens, act_tokens, _compound_notes = normalize_compound_token_splits(raw_exp_tokens, raw_act_tokens)

    wer, cer = compute_wer_cer(exp_tokens, act_tokens)
    missing, extra, align_ratio = align_words_dp(exp_tokens, act_tokens)
    seq_ratio = round(
        difflib.SequenceMatcher(None, " ".join(exp_tokens), " ".join(act_tokens)).ratio(), 4
    )

    # Son kelime, son hece ve sondan ikinci kelime kontrolleri
    last_word_ok = False
    last_syllable_ok = False
    if exp_tokens and act_tokens:
        last_sim = difflib.SequenceMatcher(None, exp_tokens[-1], act_tokens[-1]).ratio()
        last_word_ok = last_sim >= th.min_last_word_similarity
        last_syllable_ok = _check_turkish_last_syllable(exp_tokens[-1], act_tokens[-1])
    elif not exp_tokens:
        last_word_ok = True
        last_syllable_ok = True

    second_last_ok = True
    if len(exp_tokens) >= 2:
        if len(act_tokens) >= 2:
            candidates = act_tokens[-3:-1] if len(act_tokens) >= 3 else [act_tokens[-2]]
            second_last_ok = any(
                difflib.SequenceMatcher(None, exp_tokens[-2], cand).ratio() >= th.min_second_last_word_similarity
                for cand in candidates
            )
        else:
            second_last_ok = False

    acoustic_res: Optional[AudioAcousticMetrics] = None
    if audio_signal is not None:
        acoustic_res = analyze_audio_acoustics(audio_signal, sr, len(exp_tokens), th)

    # Whisper yanlış tanıma vs gerçek ses hatası ayrımı:
    # Akustik sağlam + kelime sayısı birebir aynı + son kelime & hece sağlam +
    # yalnızca özel isim veya düşük CER farkı varsa Whisper tanıma belirsizliği işaretlenir.
    proper_set = {t for n in (known_proper_nouns or []) for t in tr_norm_tokens(n)}
    # Sıkı Özel İsim & CER Kuralı:
    # CER <= 0.05 olan genel yazım farkları ASLA otomatik onaylanmaz.
    # Yalnızca açıkça bilinen özel isim sözlüğündeki (known_proper_nouns) tek bir kelime
    # farklı yazılmışsa, son kelime + son hece + sondan ikinci kelime ve akustik sağlamsa
    # Whisper özel isim tanıma belirsizliği olarak işaretlenir ve MANUAL_REVIEW kalır.
    diff_exp = [e for e, a in zip(exp_tokens, act_tokens) if e != a] if len(exp_tokens) == len(act_tokens) else []
    only_proper_or_phonetic = (
        len(exp_tokens) == len(act_tokens)
        and len(diff_exp) == 1
        and (diff_exp[0] in proper_set)
        and len(missing) <= 1
        and len(extra) <= 1
        and cer <= 0.14
        and last_word_ok
        and last_syllable_ok
        and second_last_ok
        and (acoustic_res is None or acoustic_res.passed)
    )

    issues: List[str] = []
    if acoustic_res is not None and not acoustic_res.passed:
        issues.extend(acoustic_res.issues)
    if not last_word_ok:
        issues.append(f"Son kelime uyuşmuyor: beklenen='{exp_tokens[-1] if exp_tokens else ''}', okunan='{act_tokens[-1] if act_tokens else ''}'")
    if not last_syllable_ok:
        issues.append(f"Son Türkçe hece düşmüş/bozulmuş olabilir: '{exp_tokens[-1] if exp_tokens else ''}' vs '{act_tokens[-1] if act_tokens else ''}'")
    if not second_last_ok:
        issues.append("Sondan ikinci kelime doğrulanamadı.")
    if len(missing) > th.max_missing_words and not only_proper_or_phonetic:
        issues.append(f"Eksik kelime tespit edildi: {missing}")
    if len(extra) > th.max_extra_words:
        issues.append(f"Fazladan kelime tespit edildi: {extra}")
    if wer > th.max_wer and not only_proper_or_phonetic:
        issues.append(f"WER eşiği aşıldı ({wer:.3f} > {th.max_wer}).")
    if cer > th.max_cer and not only_proper_or_phonetic:
        issues.append(f"CER eşiği aşıldı ({cer:.3f} > {th.max_cer}).")
    if align_ratio < th.min_word_alignment_ratio and not only_proper_or_phonetic:
        issues.append(f"Kelime hizalama oranı düşük ({align_ratio:.3f} < {th.min_word_alignment_ratio}).")

    # Kelime zaman damgaları varsa cümle ortasında kesilme kontrolü
    if word_timestamps and audio_signal is not None:
        audio_dur = len(audio_signal) / float(sr)
        last_ts_end = max((float(w.get("end", 0.0)) for w in word_timestamps), default=0.0)
        if audio_dur > 1.0 and last_ts_end < audio_dur * 0.45 and len(act_tokens) < len(exp_tokens):
            issues.append(f"Ses erken bitmiş/kesilmiş görünüyor (son kelime {last_ts_end:.2f}s / toplam {audio_dur:.2f}s).")

    passed = len(issues) == 0
    needs_manual = only_proper_or_phonetic and (wer > th.max_wer or len(missing) > 0)

    return AudioValidationResult(
        passed=passed,
        needs_manual_review=needs_manual,
        wer=wer,
        cer=cer,
        word_alignment_ratio=align_ratio,
        sequence_ratio=seq_ratio,
        missing_words=missing,
        extra_words=extra,
        last_word_ok=last_word_ok,
        last_syllable_ok=last_syllable_ok,
        second_last_word_ok=second_last_ok,
        whisper_error_suspected=only_proper_or_phonetic,
        acoustic=acoustic_res.to_dict() if acoustic_res else None,
        expected_text=expected_text,
        transcribed_text=transcribed_text,
        issues=issues,
    )


def validate_paragraph_audio(
    sentences: List[str],
    paragraph_transcript: str,
    audio_signal: Optional[np.ndarray] = None,
    sr: int = 24000,
    thresholds: Optional[AudioThresholds] = None,
) -> AudioValidationResult:
    """
    P0-01 DÜZELTMESİ:
    Paragraf düzeyinde tamlık skoru, eksik yüklem ve WER/CER kontrolü yapar.
    Düşük tamlık skorlu (örn. 0.15, 0.42, 0.075) bir paragraf ASLA passed=True döndüremez!
    """
    th = thresholds or AudioThresholds()
    full_expected = " ".join(sentences)
    res = validate_sentence_audio(
        expected_text=full_expected,
        transcribed_text=paragraph_transcript,
        audio_signal=audio_signal,
        sr=sr,
        thresholds=th,
    )
    if res.word_alignment_ratio < th.min_paragraph_word_coverage:
        res.passed = False
        msg = f"Paragraf kelime kapsama oranı yetersiz ({res.word_alignment_ratio:.3f} < {th.min_paragraph_word_coverage})."
        if msg not in res.issues:
            res.issues.append(msg)
    return res
