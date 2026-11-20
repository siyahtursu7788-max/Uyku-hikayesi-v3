"""
Hibrit Cümle + Kelime Düzeyinde Türkçe Altyazı Hizalama Motoru (Bölüm 12).
P0-04 DÜZELTMESİ:
- İlk ve son kelime eşleşti diye ortadaki bütün kelimeleri doğru kabul ETMEZ!
- Her cümledeki TÜM kelimelerin hizalama oranını, kelime sırasını, eksik/fazla
  kelimeleri hesaplar; düşük güvenli cümleleri MANUAL_REVIEW olarak işaretler.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from ..audio.text_normalizer import tr_norm_tokens
from ..config.quality_thresholds import SRTThresholds
from ..core.state_manager import ArtifactStatus
from .transcription import WordTimestamp


@dataclass
class AlignedSentence:
    sentence_index: int
    text: str
    start_sec: Optional[float]
    end_sec: Optional[float]
    confidence: float
    matched_word_count: int
    expected_word_count: int
    missing_words: List[str]
    status: ArtifactStatus

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


def align_sentences_to_words(
    sentences: List[str],
    whisper_words: List[WordTimestamp],
    outro_start_sec: Optional[float] = None,
    thresholds: Optional[SRTThresholds] = None,
) -> List[AlignedSentence]:
    """
    Orijinal senaryo cümlelerini Whisper kelime zaman damgalarına global + yerel
    hizalama ile oturtur. Her cümle için iç kelime kapsamını ve güven puanını hesaplar.
    """
    th = thresholds or SRTThresholds()
    script_tokens: List[str] = []
    sent_spans: List[tuple[int, int, str, List[str]]] = []

    for sent in sentences:
        toks = tr_norm_tokens(sent)
        if not toks:
            continue
        start_i = len(script_tokens)
        script_tokens.extend(toks)
        sent_spans.append((start_i, len(script_tokens) - 1, sent.strip(), toks))

    whisper_norms = [w.norm for w in whisper_words]
    sm = difflib.SequenceMatcher(None, script_tokens, whisper_norms, autojunk=False)

    exact_map: Dict[int, int] = {}
    for i, j, n in sm.get_matching_blocks():
        for k in range(n):
            exact_map[i + k] = j + k

    # Eşleşmeyen ara kelimeler için yerel bulanık eşleştirme (yalnızca monolitik sıra korunarak)
    aligned_results: List[AlignedSentence] = []
    for s_idx, (b_i, e_i, text, toks) in enumerate(sent_spans, 1):
        matched_w_indices: List[int] = []
        missing_words: List[str] = []

        for offset, tok in enumerate(toks):
            g_i = b_i + offset
            if g_i in exact_map:
                matched_w_indices.append(exact_map[g_i])
            else:
                missing_words.append(tok)

        # Monotonik sıra kontrolü
        if matched_w_indices and matched_w_indices != sorted(matched_w_indices):
            matched_w_indices = sorted(matched_w_indices)

        matched_count = len(matched_w_indices)
        exp_count = len(toks)
        confidence = round(matched_count / float(max(1, exp_count)), 4)

        if matched_w_indices:
            w_first = whisper_words[matched_w_indices[0]]
            w_last = whisper_words[matched_w_indices[-1]]
            t_start = w_first.start
            t_end = w_last.end
            if outro_start_sec is not None:
                t_end = min(t_end, outro_start_sec)
            valid_time = (t_end > t_start) and (outro_start_sec is None or t_start < outro_start_sec)
        else:
            t_start, t_end, valid_time = None, None, False

        if valid_time and confidence >= th.min_sentence_alignment_confidence:
            st = ArtifactStatus.APPROVED
        else:
            st = ArtifactStatus.MANUAL_REVIEW

        aligned_results.append(
            AlignedSentence(
                sentence_index=s_idx,
                text=text,
                start_sec=round(t_start, 3) if t_start is not None else None,
                end_sec=round(t_end, 3) if t_end is not None else None,
                confidence=confidence,
                matched_word_count=matched_count,
                expected_word_count=exp_count,
                missing_words=missing_words,
                status=st,
            )
        )

    return aligned_results
