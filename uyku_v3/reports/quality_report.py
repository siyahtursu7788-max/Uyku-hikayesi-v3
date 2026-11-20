"""
Bölüm 17: Otomatik Son Kalite Raporu Üreticisi (JSON + Markdown).
Katı kural: Herhangi bir kritik hata veya MANUAL_REVIEW bekleyen aşama varsa
genel proje sonucu ASLA APPROVED olamaz!
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from ..core.file_manager import atomic_write_json, atomic_write_text
from ..core.state_manager import ArtifactStatus


def build_final_quality_report(
    project_id: str,
    audio_metrics: Dict[str, Any],
    visual_metrics: Dict[str, Any],
    video_metrics: Dict[str, Any],
    assembly_metrics: Dict[str, Any],
    subtitle_metrics: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    critical_blockers: List[str] = []
    manual_review_items: List[str] = []

    if audio_metrics.get("manual_review_count", 0) > 0:
        manual_review_items.append(f"Ses: {audio_metrics['manual_review_count']} cümle/paragraf manuel inceleme bekliyor.")
    if not audio_metrics.get("passed", False):
        critical_blockers.append("Ses kalite kontrolü geçemedi.")

    if not visual_metrics.get("passed", False):
        critical_blockers.append("Görsel kalite kontrolü geçemedi.")

    if video_metrics.get("manual_review_count", 0) > 0:
        manual_review_items.append(f"Video: {video_metrics['manual_review_count']} klip manuel inceleme bekliyor.")
    if video_metrics.get("rejected_clips", 0) > 0 or not video_metrics.get("passed", False):
        critical_blockers.append("Video kalite kontrolü geçemedi veya reddedilmiş klip var.")

    if not assembly_metrics.get("decode_ok", False) or not assembly_metrics.get("passed", False):
        critical_blockers.append("Final montaj / decode / senkron doğrulaması geçemedi.")

    if not subtitle_metrics.get("passed", False) or subtitle_metrics.get("overlap_count", 0) > 0 or subtitle_metrics.get("black_screen_violations", 0) > 0:
        critical_blockers.append("SRT altyazı doğrulaması geçemedi.")

    if critical_blockers:
        final_status = ArtifactStatus.REJECTED.value
    elif manual_review_items:
        final_status = ArtifactStatus.MANUAL_REVIEW.value
    else:
        final_status = ArtifactStatus.APPROVED.value

    report = {
        "project_id": project_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "final_status": final_status,
        "critical_blockers": critical_blockers,
        "manual_review_items": manual_review_items,
        "SES": audio_metrics,
        "GORSEL": visual_metrics,
        "VIDEO": video_metrics,
        "MONTAJ": assembly_metrics,
        "ALTYAZI": subtitle_metrics,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / f"{project_id}_SON_KALITE_RAPORU.json", report)

    md_lines = [
        f"# Son Kalite Denetim Raporu — {project_id}",
        f"- **Tarih (UTC):** {report['generated_at']}",
        f"- **Nihai Durum:** `{final_status}`",
        "",
        "## 1. SES",
        f"- Üretilen paragraf sayısı: {audio_metrics.get('paragraph_count', 0)}",
        f"- Başarılı cümle sayısı: {audio_metrics.get('approved_sentences', 0)}",
        f"- Yeniden üretilen cümle sayısı: {audio_metrics.get('retried_sentences', 0)}",
        f"- Manuel inceleme sayısı: {audio_metrics.get('manual_review_count', 0)}",
        f"- Whisper eşleşme oranı: {audio_metrics.get('whisper_match_ratio', 0.0)}",
        f"- Eksik kelime kontrolleri: {audio_metrics.get('missing_word_checks', 'OK')}",
        f"- Ses seviyesi (RMS/LUFS): {audio_metrics.get('loudness_db', 0.0)} dB",
        f"- True Peak: {audio_metrics.get('true_peak_dbtp', 0.0)} dBTP",
        f"- Süre: {audio_metrics.get('duration_sec', 0.0)} sn",
        "",
        "## 2. GÖRSEL",
        f"- Toplam sahne sayısı: {visual_metrics.get('total_scenes', 0)}",
        f"- Başarılı görsel sayısı: {visual_metrics.get('approved_images', 0)}",
        f"- Yeniden üretilen görsel sayısı: {visual_metrics.get('regenerated_images', 0)}",
        f"- Boyut doğrulaması: {visual_metrics.get('resolution', '1920x1080')}",
        "",
        "## 3. VİDEO",
        f"- Üretilen klip sayısı: {video_metrics.get('generated_clips', 0)}",
        f"- Onaylanan klip sayısı: {video_metrics.get('approved_clips', 0)}",
        f"- Reddedilen klip sayısı: {video_metrics.get('rejected_clips', 0)}",
        f"- Manuel inceleme sayısı: {video_metrics.get('manual_review_count', 0)}",
        f"- Ortalama hareket: {video_metrics.get('mean_motion', 0.0)}",
        f"- Maksimum flicker: {video_metrics.get('max_flicker', 0.0)}",
        f"- Min keskinlik oranı: {video_metrics.get('min_sharpness_ratio', 0.0)}",
        f"- Döngü kontrolü: {video_metrics.get('loop_check', 'OK')}",
        "",
        "## 4. MONTAJ",
        f"- Toplam süre: {assembly_metrics.get('video_duration_sec', 0.0)} sn",
        f"- Çözünürlük: {assembly_metrics.get('width', 0)}x{assembly_metrics.get('height', 0)}",
        f"- FPS: {assembly_metrics.get('fps', 0.0)}",
        f"- Codec: {assembly_metrics.get('video_codec', '')} / {assembly_metrics.get('audio_codec', '')}",
        f"- Ses-video senkron farkı: {assembly_metrics.get('sync_diff_sec', 0.0)} sn",
        f"- Outro siyah ekran (#000000): {assembly_metrics.get('black_screen_verified', False)}",
        f"- Tam decode testi: {assembly_metrics.get('decode_ok', False)}",
        "",
        "## 5. ALTYAZI",
        f"- SRT satır/blok sayısı: {subtitle_metrics.get('cue_count', 0)}",
        f"- Eşleşme kapsamı: {subtitle_metrics.get('word_coverage_ratio', 0.0)}",
        f"- Zaman çakışmaları: {subtitle_metrics.get('overlap_count', 0)}",
        f"- Geçersiz satırlar: {subtitle_metrics.get('long_line_count', 0) + subtitle_metrics.get('too_many_lines_count', 0)}",
        f"- Siyah ekran ihlali: {subtitle_metrics.get('black_screen_violations', 0)}",
        "",
        "## 6. SON DURUM",
        f"**{final_status}**",
    ]
    if critical_blockers:
        md_lines.append("\n### Kritik Engelleyiciler")
        md_lines.extend([f"- {b}" for b in critical_blockers])
    if manual_review_items:
        md_lines.append("\n### Manuel İnceleme Kuyruğu")
        md_lines.extend([f"- {m}" for m in manual_review_items])

    atomic_write_text(out_dir / f"{project_id}_SON_KALITE_RAPORU.md", "\n".join(md_lines) + "\n")
    return report
