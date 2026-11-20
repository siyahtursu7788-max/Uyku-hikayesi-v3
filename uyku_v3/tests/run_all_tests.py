"""
Tüm otomatik testleri çalıştırır ve Bölüm 15 standartlarına uygun olarak
her test için beklenen sonuç, gerçek sonuç, başarı/başarısızlık, çalışma türü
(REAL_LOCAL / SIMULATED_GPU / E2E_INTEGRATION) ve hata açıklamasını JSON + Markdown
raporu olarak kaydeder.
"""

from __future__ import annotations

import json
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class StructuredTestResult(unittest.TestResult):
    def __init__(self) -> None:
        super().__init__()
        self.records = []

    def _classify_mode(self, doc: str) -> str:
        if "[E2E_INTEGRATION" in doc:
            return "E2E_INTEGRATION (REAL_LOCAL_DSP_FFMPEG + SIMULATED_GPU_MODELS)"
        if "[SIMULATED_GPU+REAL_LOCAL]" in doc:
            return "HYBRID (SIMULATED_GPU + REAL_LOCAL)"
        if "[SIMULATED_GPU]" in doc:
            return "SIMULATED_GPU"
        return "REAL_LOCAL"

    def addSuccess(self, test: unittest.TestCase) -> None:
        super().addSuccess(test)
        doc = (test._testMethodDoc or "").strip()
        self.records.append({
            "test_id": test.id(),
            "mode": self._classify_mode(doc),
            "expected": doc or "Test koşulları sağlanmalı",
            "actual": "Beklenen tüm doğrulamalar geçti",
            "status": "PASS",
            "error": "",
        })

    def addFailure(self, test: unittest.TestCase, err) -> None:
        super().addFailure(test, err)
        doc = (test._testMethodDoc or "").strip()
        self.records.append({
            "test_id": test.id(),
            "mode": self._classify_mode(doc),
            "expected": doc or "Test koşulları sağlanmalı",
            "actual": "Doğrulama başarısız",
            "status": "FAIL",
            "error": self._exc_info_to_string(err, test),
        })

    def addError(self, test: unittest.TestCase, err) -> None:
        super().addError(test, err)
        doc = (test._testMethodDoc or "").strip()
        self.records.append({
            "test_id": test.id(),
            "mode": self._classify_mode(doc),
            "expected": doc or "Test koşulları sağlanmalı",
            "actual": "Çalışma zamanı hatası",
            "status": "ERROR",
            "error": self._exc_info_to_string(err, test),
        })


def main() -> int:
    start = time.time()
    loader = unittest.TestLoader()
    suite = loader.discover(str(Path(__file__).resolve().parent), pattern="test_*.py")
    res = StructuredTestResult()
    suite.run(res)
    elapsed = round(time.time() - start, 2)

    passed = sum(1 for r in res.records if r["status"] == "PASS")
    failed = sum(1 for r in res.records if r["status"] != "PASS")
    summary = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_tests": len(res.records),
        "passed": passed,
        "failed": failed,
        "elapsed_sec": elapsed,
        "environment_note": (
            "Yerel Linux CPU ortamında çalıştırıldı (Python 3.11, FFmpeg, FFprobe, OpenCV, NumPy, SciPy, PIL). "
            "T4 GPU gerektiren yapay zekâ modelleri (XTTS-v2, Whisper Turbo, SDXL 1.0, SVD-XT, CLIP) "
            "SIMULATED_GPU olarak enjekte edilmiş; tüm ses DSP, normalizasyon, kalite kontrol, OpenCV, "
            "FFmpeg montaj/decode, SRT ve kurtarma motorları GERÇEK yerel ortamda çalıştırılmıştır."
        ),
        "results": res.records,
    }

    out_dir = Path(__file__).resolve().parent.parent / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "TEST_SONUCLARI_RAPORU.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Otomatik Test ve Kabul Kriterleri Raporu — uyku_v3",
        f"- **Çalıştırma Tarihi (UTC):** {summary['executed_at_utc']}",
        f"- **Toplam Test:** {summary['total_tests']} | **Başarılı:** {passed} | **Başarısız:** {failed} | **Süre:** {elapsed} sn",
        f"- **Ortam Notu:** {summary['environment_note']}",
        "",
        "| # | Test | Çalışma Türü | Durum | Beklenen / Açıklama |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(res.records, 1):
        short_name = r["test_id"].split(".")[-1]
        exp_clean = r["expected"].replace("\n", " ").replace("|", "/")
        md_lines.append(f"| {i} | `{short_name}` | `{r['mode']}` | **{r['status']}** | {exp_clean} |")

    if failed > 0:
        md_lines.append("\n## Hata Detayları")
        for r in res.records:
            if r["status"] != "PASS":
                md_lines.append(f"### {r['test_id']}\n```\n{r['error']}\n```")

    md_path = out_dir / "TEST_SONUCLARI_RAPORU.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"TEST SONUCU: {passed}/{len(res.records)} BAŞARILI, {failed} BAŞARISIZ ({elapsed} sn)")
    if failed > 0:
        for r in res.records:
            if r["status"] != "PASS":
                print(f"FAIL: {r['test_id']}\n{r['error']}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
