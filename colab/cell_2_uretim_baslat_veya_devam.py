# ==============================================================================
# UYKU_HIKAYESI_V3_PATCHED — COLAB PRO T4 HÜCRE 2: BÜTÜNLEŞİK ÜRETİM / DEVAM
# ==============================================================================
import subprocess, sys
from pathlib import Path

PKG_ROOT = Path("/content/uyku_v3_patched_pkg/V3_PATCHED")
RUNNER = PKG_ROOT / "v3_runner.py"
ZIP_PATH = Path("/content/drive/MyDrive/V3_PATCHED/UYKU_HIKAYESI_V3_PATCHED.zip")

# Hücre 1'den sonra oturum yenilendiyse v3_runner ZIP'i otomatik bulup çıkarır
if not RUNNER.exists():
    RUNNER = Path("/content/drive/MyDrive/V3_PATCHED/v3_runner.py")
assert RUNNER.exists(), f"v3_runner.py bulunamadı: {RUNNER}"

# Ayarlar: İlk canlı T4 uçtan uca kısa hikâye doğrulaması için MAX_PARAGRAF = 3 önerilir;
# tam 125 paragraf üretimi için MAX_PARAGRAF = 0 yapın.
SENARYO_YOLU = None
MAX_PARAGRAF = 3
STOP_AFTER = 0       # Kesinti/devam testi için 2 yapıp sonra 0 ile devam ettirebilirsiniz
STAGES = "all"       # all = XTTS -> Whisper -> AudioLDM -> Mastering -> SDXL -> SVD-XT/CLIP/OpenCV -> SRT -> FFmpeg MP4

cmd = [sys.executable, str(RUNNER), "--zip-path", str(ZIP_PATH), "--stages", STAGES]
if SENARYO_YOLU:
    cmd += ["--script", SENARYO_YOLU]
if MAX_PARAGRAF > 0:
    cmd += ["--max-paragraphs", str(MAX_PARAGRAF)]
if STOP_AFTER > 0:
    cmd += ["--stop-after", str(STOP_AFTER)]

print("Çalıştırılıyor:", " ".join(cmd), flush=True)
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
assert proc.stdout is not None
for line in proc.stdout:
    print(line, end="", flush=True)
rc = proc.wait()
print("V3_PATCHED ÇIKIŞ KODU:", rc, flush=True)
