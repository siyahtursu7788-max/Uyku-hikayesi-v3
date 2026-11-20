# ==============================================================================
# UYKU_HIKAYESI_V3_PATCHED — COLAB PRO T4 HÜCRE 1: KURULUM, ZIP ÇIKARMA VE DOĞRULAMA
# ==============================================================================
import hashlib, importlib.util, shutil, subprocess, sys, tempfile, zipfile
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive", force_remount=True)

V3_DRIVE_ROOT = Path("/content/drive/MyDrive/V3_PATCHED")
ZIP_PATH = V3_DRIVE_ROOT / "UYKU_HIKAYESI_V3_PATCHED.zip"
EXPECTED_ZIP_SHA256 = "8b249b784fa68a2b05239efb63e0687665a906e32262696916c46007d4543d72"
LOCAL_EXTRACT_ROOT = Path("/content/uyku_v3_patched_pkg")
PKG_ROOT = LOCAL_EXTRACT_ROOT / "V3_PATCHED"
MARKER_FILE = LOCAL_EXTRACT_ROOT / ".v3_patched_zip_sha256"

assert ZIP_PATH.exists(), f"V3_PATCHED ZIP bulunamadı: {ZIP_PATH}"
actual_sha = hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest()
if EXPECTED_ZIP_SHA256 != "__SYNCED_ON_BUILD__":
    assert actual_sha == EXPECTED_ZIP_SHA256, (
        f"ZIP SHA-256 uyuşmuyor! beklenen={EXPECTED_ZIP_SHA256}, bulunan={actual_sha}"
    )
print("OK - ZIP SHA-256 doğrulandı:", actual_sha, flush=True)

# 1. Idempotent ve atomik ZIP çıkarma (tekrar çalıştırmada gereksiz çıkarma yapmaz)
cached_sha = MARKER_FILE.read_text(encoding="utf-8").strip() if MARKER_FILE.exists() else ""
if (PKG_ROOT / "uyku_v3" / "__init__.py").exists() and cached_sha == actual_sha:
    print("OK - Paket zaten güncel olarak çıkarılmış, tekrar çıkarılmadı:", PKG_ROOT, flush=True)
else:
    LOCAL_EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=".extract_v3_", dir=str(LOCAL_EXTRACT_ROOT)))
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(tmp_dir)
        extracted = tmp_dir / "V3_PATCHED"
        assert (extracted / "uyku_v3" / "__init__.py").exists(), "ZIP içinde V3_PATCHED/uyku_v3 bulunamadı!"
        if PKG_ROOT.exists():
            shutil.rmtree(PKG_ROOT)
        shutil.move(str(extracted), str(PKG_ROOT))
        MARKER_FILE.write_text(actual_sha, encoding="utf-8")
        print("OK - Paket yerel SSD'ye çıkarıldı:", PKG_ROOT, flush=True)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

# 2. Python import yolu doğrulaması
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))
import uyku_v3
print("OK - uyku_v3 import edildi:", uyku_v3.__file__, flush=True)

# 3. Eksik paketleri Colab CUDA PyTorch'u EZMEDEN kur
check_mods = [
    ("whisper", "openai-whisper"),
    ("TTS", "coqui-tts"),
    ("pyloudnorm", "pyloudnorm"),
    ("pydub", "pydub"),
    ("soundfile", "soundfile"),
    ("diffusers", "diffusers"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("psutil", "psutil"),
]
missing = [pkg for mod, pkg in check_mods if importlib.util.find_spec(mod) is None]
if missing:
    print("Eksik paketler kuruluyor:", missing, flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade-strategy", "only-if-needed", *missing])
else:
    print("Tüm zorunlu paketler zaten kurulu.", flush=True)

# 4. Gerçek NVIDIA T4 GPU kontrolü
import torch
has_cuda = bool(getattr(torch.cuda, "is_available")())
assert has_cuda, "HATA: CUDA GPU aktif değil! Çalışma Zamanı -> T4 GPU seçin."
gpu_name = torch.cuda.get_device_name(0)
assert "T4" in gpu_name.upper(), f"UYARI: Beklenen NVIDIA T4 yerine '{gpu_name}' algılandı!"
print("OK - GPU:", gpu_name, "| Python:", sys.version.split()[0], flush=True)

# 5. Kuru çalıştırma (--dry-run) ile senaryo, sahne planı ve eski dosya koruması testi
RUNNER = PKG_ROOT / "v3_runner.py"
rc = subprocess.run([sys.executable, str(RUNNER), "--dry-run"], check=False).returncode
assert rc == 0, f"Dry-run başarısız (kod={rc})"
print("HAZIR: V3_PATCHED kurulum, ZIP doğrulama ve dry-run başarıyla tamamlandı.", flush=True)
