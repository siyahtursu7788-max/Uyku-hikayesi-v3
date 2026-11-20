# UYKU_HIKAYESI_V3_FINAL_PRODUCTION — Kullanım Kılavuzu (Google Colab Pro T4)

## 1. İki Hücreli Standart Kullanım
Google Drive içindeki `UYKU_HIKAYESI_V3_FINAL_PRODUCTION/V3_Colab_T4_Uretim.ipynb` defterini Google Colab'da açın ve **Çalışma Zamanı → Çalışma zamanı türünü değiştir → T4 GPU** seçeneğini doğrulayın.

### Hücre 1 — Kurulum, Ortam ve Koruma Doğrulaması (`colab/cell_1_kurulum_ve_hazirlik.py`)
- Google Drive'ı bağlar.
- Colab CUDA PyTorch kurulumunu ezmeden yalnızca eksik paketleri kurar.
- T4 GPU'yu doğrular.
- `v3_runner.py --dry-run` çalıştırarak senaryo ayrıştırmasını, manifesti ve eski dosya SHA-256 korumasını üretim başlatmadan test eder.

### Hücre 2 — Üretimi Başlatma veya Kaldığı Yerden Devam (`colab/cell_2_uretim_baslat_veya_devam.py`)
- Varsayılan olarak `Unutulmus_Zamanlar_Arsivi_Final_v2.txt` senaryosunu işler (farklı bir hikâye için `SENARYO_YOLU` değişkenini değiştirmeniz yeterlidir).
- İlk canlı T4 doğrulaması için `MAX_PARAGRAF = 10` ve `STOP_AFTER = 3` ile kesinti/devam testi yapabilir, ardından `STOP_AFTER = 0` ve `MAX_PARAGRAF = 0` ile 125 paragrafın tamamını çalıştırabilirsiniz.
- Oturum kesilirse aynı Hücre 2'yi tekrar çalıştırmanız yeterlidir: sistem `APPROVED` çıktıları SHA-256 ile doğrulayarak atlar, yarım kalanları karantinaya alır ve kaldığı yerden devam eder.

## 2. Komut Satırı (CLI) Kullanımı
```bash
# 1. Kuru doğrulama (GPU olmadan senaryo + koruma testi)
python3 v3_runner.py --dry-run

# 2. İlk 10 paragrafta kesinti testi (3 paragraf sonra dur)
python3 v3_runner.py --max-paragraphs 10 --stop-after 3

# 3. Kaldığı yerden tüm senaryoyu tamamla
python3 v3_runner.py --stages all
```

## 3. Çıktı Klasörleri (`UYKU_HIKAYESI_V3_FINAL_PRODUCTION/uretim_ciktilari/`)
- `v3_production_manifest.json`: Tek doğruluk kaynağı olan atomik üretim manifesti.
- `paragraflar/`: Yalnızca `APPROVED` paragraf WAV dosyaları.
- `cumle_onbellek/` ve `v3_6factor_cache/`: SHA-256 doğrulamalı kalıcı önbellekler.
- `karantina/`: Başarısız veya yarım kalmış ses/video kayıtları (asla final klasörüne taşınmaz).
- `raporlar/`: Her çalıştırmanın tekil SHA-256 koruma dökümlü JSON raporu.
- `final/`: Doğrulanmış 1080p MP4, MASTER WAV/MP3, SRT ve YouTube bölümleri.
