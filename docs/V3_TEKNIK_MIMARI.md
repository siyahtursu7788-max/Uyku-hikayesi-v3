# UYKU_HIKAYESI_V3_FINAL_PRODUCTION — Teknik Mimari Belgesi

## 1. Genel Bakış
`UYKU_HIKAYESI_V3_FINAL_PRODUCTION`, 60–150 dakikalık (125+ paragraf) Türkçe uyku hikâyesi videolarını uçtan uca üreten, **fail-closed** durum makineli, **6 faktörlü SHA-256 önbellekli** ve **Google Colab Pro NVIDIA T4 (15 GB VRAM)** donanımına göre optimize edilmiş modüler üretim platformudur.

Bu sürüm, Drive'daki `GENEL_UYKU_VIDEOSU_BORU_HATTI_REHBERI.md` ve `UYKU_HIKAYESI_BORU_HATTI_ANAYASASI.md` standartları ile V1, V2, V2.1 ve V2.2'nin doğrulanmış bileşenlerini tek bir çatı altında toplar; tespit edilen tüm kök hataları (V2.2 `profile_success_stats` `passes: 0` hatası, eksik tekil SHA-256 koruma dökümü, eksik siyah/boş/tekrar kare OpenCV denetimi, 6 faktörlü önbellek eksikleri) giderir.

## 2. Modül Haritası (Korunanlar vs. Yeniden Yazılanlar/Eklenenler)

| Alt Sistem | Paket Yolu | Durum | Temel Yetenekler |
| :--- | :--- | :--- | :--- |
| Çekirdek Durum & Manifest | `uyku_v3/core/state_manager.py`, `manifest.py`, `file_manager.py`, `recovery.py` | **REUSE + UPGRADE** | 8 durumlu fail-closed makine (`PENDING..INTERRUPTED`), atomik JSON yazımı, kesinti kurtarma ve karantina |
| 6 Faktörlü SHA-256 Önbellek | `uyku_v3/core/v3_cache.py`, `audio/sentence_cache.py` | **REBUILD (YENİ)** | Metin + model + referans ses + parametre + eşik/config + kaynak dosya hash doğrulaması; yalnızca `APPROVED` kayıt |
| Eski Dosya Koruma Bekçisi | `uyku_v3/core/protection.py` | **REBUILD (YENİ)** | 15 korunan eski dosyanın tekil işlem öncesi/sonrası SHA-256 özetlerini rapora yazar |
| T4 Kaynak İzleyici | `uyku_v3/core/resource_monitor.py` | **REBUILD (YENİ)** | VRAM, RAM ve yerel SSD disk izleme + aşama arası `gc.collect()` / `cuda.empty_cache()` |
| Türkçe Normalizasyon & Sözlük | `uyku_v3/audio/text_normalizer.py`, `pronunciation_dictionary.py` | **REUSE** | Türkçe sayı/tarih açılımı, şapkalı kelime sözlüğü, virgül/nefes grubu akıllı bölme, kısa cümle gruplama |
| Türkçe XTTS-v2 + Whisper QC | `uyku_v3/audio/xtts_engine.py`, `whisper_validator.py`, `audio_quality.py` | **REUSE + FIX** | Doğrudan XTTS-v2 çıkarım, 3 aşamalı Whisper Turbo denetimi, son kelime/son hece kontrolü, kontrollü `atempo`, düzeltilmiş `profile_success_stats` (`needs_manual_review` alanı), özel isim otomatik onayı KAPALI (`MANUAL_REVIEW`) |
| Profesyonel Ses Mastering | `uyku_v3/audio/mastering.py` | **REUSE** | 70 Hz HPF, doğal vokal dengeleme, K1–K6 kademeli LUFS (-14 → -18 LUFS), -2.5 dBTP True Peak sınırlama, DC offset ve klik koruması, 160 sn saf karanlık kapanış |
| Görsel Üretim & Devamlılık | `uyku_v3/visuals/image_generator.py`, `image_validator.py`, `continuity_manager.py` | **REUSE + UPGRADE** | SDXL 1.0 FP16 (`attention_slicing` + `vae_tiling` + CPU offload) / Imagen 3, karakter/mekân/ışık/nesne stil rehberi, sahneler arası parlaklık/renk sapması denetimi |
| SVD-XT Image-to-Video | `uyku_v3/video/svd_engine.py` | **REUSE** | `stable-video-diffusion-img2vid-xt` FP16, `enable_model_cpu_offload()`, `decode_chunk_size=4`, 7 sahne hareket profili (`MOTION_PROFILES`), 3 başarısız denemede `MANUAL_REVIEW` |
| Ping-Pong & Video Döngü | `uyku_v3/video/loop_engine.py` | **REUSE** | Simetrik sahneler için `ping_pong`, yönlü hareketler (ateş/yağmur/dalga/buhar) için `forward_crossfade` veya `optical_blend` |
| CLIP & OpenCV Video QC | `uyku_v3/video/clip_validator.py`, `motion_validator.py`, `video_validator.py` | **REUSE + UPGRADE** | CLIP bağımsız kosinüs benzerliği (`margin >= 0.03`, softmax yasak); OpenCV flicker, morphing drift, kare sıçraması, keskinlik + yeni siyah/boş/tekrarlı kare denetimi |
| AudioLDM Ambiyans Motoru | `uyku_v3/ambience/audioldm_engine.py` | **REBUILD (YENİ)** | Yağmur, rüzgâr, deniz, dalga, şömine, orman, gece böcekleri, uzak gök gürültüsü, kütüphane, dere profilleri; yerel lisanslı ses desteği; kesintisiz crossfade döngü; anlatıcıyı bastırmayan seviye (`<= -24 dB`) |
| FFmpeg Video Kurgu & Senkron | `uyku_v3/assembly/ffmpeg_engine.py`, `sync_validator.py`, `timeline.py`, `final_export.py` | **REUSE** | Yerel SSD 1080p/24fps birleştirme (Drive FUSE exit 183 koruması), `-threads 2`, `-movflags +faststart`, saf siyah `#000000` doğrulaması, A/V senkronizasyon (`<= 0.35 sn`) |
| Altyazı (Whisper Turbo SRT) | `uyku_v3/subtitles/*` | **REUSE** | Kelime zaman damgalı Türkçe SRT, maks 42 karakter × 2 satır, sıfır çakışma, siyah ekranda 0 hayalet altyazı |
| Tek Ana Çalıştırıcı & 2 Hücre | `v3_runner.py`, `V3_Colab_T4_Uretim.ipynb`, `colab/*` | **REBUILD (YENİ)** | Tek komutla tüm aşamalar, kesintiden devam, tekil SHA-256 koruma raporu ve 2 hücreli Colab Pro T4 deneyimi |

## 3. Colab Pro T4 Donanım Optimizasyonu ve Gerçekçi Sınırlar
- Modeller **asla** aynı anda VRAM'de tutulmaz: Seslendirme (XTTS-v2 + Whisper Turbo ~5.4 GB tepe) → boşaltma → Görsel (SDXL FP16 CPU offload) → boşaltma → Video (SVD-XT FP16 `decode_chunk_size=4` + CPU offload) → boşaltma → AudioLDM → boşaltma → FFmpeg CPU yerel SSD kurgu.
- Tüm ağır FFmpeg `concat` ve `-movflags +faststart` işlemleri `/content/uyku_v3_work` yerel SSD üzerinde yapılır, doğrulandıktan sonra atomik olarak Drive'a kopyalanır.
- `requirements_v3_colab_py313.txt` Python 3.13 + Colab CUDA PyTorch kurulumunu ezmeden yalnızca eksik paketleri yükler.
