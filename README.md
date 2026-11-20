# Uyku Hikâyesi V3_PATCHED

Google Colab Pro (NVIDIA Tesla T4, Python 3.13) + Google Drive üzerinde çalışan, hata durumunda duran (fail-closed), kaldığı yerden devam edebilen (idempotent) 1080p / 24 fps Türkçe uyku hikâyesi video üretim hattı.

## Boru Hattı Aşamaları

1. **Seslendirme & Kalite Kontrol:** XTTS-v2 + Whisper Turbo (WER, CER, WPM, kelime hizalama, cümle önbelleği ve karantina).
2. **Ambiyans:** AudioLDM (`cvssp/audioldm-s-full-v2`) ile anlatıcı dostu düşük seviyeli ambiyans.
3. **Ses Mastering:** RMS / True Peak normalizasyon + kademeli outro fade-out.
4. **Görsel Üretim:** SDXL 1.0 FP16 (1920x1080, parlaklık & keskinlik doğrulamalı).
5. **Image-to-Video:** SVD-XT + OpenCV hareket analizi + CLIP anlamsal/negatif doğrulama.
6. **Altyazı:** Whisper Turbo zaman damgalı SRT.
7. **Montaj:** FFmpeg 1080p/24fps MP4 birleştirme ve ses-görüntü senkron doğrulaması.

## Mevcut Durum ve Çalışma Kuralları

Güncel canlı T4 durumu, açık kalan tek engel, kilitli hata düzeltmeleri ve koruma kuralları için **[`HANDOFF.md`](HANDOFF.md)** dosyasını oku.

## Yerel Testleri Çalıştırma

```bash
python3 -m unittest discover -s uyku_v3/tests -v
```
