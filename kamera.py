import cv2
import easyocr
import numpy as np
import json
import time
import threading
import requests

# ── Renk tanımlarını yükle ──────────────────────
def renkleri_yukle():
    with open("colors.json", "r", encoding="utf-8") as f:
        return json.load(f)

# ── HSV renk aralıkları ─────────────────────────
RENK_ARALIKLARI = {
    "kirmizi":    ([0,   120,  70], [10,  255, 255]),
    "turuncu":    ([11,  120,  70], [25,  255, 255]),
    "sari":       ([26,  120,  70], [35,  255, 255]),
    "yesil":      ([36,   80,  50], [85,  255, 255]),
    "acik_yesil": ([86,   80,  50], [95,  255, 255]),
    "mavi":       ([96,  120,  70], [130, 255, 255]),
    "mor":        ([131,  80,  50], [155, 255, 255]),
    "pembe":      ([156,  80,  50], [170, 255, 255]),
    "beyaz":      ([0,    0,  200], [180,  30, 255]),
    "siyah":      ([0,    0,    0], [180, 255,  50]),
}

# ── Paylaşılan state ────────────────────────────
son_tespit   = None
analiz_devam = False
kilitli      = threading.Lock()

# ── Renk tespiti ────────────────────────────────
def renk_tespit(frame):
    hsv        = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    maks_alan  = 0
    tespit     = None

    for renk, (alt, ust) in RENK_ARALIKLARI.items():
        maske = cv2.inRange(hsv, np.array(alt), np.array(ust))
        alan  = cv2.countNonZero(maske)
        if alan > maks_alan:
            maks_alan = alan
            tespit    = renk

    toplam = frame.shape[0] * frame.shape[1]
    return tespit if maks_alan > toplam * 0.10 else None

# ── OCR (arka plan thread'i) ────────────────────
def ocr_thread(frame, reader):
    global son_tespit, analiz_devam

    try:
        renkler = renkleri_yukle()
        renk    = renk_tespit(frame)
        sayi    = None

        if renk:
            gri     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sonuclar = reader.readtext(gri, allowlist="0123456789")
            for (_, metin, guven) in sonuclar:
                if guven > 0.5 and metin.isdigit():
                    sayi = int(metin)
                    break

        if renk and sayi and renk in renkler:
            bilgi  = renkler[renk]
            miktar = sayi * bilgi["carpan"]
            yeni   = {
                "renk":   renk,
                "urun":   bilgi["urun"],
                "sayi":   sayi,
                "miktar": miktar,
                "birim":  bilgi["birim"],
            }
            with kilitli:
                son_tespit = yeni
            print(f"✅ TESPİT: {renk.upper()} + {sayi}")
            print(f"   Ürün  : {bilgi['urun']}")
            print(f"   Miktar: {miktar} {bilgi['birim']}")
            print("-" * 40)

            # API'ye gönder
            try:
                r = requests.post(
                    "http://localhost:8000/stok/kamera",
                    json={"renk": renk, "sayi": sayi},
                    timeout=3
                )
                if r.status_code == 200:
                    print(f"   📡 Stok güncellendi!")
                else:
                    print(f"   ⚠️ API hatası: {r.status_code}")
            except Exception as e:
                print(f"   ❌ API bağlantı hatası: {e}")

    finally:
        analiz_devam = False

# ── Ana döngü ───────────────────────────────────
def main():
    global analiz_devam

    print("EasyOCR yükleniyor, bir dakika...")
    reader = easyocr.Reader(["tr", "en"], gpu=False)
    print("Hazır! Kamera açılıyor...")

    kamera = cv2.VideoCapture(0)
    kamera.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    kamera.set(cv2.CAP_PROP_FPS, 30)

    if not kamera.isOpened():
        print("HATA: Kamera açılamadı!")
        return

    print("Kamera hazır. Çıkmak için 'q' tuşuna bas.")
    print("-" * 40)

    son_analiz = 0

    while True:
        ret, frame = kamera.read()
        if not ret:
            break

        simdi = time.time()

        # Her 2 saniyede bir, thread boştaysa yeni analiz başlat
        if simdi - son_analiz > 2.0 and not analiz_devam:
            analiz_devam = True
            son_analiz   = simdi
            t = threading.Thread(target=ocr_thread, args=(frame.copy(), reader))
            t.daemon = True
            t.start()

        # ── Ekran çizimi ──
        goster = frame.copy()

        with kilitli:
            tespit = son_tespit

        if tespit:
            etiket = f"{tespit['urun']}: {tespit['miktar']} {tespit['birim']}"
            cv2.rectangle(goster, (5, 5), (450, 55), (0, 0, 0), -1)
            cv2.putText(goster, etiket, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Alt bilgi
        durum = "Analiz ediliyor..." if analiz_devam else "Hazir"
        cv2.putText(goster, f"Durum: {durum}",
                    (10, goster.shape[0] - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(goster, "Koli etiketi gosterin | Q: Cikis",
                    (10, goster.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Akilli Depo - Kamera", goster)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    kamera.release()
    cv2.destroyAllWindows()
    print("Kamera kapatıldı.")

if __name__ == "__main__":
    main()