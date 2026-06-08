from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import sqlite3
import json
import os
import uvicorn
import base64
import numpy as np
import cv2
import easyocr

# ── OCR ─────────────────────────────────────────
ocr_reader = None

def ocr_yukle():
    global ocr_reader
    if ocr_reader is None:
        ocr_reader = easyocr.Reader(["tr", "en"], gpu=False)
    return ocr_reader

# ── LIFESPAN (başlangıçta OCR yükle) ────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("EasyOCR modeli yükleniyor...")
    ocr_yukle()
    print("EasyOCR hazır!")
    yield

app = FastAPI(title="Akıllı Depo Yönetim Sistemi API", lifespan=lifespan)

# ── CORS ────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ── Veritabanı ──────────────────────────────────
DB = "database.db"

def db_baglanti():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def db_olustur():
    conn = db_baglanti()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stok (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            urun      TEXT    NOT NULL,
            miktar    INTEGER DEFAULT 0,
            birim     TEXT    DEFAULT 'adet',
            renk      TEXT,
            guncelleme TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hareketler (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            urun      TEXT,
            miktar    INTEGER,
            renk      TEXT,
            zaman     TEXT DEFAULT (datetime('now', 'localtime'))
        )
    """)
    conn.commit()
    conn.close()

# ── Modeller ────────────────────────────────────
class StokGuncelle(BaseModel):
    renk:  str
    sayi:  int

class UrunEkle(BaseModel):
    urun:   str
    miktar: int
    birim:  Optional[str] = "adet"
    renk:   Optional[str] = None

class RenkGuncelle(BaseModel):
    urun:   str
    carpan: int
    birim:  str

class GoruntuData(BaseModel):
    goruntu: str

# ── Yardımcı ────────────────────────────────────
def renkleri_yukle():
    with open("colors.json", "r", encoding="utf-8") as f:
        return json.load(f)

def renkleri_kaydet(data):
    with open("colors.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ── STOK ENDPOINTLERİ ───────────────────────────
@app.get("/stok")
def stok_listesi():
    conn = db_baglanti()
    rows = conn.execute("SELECT * FROM stok ORDER BY urun").fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/stok/ekle")
def stok_ekle(veri: UrunEkle):
    conn = db_baglanti()
    mevcut = conn.execute("SELECT * FROM stok WHERE urun = ?", (veri.urun,)).fetchone()
    if mevcut:
        conn.execute(
            "UPDATE stok SET miktar = miktar + ?, guncelleme = datetime('now','localtime') WHERE urun = ?",
            (veri.miktar, veri.urun)
        )
    else:
        conn.execute(
            "INSERT INTO stok (urun, miktar, birim, renk, guncelleme) VALUES (?, ?, ?, ?, datetime('now','localtime'))",
            (veri.urun, veri.miktar, veri.birim, veri.renk)
        )
    conn.commit()
    conn.close()
    return {"durum": "ok", "urun": veri.urun, "eklenen": veri.miktar}

@app.post("/stok/kamera")
def kamera_tespiti(veri: StokGuncelle):
    renkler = renkleri_yukle()
    if veri.renk not in renkler:
        raise HTTPException(status_code=400, detail=f"Bilinmeyen renk: {veri.renk}")
    bilgi  = renkler[veri.renk]
    miktar = veri.sayi * bilgi["carpan"]
    conn = db_baglanti()
    mevcut = conn.execute("SELECT * FROM stok WHERE urun = ?", (bilgi["urun"],)).fetchone()
    if mevcut:
        conn.execute(
            "UPDATE stok SET miktar = miktar + ?, renk = ?, guncelleme = datetime('now','localtime') WHERE urun = ?",
            (miktar, veri.renk, bilgi["urun"])
        )
    else:
        conn.execute(
            "INSERT INTO stok (urun, miktar, birim, renk, guncelleme) VALUES (?, ?, ?, ?, datetime('now','localtime'))",
            (bilgi["urun"], miktar, bilgi["birim"], veri.renk)
        )
    conn.execute(
        "INSERT INTO hareketler (urun, miktar, renk) VALUES (?, ?, ?)",
        (bilgi["urun"], miktar, veri.renk)
    )
    conn.commit()
    conn.close()
    return {"durum": "ok", "urun": bilgi["urun"], "miktar": miktar, "birim": bilgi["birim"]}

@app.delete("/stok/{urun_id}")
def stok_sil(urun_id: int):
    conn = db_baglanti()
    conn.execute("DELETE FROM stok WHERE id = ?", (urun_id,))
    conn.commit()
    conn.close()
    return {"durum": "silindi"}

@app.get("/hareketler")
def hareket_listesi():
    conn = db_baglanti()
    rows = conn.execute("SELECT * FROM hareketler ORDER BY zaman DESC LIMIT 50").fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ── RENK YÖNETİMİ ───────────────────────────────
@app.get("/renkler")
def renk_listesi():
    return renkleri_yukle()

@app.put("/renkler/{renk}")
def renk_guncelle(renk: str, veri: RenkGuncelle):
    renkler = renkleri_yukle()
    renkler[renk] = {"urun": veri.urun, "carpan": veri.carpan, "birim": veri.birim}
    renkleri_kaydet(renkler)
    return {"durum": "ok", "renk": renk}

@app.delete("/renkler/{renk}")
def renk_sil(renk: str):
    renkler = renkleri_yukle()
    if renk not in renkler:
        raise HTTPException(status_code=404, detail="Renk bulunamadı")
    del renkler[renk]
    renkleri_kaydet(renkler)
    return {"durum": "silindi"}

# ── OCR ENDPOINTİ ───────────────────────────────
@app.post("/ocr")
async def goruntu_ocr(veri: GoruntuData):
    try:
        img_data = base64.b64decode(veri.goruntu)
        np_arr   = np.frombuffer(img_data, np.uint8)
        frame    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        gri      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gri      = cv2.resize(gri, (640, 480))
        gri      = cv2.equalizeHist(gri)
        reader   = ocr_yukle()
        sonuclar = reader.readtext(gri, allowlist="0123456789")
        sayi = None
        for (_, metin, guven) in sonuclar:
            if guven > 0.4 and metin.isdigit():
                sayi = int(metin)
                break
        return {"sayi": sayi}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── BAŞLAT ──────────────────────────────────────
if __name__ == "__main__":
    db_olustur()
    print("✅ Veritabanı hazır")
    print("✅ API başlatılıyor → http://localhost:8000")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)