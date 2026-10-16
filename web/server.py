"""AnemiaLens Web: server lokal yang menyajikan halaman web + API model.

Jalankan (dari folder AnemiaLens):
    python web/server.py                      # buka http://localhost:8000
    python web/server.py --model hasil_model_v2/model --port 8000
    python web/server.py --no-browser         # mis. di Colab / server

Struktur folder yang diharapkan:
    AnemiaLens/
      anemialens/            <- paket model (kode training & inferensi)
      hasil_model_v2/model/  <- hasil training (seg_unet.pt, cnn.pt, *.joblib, thresholds.json, ...)
      web/server.py          <- file ini
      web/static/            <- tampilan web (HTML/CSS/JS)
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

import warnings

import cv2
import numpy as np

try:   # model dilatih di scikit-learn 1.6.1; versi lebih baru memberi hasil identik (sudah diuji)
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

WEB = Path(__file__).resolve().parent
ROOT = WEB.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

MODEL_CANDIDATES = [
    os.environ.get("ANEMIALENS_MODEL", ""),
    ROOT / "hasil_model_v2" / "model",
    ROOT / "model",
    Path("/content/work/model"),
    Path("/content/AnemiaLens_model/model"),
]
MAX_SIDE = 2000          # foto HP 12 MP diperkecil dulu: lebih cepat, hasil setara
MAX_UPLOAD_MB = 25

app = FastAPI(title="AnemiaLens", docs_url="/api/docs")
STATE = {"lens": None, "error": None, "model_dir": None, "info": {}}
LOCK = threading.Lock()


# ---------------------------------------------------------------- utilitas gambar
def decode_rgb(data: bytes) -> np.ndarray:
    """Sama dengan common.read_rgb tetapi dari bytes: rotasi EXIF diterapkan,
    PNG transparan dijadikan latar hitam. HEIC (iPhone) didukung bila pillow-heif terpasang."""
    buf = np.frombuffer(data, dtype=np.uint8)
    raw = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if raw is not None and raw.ndim == 3 and raw.shape[2] == 4:
        if raw.dtype != np.uint8:
            raw = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        a = raw[:, :, 3:4].astype(np.float32) / 255.0
        bgr = (raw[:, :, :3].astype(np.float32) * a).astype(np.uint8)
    elif raw is not None:
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    else:
        bgr = None
    if bgr is None:                     # coba HEIC/format lain lewat Pillow
        try:
            import io

            from PIL import Image, ImageOps
            try:
                import pillow_heif
                pillow_heif.register_heif_opener()
            except ImportError:
                pass
            im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
            return np.asarray(im)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, "Format gambar tidak dikenali. Gunakan JPG atau PNG.") from e
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def shrink(rgb, max_side=MAX_SIDE):
    h, w = rgb.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return rgb
    return cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def to_data_url(rgb, max_side=900, quality=85):
    img = shrink(rgb, max_side)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode() if ok else None


def apply_crop(rgb, crop):
    """crop = [x, y, w, h] relatif 0..1 terhadap foto (dari alat crop di web)."""
    if not crop:
        return rgb, False
    x, y, w, h = [float(v) for v in crop]
    H, W = rgb.shape[:2]
    x0, y0 = int(np.clip(x, 0, 1) * W), int(np.clip(y, 0, 1) * H)
    x1, y1 = int(np.clip(x + w, 0, 1) * W), int(np.clip(y + h, 0, 1) * H)
    if x1 - x0 < 32 or y1 - y0 < 32:
        return rgb, False
    return rgb[y0:y1, x0:x1].copy(), True


# ---------------------------------------------------------------- model
def find_model_dir(arg):
    for c in ([arg] if arg else []) + MODEL_CANDIDATES:
        if c and (Path(c) / "thresholds.json").exists():
            return Path(c).resolve()
    return None


def read_info(model_dir: Path):
    thr = json.loads((model_dir / "thresholds.json").read_text())
    info = {"models": thr.get("models", []), "auc": thr.get("auc_ensemble"), "auc_ci95": thr.get("auc_ci95"),
            "hb_mae": thr.get("hb_mae"), "thr_sensitive": thr.get("thr_sensitive"),
            "thr_specific": thr.get("thr_specific"), "target_sensitivity": thr.get("target_sensitivity")}
    seg = model_dir / "seg_metrics.txt"
    if seg.exists():
        kv = dict(l.split("=", 1) for l in seg.read_text().split() if "=" in l)
        info.update(seg_dice=float(kv.get("dice_val", "nan")),
                    seg_dice_hp=float(kv["dice_val_hp"]) if "dice_val_hp" in kv else None,
                    seg_version="v2 (robust: foto HP & konjungtiva pucat)" if kv.get("robust") == "1" else "v1")
    for rep in (model_dir / "report.md", model_dir.parent / "report.md"):
        if rep.exists():
            t = rep.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"Data:\s*(\d+) foto,\s*(\d+) pasien,\s*prevalensi anemia ([\d.]+)%", t)
            if m:
                info.update(n_photos=int(m[1]), n_patients=int(m[2]), prevalence=float(m[3]))
            ens = re.search(r"^\| ensemble\s*\|([^\n]+)", t, re.M)
            if ens:
                cells = [c.strip() for c in ens[1].split("|")]
                try:   # kolom: n, anemia, auc, sens, spec, ...
                    info.update(sensitivity=float(cells[3]), specificity=float(cells[4]))
                except (ValueError, IndexError):
                    pass
            info["n_sources"] = len(set(re.findall(r"^\| (cp_anemic|eyes_defy_anemia/\w+)", t, re.M)))
            break
    return info


def load_model(model_dir):
    try:
        t0 = time.time()
        from anemialens.predict import AnemiaLens
        lens = AnemiaLens(model_dir)
        # pemanasan supaya foto pertama tidak lambat
        dummy = np.full((480, 640, 3), (190, 120, 120), np.uint8)
        try:
            lens.analyze(dummy)
        except Exception:  # noqa: BLE001
            pass
        STATE["lens"] = lens
        print(f"[AnemiaLens] Model siap ({time.time() - t0:.1f} dtk) dari {model_dir}")
    except Exception as e:  # noqa: BLE001
        STATE["error"] = f"{type(e).__name__}: {e}"
        print("[AnemiaLens] GAGAL memuat model:", STATE["error"])


# ---------------------------------------------------------------- API
@app.get("/api/info")
def api_info():
    return {"ready": STATE["lens"] is not None, "error": STATE["error"],
            "model_dir": str(STATE["model_dir"]) if STATE["model_dir"] else None, **STATE["info"]}


def polygon_mask(shape, poly):
    """poly = [[x, y], ...] relatif 0..1 (dari alat lingkari di web) -> mask biner ukuran foto."""
    H, W = shape[:2]
    pts = np.array([[float(x) * W, float(y) * H] for x, y in poly], dtype=np.float32)
    pts = np.clip(pts, 0, [W - 1, H - 1]).round().astype(np.int32)
    m = np.zeros((H, W), np.uint8)
    if len(pts) >= 3:
        cv2.fillPoly(m, [pts], 1)
    return m


@app.post("/api/analyze")
async def api_analyze(file: UploadFile = File(...), use_card: bool = Form(False), crop: str = Form(""),
                      polygon: str = Form(""), manual_crop: bool = Form(False), force: bool = Form(False)):
    """Tiga mode:
    - otomatis: U-Net mencari konjungtiva;
    - polygon: pengguna melingkari konjungtiva -> area itu yang dinilai (paling andal untuk mata pucat);
    - crop: kotak untuk memperbesar area sebelum deteksi otomatis."""
    lens = STATE["lens"]
    if lens is None:
        raise HTTPException(503, STATE["error"] or "Model masih dimuat, coba lagi beberapa detik.")
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Foto terlalu besar (maks {MAX_UPLOAD_MB} MB).")
    t0 = time.time()
    rgb = shrink(decode_rgb(data))
    user_mask, cropped = None, False
    poly = json.loads(polygon) if polygon else None
    if poly and len(poly) >= 3:
        m = polygon_mask(rgb.shape, poly)
        if m.sum() >= 50:
            # foto utuh tetap dipakai (cek blur/cahaya butuh konteks); lingkaran = mask konjungtiva
            user_mask, cropped = m, True
    if user_mask is None:
        rgb, cropped = apply_crop(rgb, json.loads(crop) if crop else None)
    with LOCK:
        if user_mask is not None:
            r = lens.analyze(rgb, use_card=use_card, force=force, mask=user_mask, manual_crop=True)
        else:
            r = lens.analyze(rgb, use_card=use_card, force=force, manual_crop=manual_crop and cropped)
    overlay = r.pop("overlay", None)
    q = r.get("quality", {})
    out = {k: v for k, v in r.items() if k != "quality"}
    out.update(
        overlay=to_data_url(overlay) if overlay is not None else None,
        quality={k: (float(v) if isinstance(v, (np.floating, float, int)) else v) for k, v in q.items()},
        cropped=cropped, user_outline=user_mask is not None, latency_ms=int((time.time() - t0) * 1000),
        thr_sensitive=lens.thr["thr_sensitive"], thr_specific=lens.thr["thr_specific"],
    )
    return JSONResponse(json.loads(json.dumps(out, default=lambda o: o.item() if hasattr(o, "item") else str(o))))


@app.get("/")
def index():
    return FileResponse(WEB / "static" / "index.html")


app.mount("/", StaticFiles(directory=WEB / "static"), name="static")


# ---------------------------------------------------------------- main
def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="folder model (berisi thresholds.json)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    md = find_model_dir(a.model)
    if md is None:
        STATE["error"] = ("Folder model tidak ditemukan. Ekstrak AnemiaLens_model.zip sehingga ada "
                          "AnemiaLens/hasil_model_v2/model/thresholds.json, atau pakai --model <folder>.")
        print("[AnemiaLens]", STATE["error"])
    else:
        STATE["model_dir"], STATE["info"] = md, read_info(md)
        threading.Thread(target=load_model, args=(md,), daemon=True).start()
    import uvicorn
    print("\n" + "=" * 60)
    print(f"  AnemiaLens Web berjalan di  http://localhost:{a.port}")
    ip = lan_ip()
    if ip:
        print(f"  Dari HP (WiFi yang sama):   http://{ip}:{a.port}")
    print("  Tekan Ctrl+C untuk berhenti.")
    print("=" * 60 + "\n")
    if not a.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{a.port}")).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
