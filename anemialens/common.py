"""AnemiaLens - utilitas bersama: konfigurasi, I/O gambar, ambang anemia WHO,
pemeriksaan kualitas foto, pemotongan ROI konjungtiva, dan fitur warna."""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# Kata kunci nama file yang menandakan mask segmentasi, bukan foto asli
MASK_KEYS = ("mask", "palpebral", "palplebral", "papebral", "palpebrale", "forniceal", "fornice", "segment", "_seg", "label")


@dataclass
class Config:
    work_dir: str = "/content/work"          # tempat semua keluaran
    seed: int = 42
    n_folds: int = 5
    img_size: int = 224                      # input classifier
    seg_size: int = 384                      # input segmentasi
    cnn_backbone: str = "efficientnet_b0"    # alternatif: convnext_tiny.fb_in22k_ft_in1k
    dino_backbone: str = "vit_small_patch14_dinov2.lvd142m"
    cnn_epochs: int = 20
    seg_epochs: int = 40
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    hb_loss_weight: float = 0.3
    target_sensitivity: float = 0.85         # ambang dipilih agar sensitivitas >= ini
    num_workers: int = 2
    pretrained: bool = True                  # False hanya untuk uji offline

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXT


def looks_like_mask(p: Path) -> bool:
    s = p.stem.lower()
    return any(k in s for k in MASK_KEYS)


def read_rgb(path) -> np.ndarray:
    """Baca RGB dengan rotasi EXIF diterapkan (foto HP sering disimpan miring + tag rotasi;
    mask dataset dibuat pada orientasi yang sudah diputar). Piksel transparan (PNG RGBA)
    dijadikan hitam = latar ROI."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    raw = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Gagal membaca gambar: {path}")
    if raw.ndim == 3 and raw.shape[2] == 4:
        if raw.dtype != np.uint8:
            raw = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        a = raw[:, :, 3:4].astype(np.float32) / 255.0
        img = (raw[:, :, :3].astype(np.float32) * a).astype(np.uint8)
    else:
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)      # IMREAD_COLOR menerapkan orientasi EXIF
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _fill_holes(m):
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m, dtype=np.uint8)
    cv2.drawContours(out, cnts, -1, 1, thickness=-1)
    return out


def read_mask(path, shape=None) -> np.ndarray:
    """Baca mask dalam berbagai format yang ada di dataset:
    - mask biner hitam/putih (atau abu-abu),
    - PNG transparan: area objek = alfa > 0,
    - potongan konjungtiva berwarna di atas latar polos (putih/hitam), mis. Eyes-defy Italia 001-031.
    Latar ditentukan dari warna median piksel tepi gambar."""
    m = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise ValueError(f"Gagal membaca mask: {path}")
    if m.dtype != np.uint8:
        m = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if m.ndim == 3 and m.shape[2] == 4:
        alpha = m[:, :, 3]
        frac = (alpha > 127).mean()
        if 0.002 < frac < 0.98:
            out = (alpha > 127).astype(np.uint8)
            m = None
        else:
            m = m[:, :, :3]
    if m is not None:
        if m.ndim == 2:
            m = m[:, :, None]
        edge = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]]).astype(np.int16)
        bg = np.median(edge, axis=0)
        diff = np.abs(m.astype(np.int16) - bg).max(axis=2)
        out = (diff > 40).astype(np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        out = _fill_holes(out)
    if shape is not None and out.shape[:2] != tuple(shape[:2]):
        out = cv2.resize(out, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return out


# ---------------------------------------------------------------- ambang WHO
def anemia_threshold(age=None, sex=None) -> float:
    """Ambang Hb (g/dL) anemia menurut WHO.
    6-59 bulan: 11 | 5-11 th: 11.5 | 12-14 th: 12 | >=15 th: perempuan 12, laki-laki 13.
    Usia tidak diketahui -> 12 (paling umum untuk remaja/dewasa)."""
    try:
        age = float(age)
    except (TypeError, ValueError):
        age = None
    s = str(sex).strip().lower() if sex is not None else ""
    male = s in {"m", "male", "l", "laki-laki", "laki", "pria", "1", "man"}
    if age is not None:
        if age < 5:
            return 11.0
        if age < 12:
            return 11.5
        if age < 15:
            return 12.0
    return 13.0 if male else 12.0


# ---------------------------------------------------------- kualitas foto
def quality_check(rgb: np.ndarray, mask: np.ndarray | None = None, ref: dict | None = None) -> dict:
    """Mengembalikan metrik kualitas + daftar masalah. Dipakai saat training
    (untuk laporan) dan saat inferensi (untuk menolak foto buruk)."""
    h, w = rgb.shape[:2]
    scale = 512 / max(h, w)
    small = cv2.resize(rgb, (int(w * scale), int(h * scale))) if scale < 1 else rgb
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(hsv[:, :, 2].mean())
    ref = ref or {}
    issues = []
    if blur < min(40.0, ref.get("blur_min", 40.0)):
        issues.append("buram")
    if bright < min(50.0, ref.get("bright_min", 50.0)):
        issues.append("terlalu_gelap")
    if bright > max(225.0, ref.get("bright_max", 225.0)):
        issues.append("terlalu_terang")
    glare = 0.0
    area = None
    if mask is not None and mask.any():
        m = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        v, s = hsv[:, :, 2][m], hsv[:, :, 1][m]
        glare = float(((v > 245) & (s < 40)).mean())
        area = float(mask.mean())
        if glare > max(0.08, ref.get("glare_max", 0.08)):
            issues.append("pantulan_cahaya")
        # terlalu kecil = terlalu sedikit piksel untuk menilai warna (bukan sekadar persentase,
        # agar konjungtiva yang dilingkari pada foto besar tetap diterima)
        if int(mask.sum()) < 250 or area < 0.002:
            issues.append("konjungtiva_terlalu_kecil")
    elif mask is not None:
        issues.append("konjungtiva_tidak_terdeteksi")
    return {"blur": blur, "brightness": bright, "glare": glare, "mask_area": area, "issues": issues}


# ------------------------------------------------------------- crop ROI
def crop_roi(rgb: np.ndarray, mask: np.ndarray | None, pad: float = 0.08, zero_bg: bool = True) -> np.ndarray:
    """Potong kotak pembatas konjungtiva. Piksel di luar mask dihitamkan supaya
    model fokus pada warna konjungtiva, bukan kulit/sklera/latar."""
    if mask is None or not mask.any():
        return rgb.copy()
    ys, xs = np.where(mask > 0)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    ph, pw = int((y1 - y0) * pad), int((x1 - x0) * pad)
    y0, x0 = max(0, y0 - ph), max(0, x0 - pw)
    y1, x1 = min(rgb.shape[0], y1 + ph + 1), min(rgb.shape[1], x1 + pw + 1)
    out = rgb[y0:y1, x0:x1].copy()
    if zero_bg:
        out[mask[y0:y1, x0:x1] == 0] = 0
    return out


def letterbox(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size, 3), dtype=img.dtype)
    y, x = (size - nh) // 2, (size - nw) // 2
    out[y:y + nh, x:x + nw] = r
    return out


# --------------------------------------------------------- white balance
def white_balance_from_patch(rgb: np.ndarray, patch_box) -> np.ndarray:
    """Koreksi warna memakai kartu referensi putih/abu yang ikut difoto.
    patch_box = (x0, y0, x1, y1) area kartu. Menyamakan rata-rata kartu ke abu netral."""
    x0, y0, x1, y1 = [int(v) for v in patch_box]
    patch = rgb[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    mean = np.median(patch, axis=0) + 1e-6
    gain = np.clip(mean.mean() / mean, 0.8, 1.25)   # batasi koreksi agar tidak ekstrem
    return np.clip(rgb.astype(np.float32) * gain, 0, 255).astype(np.uint8)


# ------------------------------------------------------------ fitur warna
COLOR_FEATURE_NAMES = [
    "R_mean", "G_mean", "B_mean", "R_med", "G_med", "B_med",
    "L_mean", "a_mean", "b_mean", "a_med", "a_p25", "a_p75",
    "H_mean", "S_mean", "V_mean",
    "erythema_index", "r_chroma", "g_chroma", "RG_ratio", "RB_ratio",
    "high_hue_ratio", "entropy_R", "a_std", "L_std",
]


def color_features(roi_rgb: np.ndarray) -> np.ndarray:
    """Fitur warna klasik dari literatur pucat konjungtiva: CIELAB a*,
    erythema index, rasio kromatik, high hue ratio (HHR), dan entropi.
    Hanya piksel non-hitam (di dalam mask) yang dihitung."""
    px = roi_rgb.reshape(-1, 3)
    valid = px.sum(axis=1) > 15
    if valid.sum() < 50:
        valid = np.ones(len(px), dtype=bool)
    sel = roi_rgb.reshape(-1, 1, 3)[valid]
    rgb = sel.reshape(-1, 3).astype(np.float32)
    lab = cv2.cvtColor(sel.astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    lab[:, 0] *= 100 / 255
    lab[:, 1:] -= 128
    hsv = cv2.cvtColor(sel.astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    s = R + G + B + 1e-6
    ei = np.log10(R + 1) - np.log10(G + 1)
    hue = hsv[:, 0] * 2.0                     # OpenCV H 0-180 -> derajat
    hhr = float(((hue < 15) | (hue > 345)).mean())
    hist = np.histogram(R, bins=32, range=(0, 255))[0].astype(np.float64)
    p = hist / (hist.sum() + 1e-9)
    ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    a = lab[:, 1]
    feats = [
        R.mean(), G.mean(), B.mean(), np.median(R), np.median(G), np.median(B),
        lab[:, 0].mean(), a.mean(), lab[:, 2].mean(), np.median(a), np.percentile(a, 25), np.percentile(a, 75),
        hue.mean(), hsv[:, 1].mean(), hsv[:, 2].mean(),
        ei.mean(), (R / s).mean(), (G / s).mean(), (R / (G + 1)).mean(), (R / (B + 1)).mean(),
        hhr, ent, a.std(), lab[:, 0].std(),
    ]
    return np.asarray(feats, dtype=np.float32)


def risk_tier(prob: float, thr_sens: float, thr_spec: float) -> str:
    """3 tingkat risiko dari probabilitas ensemble.
    tinggi: >= ambang spesifisitas tinggi | sedang: >= ambang sensitivitas | rendah: lainnya."""
    if prob >= thr_spec:
        return "tinggi"
    if prob >= thr_sens:
        return "sedang"
    return "rendah"
