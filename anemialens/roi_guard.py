"""Pengaman ROI: memastikan area yang dinilai benar-benar konjungtiva, bukan kulit/sklera.

1. build_ref(): dari data latih, simpan statistik "seperti apa konjungtiva yang sah":
   - bentuk mask (luas relatif, kelonjongan) dari mask manual dataset,
   - warna (a* minimum) dan jarak Mahalanobis fitur warna (deteksi out-of-distribution).
2. check(): menolak ROI yang bentuk/warnanya di luar jangkauan data latih.
3. manual_mask(): untuk foto yang sudah di-crop manual oleh pengguna, ambil piksel
   kemerahan (CIELAB a* tinggi) sebagai konjungtiva.

Pakai:  python anemialens/roi_guard.py /content/work /content/work/model
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anemialens.common import COLOR_FEATURE_NAMES, read_mask  # noqa: E402


AREA_MIN_ABS = 0.002


def mask_shape(mask):
    """(luas relatif, kelonjongan) dari komponen terbesar."""
    m = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0, 0.0
    c = max(cnts, key=cv2.contourArea)
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    elong = max(w, h) / max(1.0, min(w, h))
    return float(m.mean()), float(elong)


def build_ref(work_dir, model_dir):
    work, out = Path(work_dir), Path(model_dir)
    f = pd.read_csv(work / "features.csv", keep_default_na=False, na_values=[""])
    X = f[COLOR_FEATURE_NAMES].values.astype(np.float64)
    from sklearn.covariance import LedoitWolf
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    lw = LedoitWolf().fit(Z)
    d = np.sqrt(np.einsum("ij,jk,ik->i", Z - lw.location_, lw.precision_, Z - lw.location_))
    gt = f[f.roi_mode == "gt"] if "roi_mode" in f.columns else f.iloc[0:0]
    if "mask_area_frac" in f.columns:
        areas, elongs = gt.mask_area_frac.dropna().tolist(), gt.mask_elong.dropna().tolist()
    else:
        areas, elongs = [], []
        for _, r in gt.iterrows():
            try:
                a, e = mask_shape(read_mask(r.roi_mask))
            except Exception:  # noqa: BLE001
                continue
            if a > 0:
                areas.append(a)
                elongs.append(e)
    ref = {
        "feat_mu": mu.tolist(), "feat_sd": sd.tolist(), "lw_loc": lw.location_.tolist(),
        "lw_prec": lw.precision_.tolist(), "mahal_max": float(np.percentile(d, 99) * 1.25),
        "a_mean_min": float(np.percentile(f.a_mean, 1)),
        "area_min": float(np.percentile(areas, 1) * 0.5) if areas else 0.003,
        "area_max": float(np.percentile(areas, 99) * 1.5) if areas else 0.35,
        "elong_min": float(max(1.5, np.percentile(elongs, 2) * 0.8)) if elongs else 1.8,
        "n_gt_masks": len(areas),
    }
    (out / "roi_ref.json").write_text(json.dumps(ref))
    print("roi_ref.json tersimpan:", {k: (round(v, 4) if isinstance(v, float) else v)
                                      for k, v in ref.items() if not isinstance(v, list)})
    return ref


def mahalanobis(ref, x):
    z = (np.asarray(x, dtype=np.float64).ravel() - np.array(ref["feat_mu"])) / np.array(ref["feat_sd"])
    v = z - np.array(ref["lw_loc"])
    return float(np.sqrt(v @ np.array(ref["lw_prec"]) @ v))


def check(ref, mask, mode, feats):
    """Kembalikan daftar alasan penolakan (kosong = ROI layak)."""
    if ref is None:
        return []
    reasons = []
    if mode == "pred" and mask is not None and mask.any():
        a, e = mask_shape(mask)
        # batas bawah luas dilonggarkan: segmentasi v2 menangani foto HP satu mata utuh, di mana
        # konjungtiva hanya ~0,5-2% bingkai (data latih close-up >1%). Bentuk lonjong tetap diwajibkan.
        if not (min(ref["area_min"], AREA_MIN_ABS) <= a <= ref["area_max"]) or e < ref["elong_min"]:
            reasons.append("bentuk_roi_tidak_wajar")
    fdict = dict(zip(COLOR_FEATURE_NAMES, np.asarray(feats).ravel()))
    if fdict["a_mean"] < ref["a_mean_min"]:
        reasons.append("roi_tidak_kemerahan")
    if mahalanobis(ref, feats) > ref["mahal_max"]:
        reasons.append("warna_di_luar_data_latih")
    return reasons


def refine_user_mask(rgb, mask):
    """Mask dari lingkaran (lasso) pengguna. Area dipercaya apa adanya - TIDAK dipilih berdasarkan
    kemerahan, karena konjungtiva anemia justru pucat. Hanya buang piksel yang pasti bukan jaringan:
    bulu mata/pupil (sangat gelap) dan pantulan cahaya (putih menyilaukan)."""
    m = (mask > 0).astype(np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    v, s = hsv[:, :, 2], hsv[:, :, 1]
    bad = (v < 45) | ((v > 245) & (s < 30))
    keep = (m & ~bad).astype(np.uint8)
    keep = cv2.morphologyEx(keep, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return keep if keep.sum() >= 0.3 * m.sum() else m


def manual_mask(rgb, top_frac=0.25):
    """Foto sudah di-crop pengguna ke sekitar konjungtiva. Konjungtiva = jaringan paling
    merah DAN jenuh di crop itu (kulit sawo matang juga punya a* tinggi, jadi dipakai
    persentil atas + saturasi, bukan ambang tetap). Ambil komponen terbesar lalu rapikan."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    a = cv2.GaussianBlur(lab[:, :, 1], (5, 5), 0)
    sat = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
    valid = rgb.sum(axis=2) > 15
    score = (a - a[valid].mean()) / (a[valid].std() + 1e-6) + 0.5 * (sat - sat[valid].mean()) / (sat[valid].std() + 1e-6)
    thr = np.percentile(score[valid], 100 * (1 - top_frac))
    m = ((score >= thr) & valid).astype(np.uint8)
    k = max(3, int(min(rgb.shape[:2]) * 0.03) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k * 2 + 1, k * 2 + 1), np.uint8))
    n, lab_cc, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m
    kk = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    return (lab_cc == kk).astype(np.uint8)


if __name__ == "__main__":
    build_ref(sys.argv[1] if len(sys.argv) > 1 else "/content/work",
              sys.argv[2] if len(sys.argv) > 2 else "/content/work/model")
