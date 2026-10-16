"""Inferensi AnemiaLens untuk satu foto.

    from anemialens.predict import AnemiaLens
    lens = AnemiaLens("/content/work/model")
    hasil = lens.analyze(rgb_array, use_card=True)

Alur: cek kualitas -> (opsional) koreksi warna dengan kartu putih -> segmentasi
konjungtiva -> crop -> 3 model -> ensemble -> tingkat risiko + estimasi Hb.
Pakai CLI:  python anemialens/predict.py /content/work/model foto.jpg
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anemialens.common import (Config, color_features, crop_roi, letterbox, quality_check,  # noqa: E402
                               read_rgb, risk_tier, white_balance_from_patch)
from anemialens.models import MultiTaskNet, build_dino, dino_embed, predict_cnn_arrays  # noqa: E402
from anemialens.roi_guard import check as roi_check, manual_mask, refine_user_mask  # noqa: E402

BLOCKING = {"buram", "terlalu_gelap", "terlalu_terang", "pantulan_cahaya",
            "konjungtiva_terlalu_kecil", "konjungtiva_tidak_terdeteksi"}
ADVICE = {
    "buram": "Foto buram. Pegang HP stabil dan ketuk layar untuk fokus pada mata.",
    "terlalu_gelap": "Foto terlalu gelap. Foto di dekat jendela atau lampu putih, jangan pakai flash.",
    "terlalu_terang": "Foto terlalu terang. Jauhi cahaya langsung atau matikan flash.",
    "pantulan_cahaya": "Ada pantulan cahaya di kelopak mata. Ubah sudut HP sedikit.",
    "konjungtiva_terlalu_kecil": "Bagian dalam kelopak bawah terlalu kecil. Dekatkan kamera dan tarik kelopak bawah ke bawah.",
    "konjungtiva_tidak_terdeteksi": "AI belum menemukan bagian dalam kelopak bawah. Tarik kelopak bawah lebih lebar lalu foto ulang, "
                                    "atau lingkari sendiri bagian dalam kelopak (yang basah, merah muda atau pucat).",
    "bentuk_roi_tidak_wajar": "Area yang terdeteksi tidak berbentuk seperti bagian dalam kelopak bawah. Tarik kelopak bawah "
                              "sampai bagian dalam yang merah muda terlihat jelas, foto dari dekat (mata memenuhi layar), "
                              "atau lingkari sendiri bagian dalam kelopak.",
    "roi_tidak_kemerahan": "Area yang terdeteksi lebih mirip kulit atau bagian putih mata daripada konjungtiva. "
                           "Tarik kelopak bawah lebih jelas, atau lingkari tepat bagian dalam kelopak (bukan kulit di bawahnya).",
    "warna_di_luar_data_latih": "Warna atau pencahayaan foto terlalu berbeda dari data latih, jadi hasilnya tidak bisa "
                                "dipercaya. Foto ulang di cahaya putih yang merata, tanpa flash dan tanpa filter.",
}
GUARD_WARN = ("Warna area yang dilingkari di luar jangkauan data latih: bisa karena sangat pucat, atau area itu bukan "
              "konjungtiva. Hasil kurang pasti, sebaiknya langsung cek hemoglobin.")
DISCLAIMER = ("Hasil ini adalah skrining awal berbasis foto, BUKAN diagnosis. "
              "Pastikan dengan pemeriksaan hemoglobin (cek darah) di puskesmas, klinik, atau laboratorium.")


def auto_white_patch(rgb):
    """Cari kartu putih/abu: area besar, saturasi rendah, cukup terang."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    cand = ((hsv[:, :, 1] < 35) & (hsv[:, :, 2] > 150)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand, 8)
    total = rgb.shape[0] * rgb.shape[1]
    best = None
    for k in range(1, n):
        x, y, w, h, a = st[k]
        rect = a / max(1, w * h)                 # kartu = persegi panjang penuh; sklera = lonjong
        if 0.01 * total <= a <= 0.4 * total and rect >= 0.85 and 0.25 <= w / max(1, h) <= 4:
            if best is None or a > best[4]:
                best = (x, y, w, h, a)
    if best is None:
        return None
    x, y, w, h, _ = best
    sx, sy = int(w * 0.2), int(h * 0.2)          # ambil bagian tengah kartu
    return (x + sx, y + sy, x + w - sx, y + h - sy)


class AnemiaLens:
    def __init__(self, model_dir):
        d = Path(model_dir)
        self.cfg = Config.load(d / "config.json")
        self.thr = json.loads((d / "thresholds.json").read_text())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seg = None
        if (d / "seg_unet.pt").exists():
            from anemialens.train_seg import build_unet
            self.seg = build_unet(False).to(self.device)
            self.seg.load_state_dict(torch.load(d / "seg_unet.pt", map_location=self.device))
            self.seg.eval()
        names = [n for n in ("color", "color_rf", "dino") if (d / f"{n}_clf.joblib").exists()]
        self.sk = {n: joblib.load(d / f"{n}_clf.joblib") for n in names}
        self.hbm = {n: joblib.load(d / f"{n}_hb.joblib") for n in names if (d / f"{n}_hb.joblib").exists()}
        # DINOv2 hanya dimuat bila termasuk ensemble (hemat RAM & tidak perlu unduh dari HF Hub)
        if "dino" not in self.thr["models"]:
            self.sk.pop("dino", None)
            self.hbm.pop("dino", None)
        self.dino = (build_dino(self.cfg.dino_backbone, self.cfg.pretrained, self.cfg.img_size)
                     if "dino" in self.sk else None)
        self.cnn = None
        if "cnn" in self.thr["models"] and (d / "cnn.pt").exists():
            self.cnn = MultiTaskNet(self.cfg.cnn_backbone, pretrained=False).to(self.device)
            self.cnn.load_state_dict(torch.load(d / "cnn.pt", map_location=self.device))
            self.cnn.eval()
        self.hb_mae = self.thr.get("hb_mae")
        qr = d / "quality_ref.json"
        self.qref = json.loads(qr.read_text()) if qr.exists() else None
        rr = d / "roi_ref.json"
        self.rref = json.loads(rr.read_text()) if rr.exists() else None
        if self.rref is None:
            print("PERINGATAN: roi_ref.json belum ada. Jalankan roi_guard.py agar foto non-konjungtiva ditolak.")

    def segment(self, rgb):
        if self.seg is None:
            return None, "full"
        from anemialens.train_seg import predict_mask
        m = predict_mask(self.seg, rgb, self.cfg.seg_size, self.device)
        a = m.mean()
        if a >= 0.45:
            return None, "full"
        if a < 0.005:
            return np.zeros_like(m), "none"
        return m, "pred"

    def analyze(self, rgb, use_card=False, force=False, mask=None, manual_crop=False):
        card = None
        if use_card:
            card = auto_white_patch(rgb)
            if card is not None:
                rgb = white_balance_from_patch(rgb, card)
        if manual_crop and mask is not None:  # pengguna melingkari konjungtiva sendiri (lasso)
            mask, mode = refine_user_mask(rgb, mask), "manual"
        elif manual_crop:                     # pengguna hanya crop kotak ke sekitar konjungtiva
            mask, mode = manual_mask(rgb), "manual"
        elif mask is None:
            mask, mode = self.segment(rgb)
        else:
            mode = "pred" if mask.any() else "full"
        q = quality_check(rgb, mask if mode not in ("full",) else None, self.qref)
        issues = [i for i in q["issues"] if i in BLOCKING]
        overlay = rgb.copy()
        if mask is not None and mask.any():
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts, -1, (0, 255, 120), max(2, rgb.shape[1] // 300))
        if card is not None:
            cv2.rectangle(overlay, card[:2], card[2:], (80, 160, 255), max(2, rgb.shape[1] // 300))
        base = {"quality": q, "roi_mode": mode, "card_used": card is not None, "overlay": overlay,
                "disclaimer": DISCLAIMER}
        if issues and not force:
            return {**base, "status": "foto_tidak_layak", "advice": [ADVICE[i] for i in issues]}
        roi = crop_roi(rgb, mask if mode in ("pred", "manual") else None)
        roi_std = letterbox(roi, 448)                       # sama dengan pra-proses saat training
        xc = color_features(roi)[None]
        guard = roi_check(self.rref, mask, mode, xc)
        guard_warn = []
        if guard and mode == "manual":
            # Area ditandai sendiri oleh pengguna: jangan ditolak, karena konjungtiva yang sangat pucat
            # (anemia berat) bisa jatuh di luar jangkauan warna data latih. Beri peringatan saja.
            guard_warn = [GUARD_WARN]
        elif guard and not force:
            return {**base, "status": "foto_tidak_layak", "advice": [ADVICE[g] for g in guard], "guard": guard}
        feats = {"color": xc, "color_rf": xc}
        if self.dino is not None:
            feats["dino"] = dino_embed(self.dino, [roi_std], self.cfg.img_size, self.device)
        probs = {n: float(m.predict_proba(feats[n])[0, 1]) for n, m in self.sk.items()}
        hbs = [float(m.predict(feats[n])[0]) for n, m in self.hbm.items() if n in self.thr["models"]]
        if self.cnn is not None:
            p, h = predict_cnn_arrays(self.cnn, [roi_std], self.cfg.img_size, self.device)
            probs["cnn"] = float(p[0])
            hbs.append(float(h[0]))
        member = [probs[m] for m in self.thr["models"] if m in probs]
        prob = float(np.mean(member))
        disagree = len(member) > 1 and (max(member) - min(member)) > 0.5
        hbs = [float(np.clip(h, 4.0, 20.0)) for h in hbs]
        tier = risk_tier(prob, self.thr["thr_sensitive"], self.thr["thr_specific"])
        hb = float(np.mean(hbs)) if hbs else None
        msg = {
            "tinggi": "Risiko anemia TINGGI. Segera periksa hemoglobin di puskesmas/klinik dalam beberapa hari ini.",
            "sedang": "Ada tanda kemungkinan anemia. Disarankan cek hemoglobin untuk memastikan.",
            "rendah": "Tidak tampak tanda pucat yang jelas. Tetap cek Hb bila lemas, pusing, atau sesak berlanjut.",
        }[tier]
        return {**base, "status": "ok", "probability": prob, "risk_tier": tier, "message": msg,
                "hb_estimate": hb, "hb_mae": self.hb_mae, "per_model": probs,
                "quality_warnings": [ADVICE[i] for i in q["issues"] if i in ADVICE]
                + (["Model-model tidak sepakat untuk foto ini, jadi hasilnya kurang yakin. Disarankan foto ulang."]
                   if disagree else []) + guard_warn, "guard": guard}


if __name__ == "__main__":
    lens = AnemiaLens(sys.argv[1])
    r = lens.analyze(read_rgb(sys.argv[2]), use_card="--card" in sys.argv)
    r.pop("overlay")
    print(json.dumps(r, indent=2, ensure_ascii=False, default=str))
