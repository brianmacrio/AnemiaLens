"""Langkah 5: latih model segmentasi konjungtiva (U-Net) dari mask yang tersedia,
lalu buat mask untuk foto yang belum punya mask.

Pakai:  python anemialens/train_seg.py /content/work
        python anemialens/train_seg.py /content/work --roi-sources=CP-AnemiC   (paksa sumber = sudah ter-crop)
Hasil:  work/seg_unet.pt, work/pred_masks/*.png, kolom roi_mask & roi_mode di manifest.csv
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anemialens.common import Config, read_mask, read_rgb, seed_everything  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_unet(pretrained=True):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name="resnet34", encoder_weights="imagenet" if pretrained else None,
                    in_channels=3, classes=1)


def to_tensor(img):
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))


def augment(img, m, rng):
    if rng.random() < 0.5:
        img, m = img[:, ::-1], m[:, ::-1]
    h, w = m.shape
    ang = rng.uniform(-20, 20)
    sc = rng.uniform(0.85, 1.15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[:, 2] += rng.uniform(-0.06, 0.06, 2) * [w, h]
    img = cv2.warpAffine(np.ascontiguousarray(img), M, (w, h), borderMode=cv2.BORDER_REFLECT)
    m = cv2.warpAffine(np.ascontiguousarray(m), M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    img = img.astype(np.float32) * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)   # kecerahan/kontras
    img = img * rng.uniform(0.92, 1.08, 3)                                           # beda white balance kamera
    return np.clip(img, 0, 255).astype(np.uint8), m


def _feather(m, k):
    k = max(3, int(k) | 1)
    return cv2.GaussianBlur(m.astype(np.float32), (k, k), 0)[..., None]


def pallor(img, m, rng, strength=None):
    """Simulasi konjungtiva PUCAT (anemia): geser warna area mask ke arah merah-muda keputihan
    (CIELAB: lebih terang, kemerahan a* turun tetapi tetap positif), tekstur pembuluh tetap ada,
    tepi halus. Mengajari U-Net bahwa konjungtiva pucat tetap konjungtiva."""
    if not m.any():
        return img
    t = rng.uniform(0.35, 0.85) if strength is None else strength
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    sel = m.astype(bool)
    mean = lab[sel].mean(0)
    pale = np.array([rng.uniform(175, 215), rng.uniform(136, 146), rng.uniform(122, 132)], np.float32)
    tgt = lab + t * (pale - mean)                       # geser rata-rata, tekstur dipertahankan
    tgt[..., 1:] = 128 + (tgt[..., 1:] - 128) * (1 - 0.3 * t) + 0.3 * t * (pale[1:] - 128)
    w = _feather(m, max(m.shape) * 0.02)
    out = lab * (1 - w) + tgt * w
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def _soft_background(img):
    """Latar untuk zoom-out: versi sangat buram dari foto itu sendiri (warna kulit/sekitar, tanpa bentuk)."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(4, w // 16), max(4, h // 16)), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), (0, 0), max(h, w) / 20)


def augment_robust(img, m, rng):
    """Augmentasi 'foto HP dunia nyata': zoom-out (mata kecil di bingkai), rotasi/perspektif,
    konjungtiva pucat, white balance & saturasi berbeda, blur, noise, kompresi JPEG, resolusi rendah."""
    h, w = m.shape
    if rng.random() < 0.5:
        img, m = img[:, ::-1], m[:, ::-1]
    img, m = np.ascontiguousarray(img), np.ascontiguousarray(m)
    if rng.random() < 0.5:
        img = pallor(img, m, rng)
    # geometri: skala 0.35-1.3 (zoom-out kuat), rotasi, geser, sedikit perspektif
    sc = rng.uniform(0.35, 1.0) if rng.random() < 0.55 else rng.uniform(1.0, 1.3)
    ang = rng.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[:, 2] += rng.uniform(-0.2, 0.2, 2) * [w, h]
    M3 = np.vstack([M, [0, 0, 1]])
    if rng.random() < 0.4:
        d = 0.06 * w
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + rng.uniform(-d, d, (4, 2)).astype(np.float32)
        M3 = cv2.getPerspectiveTransform(src, dst) @ M3
    bg = _soft_background(img)
    valid = cv2.warpPerspective(np.ones((h, w), np.float32), M3, (w, h), borderMode=cv2.BORDER_CONSTANT)
    valid = cv2.GaussianBlur(valid, (0, 0), 2)[..., None]
    img = cv2.warpPerspective(img, M3, (w, h), borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
    img = (img * valid + bg.astype(np.float32) * (1 - valid)).astype(np.uint8)
    m = cv2.warpPerspective(m, M3, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    # warna & cahaya
    x = img.astype(np.float32)
    x = x * rng.uniform(0.85, 1.15, 3)                                   # white balance
    x = x * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)                  # kontras/kecerahan
    x = 255 * (np.clip(x, 0, 255) / 255) ** rng.uniform(0.7, 1.4)         # gamma
    hsv = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.6, 1.3)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-4, 4)) % 180
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    # kualitas kamera
    if rng.random() < 0.3:
        k = int(rng.choice([3, 5, 7]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.3:
        f = rng.uniform(0.25, 0.6)
        small = cv2.resize(img, (max(16, int(w * f)), max(16, int(h * f))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.3:
        img = np.clip(img + rng.normal(0, rng.uniform(3, 10), img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(30, 90))])
        img = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    return img, m


class SegDS(Dataset):
    def __init__(self, df, size, train, robust=False, fixed_seed=None):
        self.df, self.size, self.train, self.robust = df.reset_index(drop=True), size, train, robust
        self.fixed_seed = fixed_seed   # untuk validasi 'foto HP' yang sama setiap epoch
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.df)

    _cache = {}   # foto 12 MP cukup di-decode & diperkecil sekali (per worker)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        key = (r.image_path, self.size)
        if key not in self._cache:
            img = read_rgb(r.image_path)
            m = read_mask(r.mask_path, img.shape)
            self._cache[key] = (cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA),
                                cv2.resize(m, (self.size, self.size), interpolation=cv2.INTER_NEAREST))
        img, m = (a.copy() for a in self._cache[key])
        if self.fixed_seed is not None:
            img, m = augment_robust(img, m, np.random.default_rng(self.fixed_seed + i))
        elif self.train:
            img, m = (augment_robust if self.robust and self.rng.random() < 0.85 else augment)(img, m, self.rng)
        return to_tensor(img), torch.from_numpy(m.astype(np.float32))[None]


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)).mean()


def dice_score(logits, target):
    p = (torch.sigmoid(logits) > 0.5).float()
    inter = (p * target).sum((1, 2, 3))
    return ((2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)).mean().item()


def largest_component(m):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 1:
        return m
    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (lab == k).astype(np.uint8)


@torch.no_grad()
def predict_mask(model, img, size, device):
    """Prediksi mask + TTA flip horizontal, dikembalikan ke ukuran asli."""
    x = to_tensor(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))[None].to(device)
    p = torch.sigmoid(model(x)) + torch.sigmoid(model(torch.flip(x, [3]))).flip(3)
    p = (p / 2)[0, 0].float().cpu().numpy()
    m = (cv2.resize(p, (img.shape[1], img.shape[0])) > 0.5).astype(np.uint8)
    return largest_component(m)


def flag(name, default=None):
    for a in sys.argv:
        if a == f"--{name}":
            return True
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


def main(work_dir):
    work = Path(work_dir)
    cfg = Config.load(work / "config.json") if (work / "config.json").exists() else Config(work_dir=str(work))
    robust = bool(flag("robust", False))
    epochs = int(flag("epochs", cfg.seg_epochs))
    seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(work / "manifest.csv", keep_default_na=False)
    df["mask_path"] = df.mask_path.astype(str)
    kind = df["mask_kind"] if "mask_kind" in df else pd.Series(np.where(df.mask_path != "", "manual", ""))
    has = df[(df.mask_path != "") & (kind == "manual")]   # hanya mask foto mata utuh
    print(f"Foto dengan mask: {len(has)} dari {len(df)} | device: {device}")
    ckpt = work / "seg_unet.pt"

    if len(has) >= 20:
        pats = has.patient_id.unique()
        rng = np.random.default_rng(cfg.seed)
        val_p = set(rng.choice(pats, size=max(2, int(0.15 * len(pats))), replace=False))
        tr, va = has[~has.patient_id.isin(val_p)], has[has.patient_id.isin(val_p)]
        dl_tr = DataLoader(SegDS(tr, cfg.seg_size, True, robust=robust), batch_size=8, shuffle=True,
                           num_workers=cfg.num_workers, drop_last=len(tr) > 8)
        dl_va = DataLoader(SegDS(va, cfg.seg_size, False), batch_size=8, num_workers=cfg.num_workers)
        # validasi kedua: foto validasi yang sama tetapi 'dibuat seperti foto HP' (zoom-out, pucat, blur, JPEG)
        dl_hp = DataLoader(SegDS(va, cfg.seg_size, False, fixed_seed=123), batch_size=8, num_workers=cfg.num_workers)
        print(f"Mode augmentasi: {'ROBUST (foto HP + konjungtiva pucat)' if robust else 'standar'} | epoch: {epochs}")
        model = build_unet(cfg.pretrained).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=epochs * len(dl_tr))
        scaler = torch.amp.GradScaler(enabled=device == "cuda")
        best, best_pair = -1, (0.0, 0.0)
        for ep in range(epochs):
            model.train()
            t0, tl = time.time(), 0.0
            for x, y in dl_tr:
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, enabled=device == "cuda"):
                    out = model(x)
                    loss = F.binary_cross_entropy_with_logits(out, y) + dice_loss(out, y)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                tl += loss.item()
            model.eval()

            def evaluate(dl):
                with torch.no_grad():
                    return float(np.mean([dice_score(model(x.to(device)), y.to(device)) for x, y in dl]))
            d, dh = evaluate(dl_va), evaluate(dl_hp)
            score = (d + dh) / 2 if robust else d       # mode robust: pilih model yang bagus di keduanya
            if score > best:
                best, best_pair = score, (d, dh)
                torch.save(model.state_dict(), ckpt)
            print(f"  epoch {ep + 1:02d}/{epochs} loss {tl / len(dl_tr):.3f} | Dice val {d:.3f} | Dice 'foto HP' {dh:.3f} "
                  f"| terbaik {best:.3f} | {time.time() - t0:.0f}s")
        print(f"Segmentasi selesai. Dice validasi: {best_pair[0]:.3f} | Dice simulasi foto HP: {best_pair[1]:.3f}")
        (work / "seg_metrics.txt").write_text(f"dice_val={best_pair[0]:.4f}\ndice_val_hp={best_pair[1]:.4f}\n"
                                              f"robust={int(robust)}\nn_train={len(tr)}\nn_val={len(va)}\n")
    elif ckpt.exists():
        print("Mask kurang dari 20, memakai checkpoint segmentasi yang sudah ada.")
    else:
        print("PERINGATAN: tidak cukup mask untuk melatih segmentasi. Semua foto dipakai utuh (roi_mode=full).")

    if flag("no-predict", False):
        return
    # ------------------------------------------------ buat mask untuk semua foto
    out_dir = work / "pred_masks"
    out_dir.mkdir(exist_ok=True)
    roi_mask, roi_mode, pred_area = [], [], []
    model = None
    if ckpt.exists():
        model = build_unet(False).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
    for i, r in df.iterrows():
        if r.mask_path:
            roi_mask.append(r.mask_path)
            roi_mode.append("gt" if getattr(r, "mask_kind", "manual") != "alpha" else "alpha")
            pred_area.append(np.nan)
            continue
        if model is None:
            roi_mask.append("")
            roi_mode.append("full")
            pred_area.append(np.nan)
            continue
        img = read_rgb(r.image_path)
        m = predict_mask(model, img, cfg.seg_size, device)
        area = float(m.mean())
        pred_area.append(area)
        # Foto yang sudah berupa potongan konjungtiva (mask menutup hampir seluruh gambar)
        # atau segmentasi gagal -> pakai gambar utuh.
        if area >= 0.45 or area < 0.005:
            roi_mask.append("")
            roi_mode.append("full")
        else:
            p = out_dir / f"{i:06d}.png"
            cv2.imwrite(str(p), m * 255)
            roi_mask.append(str(p))
            roi_mode.append("pred")
    df["roi_mask"], df["roi_mode"], df["pred_mask_area"] = roi_mask, roi_mode, pred_area
    # Sumber tanpa mask asli yang fotonya sudah berupa potongan konjungtiva (mis. CP-AnemiC):
    # jika median area prediksi besar, perlakukan seluruh sumber sebagai ROI utuh.
    forced = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--roi-sources=")]
    forced = set(forced[0].split(",")) if forced else set()
    for src, g in df.groupby("source"):
        if g.roi_mode.isin(["gt", "alpha"]).any():
            continue
        med = g.pred_mask_area.median()
        if src in forced or (pd.notna(med) and med >= 0.30):
            df.loc[g.index, "roi_mask"] = ""
            df.loc[g.index, "roi_mode"] = "full"
            print(f"  Sumber '{src}' diperlakukan sebagai foto yang sudah ter-crop (median area {med:.2f}).")
    df.to_csv(work / "manifest.csv", index=False)
    print("Mode ROI:", df.roi_mode.value_counts().to_dict())
    print("Per sumber:\n", pd.crosstab(df.source, df.roi_mode).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "/content/work")
