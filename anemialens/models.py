"""Model AnemiaLens:
  A. Fitur warna klasik (a*, erythema index, HHR, dst.) -> regresi logistik / ridge
  A2. Fitur warna yang sama -> Random Forest (menangkap hubungan non-linear)
  B. Embedding DINOv2 ViT-S/14 (beku) -> regresi logistik / ridge  (linear probe)
  C. CNN multi-task (EfficientNet-B0) di-fine-tune: kepala klasifikasi anemia + kepala regresi Hb
Prediksi akhir = rata-rata (ensemble) dari model yang tersedia."""
from __future__ import annotations

import math
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
HB_MU, HB_SD = 12.0, 2.0


def to_tensor(img):
    return torch.from_numpy(((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1))


def load_roi(path, size):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def augment_roi(img, rng):
    """Augmentasi yang TIDAK merusak sinyal warna: geometri bebas, tapi perubahan
    warna dibatasi (tanpa hue jitter, gain per kanal kecil untuk meniru beda kamera)."""
    if rng.random() < 0.5:
        img = img[:, ::-1]
    if rng.random() < 0.3:
        img = img[::-1]
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-15, 15), rng.uniform(0.9, 1.15))
    M[:, 2] += rng.uniform(-0.05, 0.05, 2) * [w, h]
    img = cv2.warpAffine(np.ascontiguousarray(img), M, (w, h), borderMode=cv2.BORDER_CONSTANT)
    f = img.astype(np.float32)
    nz = f.sum(axis=2, keepdims=True) > 0
    f = f * rng.uniform(0.85, 1.15) + rng.uniform(-12, 12)      # kecerahan & kontras
    f = f * rng.uniform(0.95, 1.05, 3)                           # white balance kamera
    f = np.where(nz, f, 0)
    if rng.random() < 0.2:                                       # random erasing kecil
        eh, ew = int(h * rng.uniform(0.1, 0.25)), int(w * rng.uniform(0.1, 0.25))
        y, x = rng.integers(0, h - eh), rng.integers(0, w - ew)
        f[y:y + eh, x:x + ew] = 0
    return np.clip(f, 0, 255).astype(np.uint8)


class RoiDS(Dataset):
    def __init__(self, paths, labels, hbs, size, train):
        self.p, self.y, self.hb, self.size, self.train = list(paths), np.asarray(labels, np.float32), \
            np.asarray(hbs, np.float32), size, train
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        img = load_roi(self.p[i], self.size)
        if self.train:
            img = augment_roi(img, self.rng)
        hb = self.hb[i]
        has = 0.0 if np.isnan(hb) else 1.0
        hbn = 0.0 if np.isnan(hb) else (hb - HB_MU) / HB_SD
        return to_tensor(img), torch.tensor(self.y[i]), torch.tensor(hbn, dtype=torch.float32), torch.tensor(has)


class MultiTaskNet(nn.Module):
    def __init__(self, backbone="efficientnet_b0", pretrained=True, drop=0.3):
        super().__init__()
        import timm
        self.body = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        d = self.body.num_features
        self.drop = nn.Dropout(drop)
        self.cls = nn.Linear(d, 1)
        self.hb = nn.Linear(d, 1)

    def forward(self, x):
        f = self.drop(self.body(x))
        return self.cls(f).squeeze(1), self.hb(f).squeeze(1)


def train_cnn(cfg, tr_paths, tr_y, tr_hb, device, log_prefix=""):
    seed = cfg.seed
    torch.manual_seed(seed)
    model = MultiTaskNet(cfg.cnn_backbone, cfg.pretrained).to(device)
    ds = RoiDS(tr_paths, tr_y, tr_hb, cfg.img_size, True)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                    drop_last=len(ds) > cfg.batch_size, pin_memory=device == "cuda")
    pos = float(np.mean(tr_y))
    pos_weight = torch.tensor((1 - pos) / max(pos, 1e-3), device=device)
    body_p = list(model.body.parameters())
    head_p = list(model.cls.parameters()) + list(model.hb.parameters())
    opt = torch.optim.AdamW([{"params": body_p, "lr": cfg.lr * 0.5}, {"params": head_p, "lr": cfg.lr * 3}],
                            weight_decay=cfg.weight_decay)
    total = cfg.cnn_epochs * len(dl)
    warm = max(1, len(dl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    for ep in range(cfg.cnn_epochs):
        model.train()
        t0, tl = time.time(), 0.0
        for x, y, hb, has in dl:
            x, y, hb, has = x.to(device), y.to(device), hb.to(device), has.to(device)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                lc, lh = model(x)
                loss = F.binary_cross_entropy_with_logits(lc.float(), y, pos_weight=pos_weight)
                if has.sum() > 0:
                    lhb = (F.smooth_l1_loss(lh.float(), hb, reduction="none") * has).sum() / has.sum()
                    loss = loss + cfg.hb_loss_weight * lhb
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item()
        if (ep + 1) % 5 == 0 or ep == 0 or ep + 1 == cfg.cnn_epochs:
            print(f"    {log_prefix}epoch {ep + 1:02d}/{cfg.cnn_epochs} loss {tl / max(1, len(dl)):.3f} ({time.time() - t0:.0f}s)")
    return model


@torch.no_grad()
def predict_cnn(model, paths, size, device, batch=64):
    model.eval()
    ds = RoiDS(paths, np.zeros(len(paths)), np.full(len(paths), np.nan), size, False)
    dl = DataLoader(ds, batch_size=batch, shuffle=False)
    probs, hbs = [], []
    for x, *_ in dl:
        x = x.to(device)
        c1, h1 = model(x)
        c2, h2 = model(torch.flip(x, [3]))                # TTA flip horizontal
        probs.append(torch.sigmoid((c1 + c2) / 2).float().cpu().numpy())
        hbs.append(((h1 + h2) / 2).float().cpu().numpy() * HB_SD + HB_MU)
    return np.concatenate(probs), np.concatenate(hbs)


@torch.no_grad()
def predict_cnn_arrays(model, imgs, size, device):
    """Seperti predict_cnn tetapi untuk array RGB (dipakai aplikasi)."""
    model.eval()
    x = torch.stack([to_tensor(cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)) for a in imgs]).to(device)
    c1, h1 = model(x)
    c2, h2 = model(torch.flip(x, [3]))
    return (torch.sigmoid((c1 + c2) / 2).float().cpu().numpy(),
            ((h1 + h2) / 2).float().cpu().numpy() * HB_SD + HB_MU)


# ---------------------------------------------------------------- DINOv2
def build_dino(name, pretrained=True, size=224):
    import timm
    m = timm.create_model(name, pretrained=pretrained, num_classes=0, img_size=size)
    return m.eval()


@torch.no_grad()
def dino_embed(model, imgs_or_paths, size, device, batch=64):
    """Embedding beku + TTA flip. Menerima daftar path ROI atau array RGB."""
    model = model.to(device)
    out = []
    for i in range(0, len(imgs_or_paths), batch):
        chunk = imgs_or_paths[i:i + batch]
        arr = [load_roi(p, size) if isinstance(p, str) else cv2.resize(p, (size, size), interpolation=cv2.INTER_AREA)
               for p in chunk]
        x = torch.stack([to_tensor(a) for a in arr]).to(device)
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            e = (model(x) + model(torch.flip(x, [3]))) / 2
        out.append(F.normalize(e.float(), dim=1).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------- model sklearn
def make_clf(groups_inner=None):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=[0.003, 0.01, 0.03, 0.1, 0.3, 1.0], cv=groups_inner or 3, scoring="roc_auc",
        class_weight="balanced", max_iter=5000))


def make_reg():
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))


def make_rf_clf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=3, class_weight="balanced_subsample",
                                  random_state=0, n_jobs=-1)


def make_rf_reg():
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=0, n_jobs=-1)
