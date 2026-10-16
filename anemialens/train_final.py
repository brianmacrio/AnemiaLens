"""Langkah 8: latih model final memakai SEMUA data (ambang dari validasi silang),
lalu simpan semua artefak yang dibutuhkan aplikasi ke work/model/.

Pakai:  python anemialens/train_final.py /content/work [--no-cnn]
"""
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anemialens.common import COLOR_FEATURE_NAMES, Config, seed_everything  # noqa: E402
from anemialens.models import make_clf, make_reg, make_rf_clf, make_rf_reg, train_cnn  # noqa: E402
from anemialens.train_eval import inner_splits  # noqa: E402


def main(work_dir, use_cnn=True):
    work = Path(work_dir)
    cfg = Config.load(work / "config.json") if (work / "config.json").exists() else Config(work_dir=str(work))
    seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = work / "model"
    out.mkdir(exist_ok=True)
    df = pd.read_csv(work / "features.csv", keep_default_na=False, na_values=[""])
    y = df.label.values.astype(int)
    hb = pd.to_numeric(df.hb, errors="coerce").values.astype(float)
    g = df.patient_id.astype(str).values
    Xc = df[COLOR_FEATURE_NAMES].values.astype(np.float32)
    Xd = np.load(work / "dino_emb.npy")
    thr = json.loads((work / "thresholds.json").read_text())
    if use_cnn and "cnn" not in thr["models"]:
        print("Info: CNN bukan anggota ensemble (AUC validasi lebih rendah / tidak dilatih), jadi tidak dilatih ulang.")
        use_cnn = False
    print("Anggota ensemble:", thr["models"])
    m = ~np.isnan(hb)
    for name, X, kind in (("color", Xc, "linear"), ("color_rf", Xc, "rf"), ("dino", Xd, "linear")):
        clf = make_rf_clf() if kind == "rf" else make_clf(inner_splits(g, y))
        joblib.dump(clf.fit(X, y), out / f"{name}_clf.joblib")
        if m.sum() >= 20:
            reg = make_rf_reg() if kind == "rf" else make_reg()
            joblib.dump(reg.fit(X[m], hb[m]), out / f"{name}_hb.joblib")
        print(f"  model {name} tersimpan")
    if use_cnn:
        print("Melatih CNN final pada semua data...")
        model = train_cnn(cfg, df.roi_path.values, y, hb, device, log_prefix="[cnn final] ")
        torch.save(model.state_dict(), out / "cnn.pt")
    if (work / "seg_unet.pt").exists():
        shutil.copy(work / "seg_unet.pt", out / "seg_unet.pt")
    (out / "thresholds.json").write_text(json.dumps(thr, indent=2))
    cfg.save(out / "config.json")
    for f in ("report.md", "roc.png", "hb_scatter.png", "seg_metrics.txt", "quality_ref.json"):
        if (work / f).exists():
            shutil.copy(work / f, out / f)
    from anemialens.roi_guard import build_ref
    build_ref(work, out)
    print(f"Artefak model siap di: {out}")
    print(sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "/content/work",
         use_cnn="--no-cnn" not in sys.argv)
