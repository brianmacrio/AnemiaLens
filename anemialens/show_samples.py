"""Cek visual: grid contoh ROI per sumber & kelas, dan contoh mask hasil segmentasi.
Pakai:  python anemialens/show_samples.py /content/work
Hasil:  work/samples_roi.png, work/samples_mask.png
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anemialens.common import read_mask, read_rgb  # noqa: E402

T = 150


def tile(img):
    return cv2.resize(img, (T, T), interpolation=cv2.INTER_AREA)


def label_bar(text, w):
    bar = np.full((22, w, 3), 255, np.uint8)
    cv2.putText(bar, text[:60], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return bar


def main(work_dir, per=6):
    work = Path(work_dir)
    rng = np.random.default_rng(0)
    if (work / "features.csv").exists():
        df = pd.read_csv(work / "features.csv", keep_default_na=False)
        rows = []
        for (src, lab), g in df.groupby(["source", "label"]):
            pick = g.sample(min(per, len(g)), random_state=0)
            ims = [tile(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)) for p in pick.roi_path]
            ims += [np.full((T, T, 3), 255, np.uint8)] * (per - len(ims))
            rows += [label_bar(f"{src} | {'ANEMIA' if lab == 1 else 'normal'} | n={len(g)}", per * T), np.hstack(ims)]
        cv2.imwrite(str(work / "samples_roi.png"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
        print("Tersimpan:", work / "samples_roi.png")
    df = pd.read_csv(work / "manifest.csv", keep_default_na=False)
    if "roi_mode" in df:
        rows = []
        for mode in ("gt", "pred", "alpha"):
            g = df[df.roi_mode == mode]
            if not len(g):
                continue
            pick = g.iloc[rng.choice(len(g), min(per, len(g)), replace=False)]
            ims = []
            for _, r in pick.iterrows():
                img = read_rgb(r.image_path)
                m = read_mask(r.roi_mask, img.shape)
                ov = img.copy()
                ov[m > 0] = (0.5 * ov[m > 0] + 0.5 * np.array([0, 255, 120])).astype(np.uint8)
                ims.append(tile(ov))
            ims += [np.full((T, T, 3), 255, np.uint8)] * (per - len(ims))
            rows += [label_bar({"gt": "mask asli (dataset)", "pred": "mask prediksi U-Net", "alpha": "sudah ter-crop (latar transparan)"}[mode], per * T), np.hstack(ims)]
        if rows:
            cv2.imwrite(str(work / "samples_mask.png"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
            print("Tersimpan:", work / "samples_mask.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/content/work")
