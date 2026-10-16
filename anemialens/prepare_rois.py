"""Langkah 6: potong area konjungtiva setiap foto, hitung fitur warna & kualitas.

Pakai:  python anemialens/prepare_rois.py /content/work
Hasil:  work/rois/*.png (crop konjungtiva, latar hitam), work/features.csv
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anemialens.common import (COLOR_FEATURE_NAMES, color_features, crop_roi, letterbox,  # noqa: E402
                               quality_check, read_mask, read_rgb)

STORE = 448  # ukuran simpan ROI (classifier memakai 224)


def main(work_dir):
    work = Path(work_dir)
    df = pd.read_csv(work / "manifest.csv", keep_default_na=False)
    out = work / "rois"
    out.mkdir(exist_ok=True)
    from anemialens.roi_guard import mask_shape
    feats, qual, paths, shapes = [], [], [], []
    for i, r in df.iterrows():
        img = read_rgb(r.image_path)
        m = read_mask(r.roi_mask, img.shape) if r.roi_mask else None
        q = quality_check(img, m)
        roi = crop_roi(img, m)
        feats.append(color_features(roi))
        shapes.append(mask_shape(m) if m is not None else (np.nan, np.nan))
        qual.append(q)
        p = out / f"{i:06d}.png"
        cv2.imwrite(str(p), cv2.cvtColor(letterbox(roi, STORE), cv2.COLOR_RGB2BGR))
        paths.append(str(p))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(df)}")
    F = pd.DataFrame(np.vstack(feats), columns=COLOR_FEATURE_NAMES)
    Q = pd.DataFrame(qual)
    Q["mask_area_frac"], Q["mask_elong"] = [s_[0] for s_ in shapes], [s_[1] for s_ in shapes]
    Q["issues"] = Q.issues.map(lambda x: ";".join(x))
    res = pd.concat([df.reset_index(drop=True), pd.Series(paths, name="roi_path"), F, Q], axis=1)
    res.to_csv(work / "features.csv", index=False)
    # Ambang kualitas dikalibrasi dari data latih: foto di aplikasi ditolak hanya jika
    # lebih buruk dari 98% foto dataset (bukan angka tebakan).
    import json
    ref = {"blur_min": float(np.percentile(Q.blur, 2)), "bright_min": float(np.percentile(Q.brightness, 1)),
           "bright_max": float(np.percentile(Q.brightness, 99)),
           "glare_max": float(max(0.08, np.nanpercentile(Q.glare, 99)))}
    (work / "quality_ref.json").write_text(json.dumps(ref, indent=2))
    print("Ambang kualitas terkalibrasi:", {k: round(v, 3) for k, v in ref.items()})
    print(f"Selesai: {len(res)} ROI. Foto dengan masalah kualitas: {(Q.issues != '').sum()}")
    print(Q.issues[Q.issues != ""].value_counts().head(10).to_string())
    # Pemeriksaan cepat: apakah fitur warna memang berbeda antara anemia vs normal?
    print("\nRata-rata fitur kunci per kelas (anemia seharusnya a* & erythema index lebih rendah):")
    print(res.groupby("label")[["a_mean", "erythema_index", "high_hue_ratio", "L_mean"]].mean().round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/content/work")
