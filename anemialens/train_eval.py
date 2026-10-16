"""Langkah 7: validasi silang 5-fold BERBASIS PASIEN untuk 3 model + ensemble.
Semua angka di laporan adalah prediksi out-of-fold (gambar tidak pernah dinilai oleh
model yang melihatnya saat training), ditambah uji leave-one-source-out untuk
mengukur seberapa baik model bertahan di dataset/negara yang belum pernah dilihat.

Pakai:  python anemialens/train_eval.py /content/work            (lengkap)
        python anemialens/train_eval.py /content/work --no-cnn   (cepat, tanpa CNN)
Hasil:  work/oof.csv, work/thresholds.json, work/report.md, work/roc.png, work/hb_scatter.png
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anemialens.common import COLOR_FEATURE_NAMES, Config, seed_everything  # noqa: E402
from anemialens.models import (build_dino, dino_embed, make_clf, make_reg, make_rf_clf,  # noqa: E402
                               make_rf_reg, predict_cnn, train_cnn)


warnings.filterwarnings("ignore")


def inner_splits(groups, y, k=3):
    k = min(k, len(np.unique(groups)))
    return list(GroupKFold(n_splits=k).split(np.zeros(len(y)), y, groups))


def fit_predict_sklearn(Xtr, ytr, gtr, hbtr, Xte, kind="linear"):
    clf = make_rf_clf() if kind == "rf" else make_clf(inner_splits(gtr, ytr))
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    hb = np.full(len(Xte), np.nan)
    m = ~np.isnan(hbtr)
    if m.sum() >= 20:
        reg = (make_rf_reg() if kind == "rf" else make_reg()).fit(Xtr[m], hbtr[m])
        hb = reg.predict(Xte)
    return p, hb


def thr_for_sens(y, p, target):
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target)[0]
    return float(thr[ok[0]]) if len(ok) else 0.5


def thr_for_spec(y, p, target=0.90):
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(1 - fpr >= target)[0]
    return float(thr[ok[-1]]) if len(ok) else 0.5


def metrics(y, p, thr, hb_true=None, hb_pred=None):
    out = {"n": int(len(y)), "anemia": int(y.sum())}
    if len(np.unique(y)) == 2:
        out["auc"] = roc_auc_score(y, p)
    pred = (p >= thr).astype(int)
    tp, tn = ((pred == 1) & (y == 1)).sum(), ((pred == 0) & (y == 0)).sum()
    fp, fn = ((pred == 1) & (y == 0)).sum(), ((pred == 0) & (y == 1)).sum()
    out.update(sens=tp / max(1, tp + fn), spec=tn / max(1, tn + fp), ppv=tp / max(1, tp + fp),
               npv=tn / max(1, tn + fn), acc=(tp + tn) / len(y))
    out["bal_acc"] = (out["sens"] + out["spec"]) / 2
    if hb_true is not None and hb_pred is not None:
        m = ~np.isnan(hb_true) & ~np.isnan(hb_pred)
        if m.sum() >= 5:
            e = hb_pred[m] - hb_true[m]
            out.update(hb_n=int(m.sum()), hb_mae=float(np.abs(e).mean()),
                       hb_r=float(np.corrcoef(hb_true[m], hb_pred[m])[0, 1]),
                       hb_within1=float((np.abs(e) <= 1).mean()), hb_within2=float((np.abs(e) <= 2).mean()))
    return out


def md(d, index=True):
    try:
        return d.to_markdown(index=index, floatfmt=".3f")
    except ImportError:
        return d.to_string(index=index)


def bootstrap_auc_ci(y, p, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    aucs = []
    for _ in range(n):
        s = np.concatenate([idx_by_g[g] for g in rng.choice(ug, len(ug), replace=True)])
        if len(np.unique(y[s])) == 2:
            aucs.append(roc_auc_score(y[s], p[s]))
    return np.percentile(aucs, [2.5, 97.5]) if aucs else (np.nan, np.nan)


def main(work_dir, use_cnn=True):
    work = Path(work_dir)
    cfg = Config.load(work / "config.json") if (work / "config.json").exists() else Config(work_dir=str(work))
    seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(work / "features.csv", keep_default_na=False, na_values=[""])
    df["mask_path"] = df.mask_path.fillna("")
    y = df.label.values.astype(int)
    hb = pd.to_numeric(df.hb, errors="coerce").values.astype(float)
    groups = df.patient_id.astype(str).values
    src = df.source.astype(str).values
    Xc = df[COLOR_FEATURE_NAMES].values.astype(np.float32)
    print(f"Data: {len(df)} foto, {len(np.unique(groups))} pasien, anemia {y.mean():.1%}, punya Hb {np.isfinite(hb).sum()}")

    # --------------------------------------------------- embedding DINOv2 (sekali)
    emb_path = work / "dino_emb.npy"
    if emb_path.exists() and np.load(emb_path).shape[0] == len(df):
        Xd = np.load(emb_path)
    else:
        print("Menghitung embedding DINOv2...")
        Xd = dino_embed(build_dino(cfg.dino_backbone, cfg.pretrained, cfg.img_size), df.roi_path.tolist(),
                        cfg.img_size, device)
        np.save(emb_path, Xd)

    # --------------------------------------------------- 5-fold berbasis pasien
    strat = np.array([f"{a}_{b}" for a, b in zip(y, src)])
    k = min(cfg.n_folds, len(np.unique(groups)))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=cfg.seed)
    names = ["color", "color_rf", "dino", "cnn"]
    oof = {m: np.full(len(df), np.nan) for m in names}
    oof_hb = {m: np.full(len(df), np.nan) for m in names}
    fold_id = np.full(len(df), -1)
    for f, (tr, te) in enumerate(sgkf.split(Xc, strat, groups)):
        fold_id[te] = f
        print(f"\nFold {f + 1}/{k}: latih {len(tr)} | uji {len(te)}")
        oof["color"][te], oof_hb["color"][te] = fit_predict_sklearn(Xc[tr], y[tr], groups[tr], hb[tr], Xc[te])
        oof["color_rf"][te], oof_hb["color_rf"][te] = fit_predict_sklearn(Xc[tr], y[tr], groups[tr], hb[tr], Xc[te], "rf")
        oof["dino"][te], oof_hb["dino"][te] = fit_predict_sklearn(Xd[tr], y[tr], groups[tr], hb[tr], Xd[te])
        print(f"  AUC fold  warna={roc_auc_score(y[te], oof['color'][te]):.3f}  "
              f"warna_rf={roc_auc_score(y[te], oof['color_rf'][te]):.3f}  "
              f"dino={roc_auc_score(y[te], oof['dino'][te]):.3f}" if len(np.unique(y[te])) == 2 else "")
        if use_cnn:
            model = train_cnn(cfg, df.roi_path.values[tr], y[tr], hb[tr], device, log_prefix=f"[cnn f{f + 1}] ")
            oof["cnn"][te], oof_hb["cnn"][te] = predict_cnn(model, df.roi_path.values[te].tolist(), cfg.img_size, device)
            if len(np.unique(y[te])) == 2:
                print(f"  AUC fold  cnn={roc_auc_score(y[te], oof['cnn'][te]):.3f}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    avail = [m for m in oof if not np.isnan(oof[m]).all()]
    aucs = {m: roc_auc_score(y, oof[m]) for m in avail}
    best = max(aucs.values())
    # Anggota ensemble: model yang jelas berguna (AUC >= 0.60 dan tidak jauh dari terbaik).
    used = [m for m in avail if aucs[m] >= 0.60 and aucs[m] >= best - 0.08] or [max(aucs, key=aucs.get)]
    print("\nAUC out-of-fold:", {m: round(a, 3) for m, a in aucs.items()}, "-> anggota ensemble:", used)
    oof["ensemble"] = np.mean([oof[m] for m in used], axis=0)
    hb_stack = np.array([oof_hb[m] for m in used])
    oof_hb["ensemble"] = np.nanmean(hb_stack, axis=0) if np.isfinite(hb_stack).any() else np.full(len(df), np.nan)

    # --------------------------------------------------- ambang & metrik
    thr_s = thr_for_sens(y, oof["ensemble"], cfg.target_sensitivity)
    thr_p = max(thr_for_spec(y, oof["ensemble"], 0.90), thr_s)
    rows = []
    for m in avail + ["ensemble"]:
        t = thr_for_sens(y, oof[m], cfg.target_sensitivity)
        r = metrics(y, oof[m], t, hb, oof_hb[m])
        r["model"] = m
        rows.append(r)
    res = pd.DataFrame(rows).set_index("model")
    lo, hi = bootstrap_auc_ci(y, oof["ensemble"], groups)
    per_src = []
    for s in np.unique(src):
        mm = src == s
        r = metrics(y[mm], oof["ensemble"][mm], thr_s, hb[mm], oof_hb["ensemble"][mm])
        r["source"] = s
        per_src.append(r)
    per_src = pd.DataFrame(per_src).set_index("source")

    # --------------------------------------------------- leave-one-source-out
    loso = []
    if len(np.unique(src)) > 1:
        print("\nUji leave-one-source-out (latih di sumber lain, uji di satu sumber)...")
        for s in np.unique(src):
            te, tr = src == s, src != s
            if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
                continue
            row = {"source_uji": s, "n": int(te.sum())}
            ps = []
            for name, X, kind in (("color", Xc, "linear"), ("color_rf", Xc, "rf"), ("dino", Xd, "linear")):
                p, _ = fit_predict_sklearn(X[tr], y[tr], groups[tr], hb[tr], X[te], kind)
                row[f"auc_{name}"] = roc_auc_score(y[te], p)
                ps.append(p)
            row["auc_rata2"] = roc_auc_score(y[te], np.mean(ps, axis=0))
            loso.append(row)
    loso = pd.DataFrame(loso)

    # --------------------------------------------------- simpan
    out = df[["image_path", "source", "patient_id", "label", "hb"]].copy()
    out["fold"] = fold_id
    for m in oof:
        out[f"p_{m}"] = oof[m]
        out[f"hb_{m}"] = oof_hb.get(m, np.nan)
    out.to_csv(work / "oof.csv", index=False)
    thresholds = {"thr_sensitive": thr_s, "thr_specific": thr_p, "target_sensitivity": cfg.target_sensitivity,
                  "models": used, "auc_ensemble": float(res.loc["ensemble", "auc"]),
                  "auc_ci95": [float(lo), float(hi)],
                  "hb_mae": float(res.loc["ensemble", "hb_mae"]) if "hb_mae" in res.columns
                  and pd.notna(res.loc["ensemble", "hb_mae"]) else None}
    (work / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    cols = [c for c in ["n", "anemia", "auc", "sens", "spec", "ppv", "npv", "bal_acc",
                        "hb_mae", "hb_r", "hb_within1", "hb_within2"] if c in res.columns]
    report = [
        "# Laporan validasi AnemiaLens (out-of-fold, split per pasien)\n",
        f"Data: {len(df)} foto, {len(np.unique(groups))} pasien, prevalensi anemia {y.mean():.1%}.\n",
        f"Anggota ensemble: {', '.join(used)} (dipilih dari AUC out-of-fold; model lain hanya dilaporkan).\n",
        f"AUC ensemble: **{res.loc['ensemble', 'auc']:.3f}** (95% CI bootstrap per pasien {lo:.3f}–{hi:.3f}).\n",
        f"Ambang skrining (sensitivitas ≥ {cfg.target_sensitivity:.0%}): {thr_s:.3f}; "
        f"ambang risiko tinggi (spesifisitas ≥ 90%): {thr_p:.3f}.\n",
        "\n## Per model\n", md(res[cols]), "\n\n## Ensemble per sumber data\n",
        md(per_src[[c for c in cols if c in per_src.columns]]),
    ]
    if len(loso):
        report += ["\n\n## Leave-one-source-out (proksi validasi eksternal)\n", md(loso, index=False)]
    (work / "report.md").write_text("\n".join(report))
    print("\n" + "\n".join(report))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        for m in avail + ["ensemble"]:
            fpr, tpr, _ = roc_curve(y, oof[m])
            ax.plot(fpr, tpr, lw=2.5 if m == "ensemble" else 1.2, label=f"{m} (AUC {roc_auc_score(y, oof[m]):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set(xlabel="1 - spesifisitas", ylabel="sensitivitas", title="ROC out-of-fold")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(work / "roc.png", dpi=150)
        m = np.isfinite(hb) & np.isfinite(oof_hb["ensemble"])
        if m.sum() >= 5:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(hb[m], oof_hb["ensemble"][m], s=10, alpha=0.6)
            lims = [min(hb[m].min(), oof_hb["ensemble"][m].min()) - 0.5, max(hb[m].max(), oof_hb["ensemble"][m].max()) + 0.5]
            ax.plot(lims, lims, "k--", lw=0.8)
            ax.set(xlabel="Hb lab (g/dL)", ylabel="Hb prediksi (g/dL)", title="Estimasi Hb out-of-fold")
            fig.tight_layout()
            fig.savefig(work / "hb_scatter.png", dpi=150)
    except Exception as e:  # noqa: BLE001
        print("Plot dilewati:", e)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "/content/work",
         use_cnn="--no-cnn" not in sys.argv)
