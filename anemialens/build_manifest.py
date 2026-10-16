"""Langkah 4: ubah dataset (struktur apa pun) menjadi satu tabel manifest.csv.

Yang dikenali otomatis:
  * Gaya Eyes-defy-anemia: folder per pasien (1, 2, 3 ...) berisi foto + mask
    *_palpebral.png, dan file Excel/CSV berisi nomor pasien + Hb + jenis kelamin + usia.
  * Gaya folder kelas: .../anemia/xxx.jpg dan .../normal|non_anemic/xxx.jpg
  * Gaya tabel: CSV/Excel berisi nama file gambar + Hb atau label.
Duplikat (gambar sama/hampir sama di beberapa sumber) dideteksi dengan perceptual
hash dan dibuang, supaya tidak bocor antara data latih dan data uji.

Pakai:  python anemialens/build_manifest.py /content/data /content/work
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anemialens.common import anemia_threshold, is_image, looks_like_mask  # noqa: E402

HB_RE = re.compile(r"(^|[^a-z])(hb|hgb|hgb_?g?dl|haemoglobin|hemoglobin|hemoglobin.*|haemo.*)([^a-z]|$)", re.I)
SEX_RE = re.compile(r"sex|gender|jenis.?kelamin|^jk$|^m/f$", re.I)
AGE_RE = re.compile(r"(^|[^a-z])(age|usia|umur)([^a-z]|$)", re.I)
LABEL_RE = re.compile(r"anem|remark|class|label|status|diagnos|kategori|category", re.I)
FILE_RE = re.compile(r"file|image|img|name|nama|path|foto|photo|picture", re.I)


def norm_key(s) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\.(jpe?g|png|bmp|tiff?|webp)$", "", s)
    return s.split("/")[-1].split("\\")[-1]


def id_key(s) -> str:
    """'7', '7.0', 'P007', 'patient 7' -> '7'"""
    m = re.findall(r"\d+", str(s))
    return str(int(m[-1])) if m else str(s).strip().lower()


def parse_label(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if s in {"0", "0.0", "false", "no", "n", "tidak"}:
        return 0
    if s in {"1", "1.0", "true", "yes", "y", "ya"}:
        return 1
    if any(k in s for k in ("non", "not", "normal", "healthy", "sehat", "tidak")):
        return 0
    if "anem" in s or "positive" in s or "pale" in s:
        return 1
    return np.nan


CLASS_NEG = re.compile(r"^(non[-_ ]?an(a)?emi(a|c)|not[-_ ]?an(a)?emi(a|c)|normal|healthy|sehat|negative|0)$", re.I)
CLASS_POS = re.compile(r"^(an(a)?emi(a|c)|anemia[-_ ]?positive|positive|pale|1)$", re.I)


def folder_label(parts):
    """Label dari nama folder HANYA jika nama folder persis nama kelas
    (mis. 'Anemic', 'Non-anemic', 'normal'); 'dataset anemia' tidak dihitung."""
    for part in reversed(parts):
        s = part.strip()
        if CLASS_NEG.match(s):
            return 0
        if CLASS_POS.match(s):
            return 1
    return np.nan


def img_signature(path, size=16):
    """Perceptual hash 16x16 (256 bit) + rata-rata warna + rasio aspek.
    Dua gambar dianggap duplikat hanya jika KETIGANYA sangat mirip, supaya foto
    konjungtiva berbeda pasien yang kebetulan mirip tidak ikut terbuang."""
    try:
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            g = rgb.convert("L").resize((size + 1, size), Image.BILINEAR)
            a = np.asarray(g, dtype=np.int16)
            px = np.asarray(rgb.resize((64, 64)), dtype=np.float32).reshape(-1, 3)
            fg = px[px.sum(1) > 15]
            mean = (fg if len(fg) else px).mean(0)
        return (a[:, 1:] > a[:, :-1]).flatten(), mean, (w, h)
    except Exception:  # noqa: BLE001
        return None


def pick_mask(orig: Path, masks_in_dir):
    """Pilih mask untuk foto: prioritas palpebral murni > forniceal_palpebral > mask lain."""
    if not masks_in_dir:
        return None
    stem = orig.stem.lower()
    same = [m for m in masks_in_dir if m.stem.lower().startswith(stem)] or masks_in_dir

    def rank(m):
        # normalisasi salah ketik di dataset asli ("papebral", "palplebral")
        s = re.sub(r"pal?p?l?ebral", "palpebral", m.stem.lower())
        if "palpebral" in s and "forniceal" not in s:
            return 0
        if "forniceal_palpebral" in s or ("palpebral" in s and "forniceal" in s):
            return 1
        if "forniceal" in s:
            return 3
        return 2
    return sorted(same, key=rank)[0]


def load_tables(root: Path):
    out = []
    for t in root.rglob("*"):
        if t.suffix.lower() not in {".csv", ".xlsx", ".xls", ".tsv"} or "__MACOSX" in t.parts:
            continue
        try:
            if t.suffix.lower() in {".xlsx", ".xls"}:
                for name, df in pd.read_excel(t, sheet_name=None).items():
                    out.append((t, name, df))
            else:
                out.append((t, "", pd.read_csv(t, sep=None, engine="python")))
        except Exception as e:  # noqa: BLE001
            print(f"  [lewati] {t.name}: {e}")
    return out


def find_col(df, regex, exclude=()):
    for c in df.columns:
        if c in exclude:
            continue
        if regex.search(str(c)):
            return c
    return None


def main(data_root, work_dir):
    root, work = Path(data_root), Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    all_imgs = [p for p in root.rglob("*") if p.is_file() and is_image(p)
                and "__MACOSX" not in p.parts and not p.name.startswith("._")]
    masks = [p for p in all_imgs if looks_like_mask(p)]
    origs = [p for p in all_imgs if not looks_like_mask(p)]
    by_dir = defaultdict(list)
    for m in masks:
        by_dir[m.parent].append(m)
    origs_per_dir = defaultdict(int)
    for o in origs:
        origs_per_dir[o.parent] += 1
    print(f"Foto asli: {len(origs)} | mask: {len(masks)}")

    rel_dirs = [o.relative_to(root).parts[:-1] for o in origs]
    common = 0
    if rel_dirs:
        while all(len(r) > common for r in rel_dirs) and len({r[common] for r in rel_dirs}) == 1:
            common += 1

    rows = []
    for o in origs:
        parts = o.relative_to(root).parts[:-1]
        source = parts[common] if len(parts) > common else (parts[-1] if parts else "root")
        # folder per pasien: nama folder berupa angka, atau folder kecil berisi <=6 foto
        dname = o.parent.name
        per_patient_dir = bool(re.fullmatch(r"\d+", dname)) or (
            origs_per_dir[o.parent] <= 6 and o.parent in by_dir)
        if per_patient_dir:
            # sumber = dataset + negara/subset (folder induk pasien), mis. "eyes_defy_anemia/India"
            if len(parts) >= 2 and len(parts) - 1 > common:
                source = f"{source}/{parts[-2]}" if parts[-2] != source else source
            pid = "pasien::" + "/".join(parts)          # path folder pasien = ID unik
        else:
            pid = f"{source}::{o.stem}"
        m = pick_mask(o, by_dir.get(o.parent, []))
        rows.append(dict(image_path=str(o), mask_path=str(m) if m else "", source=source,
                         patient_id=pid, patient_dir=str(o.parent) if per_patient_dir else "",
                         folder_label=folder_label(parts), hb=np.nan, sex=np.nan, age=np.nan,
                         table_label=np.nan))
    df = pd.DataFrame(rows)
    df["sex"] = df["sex"].astype(object)
    df["mask_kind"] = np.where(df.mask_path != "", "manual", "")
    # Foto yang SUDAH tersegmentasi dengan latar transparan (mis. CP-AnemiC):
    # kanal alfa dijadikan mask. Mask ini tidak dipakai untuk melatih U-Net.
    amask_dir = work / "alpha_masks"
    n_alpha = 0
    for i in df.index[df.mask_path == ""]:
        try:
            with Image.open(df.at[i, "image_path"]) as im:
                if im.mode not in ("RGBA", "LA", "PA") and "transparency" not in im.info:
                    continue
                a = np.asarray(im.convert("RGBA"))[:, :, 3]
        except Exception:  # noqa: BLE001
            continue
        if 0.05 < (a == 0).mean() < 0.98:
            amask_dir.mkdir(exist_ok=True)
            mp = amask_dir / f"{i:06d}.png"
            Image.fromarray(((a > 0) * 255).astype(np.uint8)).save(mp)
            df.at[i, "mask_path"], df.at[i, "mask_kind"] = str(mp), "alpha"
            n_alpha += 1
    if n_alpha:
        print(f"  {n_alpha} foto sudah tersegmentasi (latar transparan) -> mask dari kanal alfa")
    if df.empty:
        sys.exit("Tidak ada gambar ditemukan. Cek path dataset.")

    # ------------------------------------------------------------ gabung tabel
    stem_idx = defaultdict(list)
    for i, p in enumerate(df.image_path):
        stem_idx[norm_key(p)].append(i)
    for t, sheet, tab in load_tables(root):
        tab = tab.dropna(how="all")
        if tab.empty:
            continue
        hb_c = find_col(tab, HB_RE)
        sex_c = find_col(tab, SEX_RE)
        age_c = find_col(tab, AGE_RE, exclude=(sex_c,))
        lab_c = find_col(tab, LABEL_RE, exclude=(hb_c, sex_c, age_c))
        if hb_c is None and lab_c is None:
            continue
        best = (0, None, None)
        # a) gabung lewat nama file
        for c in tab.columns:
            keys = tab[c].astype(str).map(norm_key)
            hits = sum(k in stem_idx for k in keys)
            if hits > best[0]:
                best = (hits, c, "file")
        # b) gabung lewat nomor pasien = nama folder pasien di bawah lokasi tabel
        under = df[df.image_path.str.startswith(str(t.parent)) & (df.patient_dir != "")]
        dir_ids = defaultdict(list)
        for i, d in zip(under.index, under.patient_dir):
            dir_ids[id_key(Path(d).name)].append(i)
        if dir_ids:
            for c in tab.columns:
                if c in (hb_c, sex_c, age_c):
                    continue
                keys = tab[c].map(id_key)
                hits = sum(k in dir_ids for k in keys)
                if hits > best[0]:
                    best = (hits, c, "id")
        hits, key_c, mode = best
        if hits < max(3, 0.3 * min(len(tab), 50)):
            print(f"  [tabel tidak tertaut] {t.name}{'/' + sheet if sheet else ''}: kolom Hb={hb_c}, label={lab_c}")
            continue
        print(f"  [tabel] {t.name}{'/' + sheet if sheet else ''}: tertaut lewat '{key_c}' ({mode}, {hits} baris) "
              f"| Hb='{hb_c}' sex='{sex_c}' age='{age_c}' label='{lab_c}'")
        for _, r in tab.iterrows():
            idxs = stem_idx.get(norm_key(r[key_c]), []) if mode == "file" else dir_ids.get(id_key(r[key_c]), [])
            for i in idxs:
                if hb_c is not None:
                    v = pd.to_numeric(str(r[hb_c]).replace(",", "."), errors="coerce")
                    if pd.notna(v) and 2 < v < 25:
                        df.at[i, "hb"] = float(v)
                if sex_c is not None and pd.notna(r[sex_c]):
                    df.at[i, "sex"] = str(r[sex_c]).strip()
                if age_c is not None:
                    df.at[i, "age"] = pd.to_numeric(r[age_c], errors="coerce")
                if lab_c is not None:
                    df.at[i, "table_label"] = parse_label(r[lab_c])

    # ------------------------------------------------------------ label akhir
    # Dataset balita (mis. CP-AnemiC, 6-59 bulan) mencatat usia dalam BULAN.
    # Deteksi: semua usia di satu sumber <= 60 dan ada yang > 18 -> anggap bulan.
    df["age_years"] = df.age
    for src, g in df.groupby("source"):
        a = g.age.dropna()
        if len(a) and a.max() <= 60 and a.min() < 12 and (a > 18).any():
            df.loc[g.index, "age_years"] = g.age / 12
            print(f"  [usia] sumber '{src}' dianggap dalam bulan (maks {a.max():.0f}).")

    def final_label(r):
        if pd.notna(r.hb):
            return int(r.hb < anemia_threshold(r.age_years, r.sex)), "hb_lab"
        if pd.notna(r.table_label):
            return int(r.table_label), "tabel"
        if pd.notna(r.folder_label):
            return int(r.folder_label), "folder"
        return -1, "tidak_ada"
    lab = df.apply(final_label, axis=1, result_type="expand")
    df["label"], df["label_source"] = lab[0], lab[1]

    # ------------------------------------------------------------ duplikat
    print("Mengecek duplikat (perceptual hash)...")
    sigs = [img_signature(p) for p in df.image_path]
    ok = [i for i, h in enumerate(sigs) if h is not None]
    H = np.array([sigs[i][0] for i in ok], dtype=bool)
    MC = np.array([sigs[i][1] for i in ok], dtype=np.float32)
    WH = np.array([sigs[i][2] for i in ok], dtype=np.float32)
    drop, group_of = set(), {}
    for a in range(len(ok)):
        ia = ok[a]
        if ia in drop:
            continue
        d = (H[a + 1:] != H[a]).sum(axis=1)
        dc = np.abs(MC[a + 1:] - MC[a]).max(axis=1)
        dwh = np.abs(WH[a + 1:] - WH[a]).max(axis=1)
        # duplikat = ukuran sama, struktur hampir identik, warna latar-depan hampir identik
        for b in np.where((d <= 3) & (dc < 1.5) & (dwh <= 2))[0]:
            ib = ok[a + 1 + b]
            if ib not in drop:
                drop.add(ib)
                group_of[ib] = ia
    # Kelompokkan duplikat. Jika satu gambar yang SAMA dipakai untuk beberapa pasien
    # dengan Hb/label berbeda, pasangan gambar-Hb itu tidak bisa dipercaya -> buang semua.
    groups = defaultdict(list)
    for ib, ia in group_of.items():
        groups[ia].append(ib)
    df["duplicate_of"], df["drop_reason"] = "", ""
    n_incons = 0
    for ia, members in groups.items():
        allm = [ia] + members
        hbv = df.loc[allm, "hb"].dropna()
        incons = (df.loc[allm, "label"].nunique() > 1) or (len(hbv) > 1 and hbv.max() - hbv.min() > 0.5)
        for ib in members:
            df.at[ib, "duplicate_of"] = df.at[ia, "image_path"]
            df.at[ib, "drop_reason"] = "duplikat"
        if incons:
            n_incons += 1
            for i in allm:
                df.at[i, "drop_reason"] = "gambar_sama_hb_berbeda"
    dups = df[df.drop_reason != ""]
    df = df[df.drop_reason == ""].reset_index(drop=True)
    print(f"  grup gambar identik: {len(groups)} | dibuang total: {len(dups)} foto")
    if n_incons:
        print(f"  PERHATIAN: {n_incons} grup berisi gambar yang sama untuk pasien dengan Hb berbeda. "
              f"Seluruh anggota grup dibuang ({(dups.drop_reason == 'gambar_sama_hb_berbeda').sum()} foto).")
        print("  Per sumber:", dups.groupby('source').size().to_dict())
    dups.to_csv(work / "duplicates.csv", index=False)

    unlabeled = (df.label < 0).sum()
    df = df[df.label >= 0].reset_index(drop=True)
    df.to_csv(work / "manifest.csv", index=False)

    print("\n================ RINGKASAN MANIFEST ================")
    print(f"Total foto berlabel: {len(df)} (tanpa label dibuang: {unlabeled})")
    print(f"Pasien unik: {df.patient_id.nunique()} | punya mask: {(df.mask_path != '').sum()} | punya Hb: {df.hb.notna().sum()}")
    summ = df.groupby("source").agg(foto=("image_path", "size"), pasien=("patient_id", "nunique"),
                                    anemia=("label", "sum"), mask=("mask_path", lambda s: (s != "").sum()),
                                    hb=("hb", lambda s: s.notna().sum()), hb_mean=("hb", "mean"))
    print(summ.to_string())
    print("\nSumber label:", df.label_source.value_counts().to_dict())
    print(f"\nTersimpan: {work / 'manifest.csv'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/content/data",
         sys.argv[2] if len(sys.argv) > 2 else "/content/work")
