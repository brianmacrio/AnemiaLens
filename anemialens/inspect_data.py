"""Langkah 3: lihat isi dataset sebelum training.
Pakai:  python anemialens/inspect_data.py /content/data
Salin seluruh keluarannya ke chat jika ada yang perlu dicek."""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anemialens.common import is_image, looks_like_mask  # noqa: E402

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/content/data")
files = [p for p in root.rglob("*") if p.is_file() and "__MACOSX" not in p.parts and not p.name.startswith("._")]
print(f"Root: {root}\nTotal file: {len(files)}\n")

ext = Counter(p.suffix.lower() for p in files)
print("Ekstensi:", dict(ext.most_common()))

imgs = [p for p in files if is_image(p)]
masks = [p for p in imgs if looks_like_mask(p)]
print(f"Gambar: {len(imgs)} | terdeteksi sebagai mask: {len(masks)} | foto asli: {len(imgs) - len(masks)}\n")

print("Struktur folder (maks. kedalaman 3, jumlah gambar):")
tree = defaultdict(int)
for p in imgs:
    rel = p.relative_to(root).parts[:-1]
    for d in range(1, min(len(rel), 3) + 1):
        tree["/".join(rel[:d])] += 1
children = defaultdict(list)
for k in tree:
    children[k.rsplit("/", 1)[0] if "/" in k else ""].append(k)


def show(parent, depth=0):
    kids = sorted(children.get(parent, []), key=lambda x: (len(x), x))
    for j, k in enumerate(kids):
        if j == 4 and len(kids) > 6:
            print("  " * depth + f"- ... (+{len(kids) - 4} folder lain, total {len(kids)})")
            break
        print("  " * depth + f"- {k.split('/')[-1]}  ({tree[k]} gambar)")
        show(k, depth + 1)


show("")

print("\nContoh nama file (10 pertama):")
for p in imgs[:10]:
    print("  ", p.relative_to(root))

print("\nContoh ukuran gambar:")
for p in imgs[:5]:
    try:
        with Image.open(p) as im:
            print(f"   {p.name}: {im.size} {im.mode}")
    except Exception as e:  # noqa: BLE001
        print(f"   {p.name}: gagal dibuka ({e})")

tables = [p for p in files if p.suffix.lower() in {".csv", ".xlsx", ".xls", ".tsv"}]
print(f"\nTabel/label ditemukan: {len(tables)}")
for t in tables[:10]:
    print(f"\n== {t.relative_to(root)}")
    try:
        if t.suffix.lower() in {".xlsx", ".xls"}:
            sheets = pd.read_excel(t, sheet_name=None)
            for name, df in sheets.items():
                print(f"   sheet '{name}': {df.shape}, kolom = {list(df.columns)}")
                print(df.head(3).to_string(max_cols=12)[:900])
        else:
            df = pd.read_csv(t, sep=None, engine="python")
            print(f"   {df.shape}, kolom = {list(df.columns)}")
            print(df.head(3).to_string(max_cols=12)[:900])
    except Exception as e:  # noqa: BLE001
        print("   gagal dibaca:", e)
