"""Langkah 10: demo web AnemiaLens (Gradio) dengan link publik sementara dari Colab.

Pakai:  python anemialens/app.py /content/work/model
"""
import sys
from pathlib import Path

import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anemialens.predict import AnemiaLens  # noqa: E402

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "/content/work/model"
lens = AnemiaLens(MODEL_DIR)

GUIDE = """
### Cara mengambil foto
1. Cari cahaya terang yang merata (dekat jendela atau lampu putih). **Jangan pakai flash.**
2. **Tarik kelopak mata bawah** dengan jari sampai bagian dalamnya yang **merah muda** terlihat jelas (seperti saat diperiksa dokter).
3. Foto **dari dekat**: mata dan kelopak bawah memenuhi sebagian besar layar. Ketuk layar untuk fokus.
4. Jika area hijau meleset, pakai ikon **crop** di editor untuk memotong tepat di bagian dalam kelopak, lalu centang *mode crop manual*.
5. (Disarankan) Letakkan **kartu/kertas putih polos** di samping mata agar warna dikoreksi otomatis.
"""
COLOR = {"tinggi": "#B91C1C", "sedang": "#B45309", "rendah": "#15803D"}


def run(img, use_card, manual):
    if img is None:
        return None, "Unggah atau ambil foto terlebih dahulu."
    if isinstance(img, dict):                    # keluaran ImageEditor (setelah crop)
        img = img.get("composite") if img.get("composite") is not None else img.get("background")
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    r = lens.analyze(img, use_card=use_card, manual_crop=manual)
    if r["status"] != "ok":
        tips = "\n".join(f"- {a}" for a in r["advice"])
        return r["overlay"], f"### Foto belum bisa dinilai\n{tips}\n\n_{r['disclaimer']}_"
    hb = ""
    if r["hb_estimate"] is not None:
        err = f" ± {r['hb_mae']:.1f}" if r.get("hb_mae") else ""
        hb = f"\n\n**Perkiraan Hb:** {r['hb_estimate']:.1f}{err} g/dL (perkiraan kasar, bukan hasil lab)"
    per = " · ".join(f"{k}: {v:.2f}" for k, v in r["per_model"].items())
    warn = "".join(f"\n- {w}" for w in r["quality_warnings"])
    md = (f"## <span style='color:{COLOR[r['risk_tier']]}'>Risiko {r['risk_tier'].upper()}</span>\n"
          f"{r['message']}{hb}\n\n"
          f"Skor ensemble: {r['probability']:.2f} (per model: {per})\n"
          f"Koreksi warna kartu: {'ya' if r['card_used'] else 'tidak'} · area: {r['roi_mode']}"
          + (f"\n\n**Catatan kualitas:**{warn}" if warn else "")
          + f"\n\n---\n_{r['disclaimer']}_")
    return r["overlay"], md


with gr.Blocks(title="AnemiaLens") as demo:
    gr.Markdown("# AnemiaLens\nSkrining awal anemia dari foto bagian dalam kelopak mata bawah (konjungtiva).")
    with gr.Row():
        with gr.Column():
            inp = gr.ImageEditor(label="Foto mata (ikon crop untuk memotong manual)", type="numpy",
                                 sources=["upload", "webcam"], brush=False, eraser=False)
            card = gr.Checkbox(label="Ada kartu putih di dalam foto (koreksi warna)", value=False)
            manual = gr.Checkbox(label="Saya sudah crop foto tepat di bagian dalam kelopak bawah (mode crop manual)",
                                 value=False)
            btn = gr.Button("Analisis", variant="primary")
            gr.Markdown(GUIDE)
        with gr.Column():
            out_img = gr.Image(label="Area yang dianalisis (garis hijau)")
            out_md = gr.Markdown()
    btn.click(run, [inp, card, manual], [out_img, out_md])

if __name__ == "__main__":
    demo.launch(share=True, debug=False)
