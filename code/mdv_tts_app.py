# ruff: noqa: E402
# ──────────────────────────────────────────────────────────────────────
# MDV-TTS  —  Multi-Dialect Vietnamese Text-to-Speech Demo
# ──────────────────────────────────────────────────────────────────────
# A Gradio interface for generating Vietnamese speech in three regional
# dialects (North / Central / South) using a fine-tuned F5-TTS model.
#
# Model loading uses the F5TTS API class
# Usage (Colab):
#   1. Mount Google Drive & install F5-TTS  (see run_mdv_tts_demo.ipynb)
#   2. %run /content/drive/MyDrive/mdv-tts/code/mdv_tts_app.py
# ──────────────────────────────────────────────────────────────────────

import os
import sys
import tempfile

import gradio as gr
import numpy as np

from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import save_spectrogram

tempfile_kwargs = {"delete_on_close": False} if sys.version_info >= (3, 12) else {"delete": False}

# ── Path configuration ──────────────────────────────────────────────

DRIVE_ROOT = "/content/drive/MyDrive/mdv-tts"

CHECKPOINT = os.environ.get(
    "MDV_CHECKPOINT",
    os.path.join(DRIVE_ROOT, "F5-TTS/ckpts/viet/model_40000.safetensors"),
)
VOCAB_FILE = os.environ.get(
    "MDV_VOCAB",
    os.path.join(DRIVE_ROOT, "F5-TTS/data/Emilia_ZH_EN_pinyin/vocab.txt"),
)

# ── Dialect config ──────────────────────────────────────────────────

DIALECT_CONFIG = {
    "North": {
        "ref_audio": os.path.join(DRIVE_ROOT, "references/north_ref.wav"),
        "ref_text": (
            "[North] ý tưởng đầu tiên tôi phải nhắc đến là vấn đề về chi phí xây dựng phần mềm biết cốt theo truyền thống."
        ),
    },
    "Central": {
        "ref_audio": os.path.join(DRIVE_ROOT, "references/central_ref.wav"),
        "ref_text": "[Central] các cái ki bốt này được đầu tư từ những năm hai không mười hai không mười ba tức là cách đây cũng khoảng hơn mười năm rồi và quá trình.",
    },
    "South": {
        "ref_audio": os.path.join(DRIVE_ROOT, "references/south_ref.wav"),
        "ref_text": "[South] Chúng tôi tập trung vào cái công tác đào tạo huấn luyện, đặc biệt là huấn luyện về cái quy trình các cái hướng dẫn chẩn đoán điều trị Covid mười chín của Bộ Y tế,",
    },
}

# ── Load model at startup ──────────────────────────────────────────

print(f"[MDV-TTS] Loading checkpoint: {CHECKPOINT}")
print(f"[MDV-TTS] Vocab file:         {VOCAB_FILE}")

tts = F5TTS(
    model="F5TTS_v1_Base",
    ckpt_file=CHECKPOINT,
    vocab_file=VOCAB_FILE,
    device="cuda",
)

print("[MDV-TTS] Model loaded!")


# ── Inference ───────────────────────────────────────────────────────

def infer(ref_audio_path, ref_text, gen_text):
    """Run TTS inference with fixed defaults."""

    if not ref_audio_path or not os.path.exists(ref_audio_path):
        gr.Warning("Reference audio not found.")
        return None, None

    if not gen_text or not gen_text.strip():
        gr.Warning("Please enter text to generate.")
        return None, None

    wav, sr, spec = tts.infer(
        ref_file=ref_audio_path,
        ref_text=ref_text,
        gen_text=gen_text.strip(),
        nfe_step=32,
        cross_fade_duration=0.15,
        speed=1.0,
        seed=None,
        remove_silence=False,
        show_info=print,
    )

    # Save spectrogram
    spectrogram_path = None
    if spec is not None:
        with tempfile.NamedTemporaryFile(suffix=".png", **tempfile_kwargs) as tmp:
            spectrogram_path = tmp.name
        save_spectrogram(spec, spectrogram_path)

    return (sr, wav), spectrogram_path


# ── Helper ──────────────────────────────────────────────────────────

def load_text_from_file(file):
    if file:
        with open(file, "r", encoding="utf-8") as f:
            return gr.update(value=f.read().strip())
    return gr.update(value="")


# ── Gradio UI ───────────────────────────────────────────────────────

with gr.Blocks() as app:
    gr.Markdown(
        """
# MDV-TTS — Multi-Dialect Vietnamese Text-to-Speech

Generate Vietnamese speech in **Northern**, **Central**, or **Southern** dialect using a fine-tuned F5-TTS model.

Select a dialect, enter your text, and click **Synthesize**.

**NOTE: Ensure audio is fully generated before playing. For best results, keep input text at a reasonable length.**
"""
    )

    dialect_radio = gr.Radio(
        choices=["North", "Central", "South"],
        value="North",
        label="Dialect",
        info="Select a regional Vietnamese dialect",
    )

    with gr.Row():
        gen_text_input = gr.Textbox(
            label="Text to Generate",
            lines=10,
            max_lines=40,
            scale=4,
            placeholder="Type or paste Vietnamese text here...",
        )
        gen_text_file = gr.File(
            label="Or load from .txt file",
            file_types=[".txt"],
            scale=1,
        )

    generate_btn = gr.Button("Synthesize", variant="primary")

    audio_output = gr.Audio(label="Synthesized Audio")
    spectrogram_output = gr.Image(label="Spectrogram")

    gen_text_file.upload(
        load_text_from_file,
        inputs=[gen_text_file],
        outputs=[gen_text_input],
    )

    def dialect_tts(dialect, gen_text):
        cfg = DIALECT_CONFIG.get(dialect)
        if cfg is None:
            gr.Warning("Please select a dialect.")
            return None, None

        audio_out, spectrogram_path = infer(
            cfg["ref_audio"],
            cfg["ref_text"],
            gen_text,
        )
        return audio_out, spectrogram_path

    generate_btn.click(
        dialect_tts,
        inputs=[dialect_radio, gen_text_input],
        outputs=[audio_output, spectrogram_output],
    )


# ── Entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    app.queue().launch(share=True)
