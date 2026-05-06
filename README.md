# Multi-Dialect Vietnamese TTS

This repository contains our project on multi-dialect Vietnamese text-to-speech (TTS). We fine-tune F5-TTS on a curated subset of the Vietnamese Multi-Dialect (ViMD) dataset, then evaluate whether generated speech preserves Northern, Central, and Southern Vietnamese dialect characteristics.

## Start Here

Project webpage and demo walkthrough: [View here](https://vybuin.github.io/viet_tts_project/)

The website shows the project results, audio samples, and the Gradio demo workflow. The rest of this README explains how to reproduce the pipeline if you want to run the project yourself.

## Repository Layout

```text
.
├── code/
│   ├── extract_data.ipynb          # Builds the ViMD subset used for training/eval
│   ├── fine_tune_f5.ipynb          # Fine-tunes F5-TTS on the extracted data
│   ├── inference_f5.ipynb          # Generates samples from trained checkpoints
│   ├── mdv_tts_app.py              # Gradio demo app
│   ├── tests/wer_eval.ipynb        # WER evaluation notebook
│   └── mdv-tts/
│       ├── F5-TTS/                 # Local F5-TTS codebase
│       └── references/             # Reference clips for dialect-conditioned inference
├── data/vimd_subset/               # Extracted ViMD subset and manifests
├── docs/                           # Project webpage and demo assets
├── samples/                        # Example generated/reference audio
├── DATA.md                         # More detail on the data extraction pipeline
└── requirements.txt
```

## What This Project Does

The pipeline has four main stages:

1. Extract a balanced ViMD subset across Northern, Central, and Southern dialects.
2. Prepare the F5-TTS directory and checkpoints.
3. Fine-tune F5-TTS on the extracted Vietnamese speech data.
4. Run inference and evaluate generated speech with WER and listening samples.

## Setup

Clone the repository:

```bash
git clone https://github.com/vybuin/viet_tts_project.git
cd viet_tts_project
```

Create an environment. Python 3.10 or 3.11 is recommended because the F5-TTS codebase depends on modern PyTorch/audio tooling.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For fine-tuning or GPU inference, install a PyTorch build that matches your machine and CUDA version. The upstream F5-TTS README in `code/mdv-tts/F5-TTS/README.md` has more detailed installation guidance.

We recommend running the notebooks with a Google Colab kernel or another high-RAM GPU environment. Data extraction, fine-tuning, and inference can exceed the memory and compute limits of a typical laptop.

## Running the Project

### 1. Extract the Data

Source dataset: https://huggingface.co/datasets/nguyendv02/ViMD_Dataset

Open and run:

```text
code/extract_data.ipynb
```

This notebook streams ViMD from Hugging Face, filters usable clips, balances the subset by dialect, and writes audio plus manifests under:

```text
data/vimd_subset/
```

For more detail on the extraction logic, see `DATA.md`.

### 2. Prepare F5-TTS and Checkpoints

The expected project structure is:

```text
code/mdv-tts/
├── F5-TTS/
│   ├── ckpts/
│   └── src/f5_tts/
└── references/
```

UMN students with access to our shared project materials can copy the prepared `mdv-tts` directory, including checkpoints, into `code/mdv-tts/`.

Non-UMN users can still run the project from scratch by using the included F5-TTS code and downloading the public pretrained F5-TTS checkpoints from Hugging Face:

```text
https://huggingface.co/SWivid/F5-TTS
```

The fine-tuned project checkpoints are not publicly bundled in this repository.

### 3. Fine-Tune F5-TTS

Open and run:

```text
code/fine_tune_f5.ipynb
```

This notebook prepares the Vietnamese vocabulary/data format and fine-tunes F5-TTS on the extracted ViMD subset. Training is GPU-heavy. Our project training used an NVIDIA RTX PRO 6000 Blackwell GPU and ran for roughly 14 hours, so expect this step to be impractical on CPU.

### 4. Run Inference

Open and run:

```text
code/inference_f5.ipynb
```

This notebook loads a checkpoint, selects a dialect reference clip, and generates Vietnamese speech for target text. Dialect identity is controlled by the reference audio rather than by a separate dialect label.

Reference clips are stored in:

```text
code/mdv-tts/references/
```

### 5. Optional: Run the Demo App

The repository also includes a Gradio demo:

```bash
python code/mdv_tts_app.py
```

This requires a usable F5-TTS installation and the expected checkpoints/reference files. If the app fails to load, first confirm that the `code/mdv-tts/` structure matches the layout above.

## Evaluation

WER evaluation is in:

```text
code/tests/wer_eval.ipynb
```

For the reported WER table, we sampled 50 transcriptions from the ViMD test split, generated audio for each transcription across dialects and checkpoints, transcribed the generated samples with PhoWhisper-medium, and compared those transcripts with the ground-truth text.

## Notes and Limitations

- The extracted dataset is a subset of ViMD, not the full dataset.
- Fine-tuned checkpoints may require UMN-only access depending on the checkpoint source.
- Running from scratch requires substantial GPU resources.
- Generated speech quality depends heavily on transcript quality, dataset size, and the selected reference audio.
- This project is intended for research and class demonstration, not production voice cloning.

## Acknowledgements

This project builds on F5-TTS and the ViMD dataset. The original F5-TTS code and pretrained checkpoints are available from the upstream project and Hugging Face.
