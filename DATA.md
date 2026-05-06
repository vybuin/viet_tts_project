# Data Extraction Pipeline

This document describes the current ViMD data extraction workflow used for the multi-dialect Vietnamese TTS project.

The extraction pipeline lives in:

```text
code/extract_data.ipynb
```

The source dataset is the Vietnamese Multi-Dialect dataset on Hugging Face:

```text
https://huggingface.co/datasets/nguyendv02/ViMD_Dataset
```

## Overview

The goal is to build a usable Vietnamese TTS training subset from ViMD without downloading and manually processing the entire dataset. The notebook streams ViMD from Hugging Face, keeps clips from the three target regions, normalizes audio, and writes a manifest that can be used by the F5-TTS fine-tuning notebook.

The current extraction setup targets:

- `North`
- `Central`
- `South`

Audio is saved as `.wav` and resampled to 24 kHz for F5-TTS.

## Recommended Runtime

Run `code/extract_data.ipynb` in Google Colab or another high-RAM environment. The notebook can stream a large amount of audio, run VAD, and optionally run PhoWhisper on long clips, so a typical laptop may run out of memory or take a long time.

If using Colab, the notebook writes to both:

```text
/content/viet_tts_project_data/
/content/drive/MyDrive/viet_tts_project_data/
```

The local Colab path is used for faster processing. The Google Drive path is used to preserve outputs after the Colab session ends.

## Current Notebook Configuration

The main configuration cell in `extract_data.ipynb` controls the extraction. Important settings include:

```text
DATASET_NAME = "nguyendv02/ViMD_Dataset"
HOURS_PER_REGION = 25
MIN_DURATION = 3.0
MAX_DURATION = 17.0
TARGET_SAMPLE_RATE = 24000
REGIONS = ["North", "Central", "South"]
```

The notebook currently uses:

```text
ASR_BACKEND = "phowhisper"
PHOWHISPER_MODEL = "vinai/PhoWhisper-base"
SEGMENT_LONG_CLIPS = True
MAX_SOURCE_DURATION = 30.0
```

Short clips within the duration range are kept directly. Longer clips can be segmented with Silero VAD and re-transcribed with PhoWhisper so that usable portions can still be included.

## Extraction Strategy

### 1. Stream ViMD

The notebook streams the dataset directly from Hugging Face instead of downloading the full dataset first. This keeps the workflow more practical for Colab and avoids storing the full 100+ hour dataset locally.

### 2. Filter Examples

Rows are filtered to keep usable examples only:

- valid target region: `North`, `Central`, or `South`
- non-empty transcript
- readable audio
- minimum duration of 3 seconds
- standard short-clip maximum duration of 17 seconds
- optional source maximum duration of 30 seconds for long-clip segmentation

Vietnamese diacritics are preserved in the transcript text.

### 3. Normalize Audio

Audio is converted to mono when needed and resampled to:

```text
24000 Hz
```

This matches the expected sample rate for F5-TTS training and inference.

### 4. Handle Long Clips

For clips longer than the direct training range, the notebook can:

- detect speech intervals using Silero VAD
- split the source audio into shorter segments
- transcribe those segments with PhoWhisper
- save aligned segment audio and generated transcripts

This step helps recover usable training material from longer ViMD recordings while keeping each training example within a TTS-friendly duration.

## Output Format

The extracted dataset is written under an output directory such as:

```text
data/vimd_subset/
```

or, in Colab:

```text
/content/viet_tts_project_data/vimd_subset_15s_2/
```

The audio directory follows this structure:

```text
audio/
├── train/
│   ├── North/
│   ├── Central/
│   └── South/
├── valid/
│   ├── North/
│   ├── Central/
│   └── South/
└── test/
    ├── North/
    ├── Central/
    └── South/
```

Depending on the notebook configuration, some runs may only extract the `train` split.

## Manifests

Manifests are saved under:

```text
manifests/
```

The main files are:

```text
manifest.csv
manifest.jsonl
manifest_original.csv
extract_checkpoint.json
```

The primary manifest columns are:

- `audio_path`
- `text`
- `duration_sec`
- `region`
- `split`
- `filename`

Some extraction runs may include extra metadata columns for segmented clips, such as original duration, segment index, or ASR source.

## Resume and Checkpoints

The notebook writes extraction progress to:

```text
manifests/extract_checkpoint.json
```

To resume an interrupted run, set this in the notebook configuration cell:

```text
RESUME_FROM_CHECKPOINT = True
```

The checkpoint stores accepted rows, accepted duration totals, segment candidates, and the last streamed position for each split. Resume only works reliably when the extraction settings match the settings used to create the checkpoint.

## How to Use the Extracted Data

After extraction, use the generated manifest and audio folders in:

```text
code/fine_tune_f5.ipynb
```

That notebook prepares the data for F5-TTS fine-tuning. If running in Colab, make sure the fine-tuning notebook points to the same Drive output directory created by the extraction notebook.
