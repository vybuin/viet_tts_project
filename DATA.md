# Data Extraction Pipeline

## Overview
We build a **balanced, efficient subset** of the ViMD Dataset for training a multi-dialect Vietnamese TTS model.

Instead of downloading the full dataset (~100+ hours), we use a **streaming pipeline** to extract only the data we need. This keeps the workflow lightweight, reproducible, and easy to scale.


## Design Goals
- Balance dialects evenly across:
  - North
  - Central
  - South  
- Preserve train/valid/test splits
- Limit speaker dominance (cap utterances per speaker)
- Avoid unnecessary storage usage
- Enable easy scaling (debug → baseline → larger experiments)


## Extraction Strategy

### Streaming (No Full Download)
We stream the dataset directly from Hugging Face instead of downloading all audio files locally.

Audio decoding is handled manually using `soundfile` rather than Hugging Face's built-in audio decoding.

**Why:**
- avoids large storage usage  
- avoids dependency issues (e.g., torchcodec, FFmpeg)  
- faster iteration  
- allows flexible subsetting  


### Balanced by Region and Split
We enforce **equal target duration per region within each split**.

Example (3 hours per region):
- Train: 2.4h per region  
- Valid: 0.3h per region  
- Test: 0.3h per region  

This ensures:
- no dialect dominates training  
- fair comparison across regions  


### Data Filtering
We apply lightweight filtering:
- remove empty transcripts  
- filter duration (default: 2–15 seconds)  
- keep only valid regions (North, Central, South)  

We intentionally **preserve Vietnamese text (including diacritics)**.


### Speaker Balancing
We cap the number of utterances per speaker to:
- reduce overfitting to specific voices  
- increase speaker diversity  


## Output Format
### Audio
Audio files are saved using the following structure:
```
data/vimd_subset/audio/
├── train/
│ ├── North/
│ ├── Central/
│ └── South/
├── valid/
│ ├── North/
│ ├── Central/
│ └── South/
└── test/
├── North/
├── Central/
└── South/
```

### Manifest
Saved as:
- `manifest.csv`
- `manifest.jsonl`

Each row contains:
- `audio_path`
- `text`
- `duration_sec`
- `region`
- `speaker_id`
- `split`
- `filename`

## Usage

### Install dependencies
```bash
pip install requirements.txt
python code/extract_data.py --hours_per_region 1
```

### Resume from a checkpoint
Extraction checkpoints are written automatically to:
```
data/vimd_subset/manifests/extract_checkpoint.json
```

If the stream stops before finishing, rerun the same command with `--resume`:
```bash
python code/extract_data.py --hours_per_region 1 --resume
```

The checkpoint stores accepted rows, accepted duration totals, and the last streamed row index for each split. Resume requires the same extraction arguments so the manifest stays consistent.
