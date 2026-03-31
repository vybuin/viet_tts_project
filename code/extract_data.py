"""
extract_data.py

Purpose:
To build a balanced Vietnamese TTS dataset subset.

Key design:
- Uses Hugging Face streaming (no full dataset download)
- Filters unusable samples
- Balances data evenly across:
    - regions: North/Central/South
    - splits: train/valid/test
- Caps utterances per speaker to avoid speaker dominance
- Saves only selected audio + a clean manifest

Output:
- audio files under data/.../audio/
- manifest.csv + manifest.jsonl under data/.../manifests/

Why this matters:
This ensures a fair baseline for dialect comparison and allows
easy scaling (1h → 3h → 9h per region) without rewriting code.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import io
from datasets import load_dataset


# 3 regional dialects
REGIONS = ["North", "Central", "South"]

# Hugging Face dataset splits
SPLITS = ["train", "valid", "test"]

# Used to allocate hours across splits
DEFAULT_SPLIT_RATIOS = {
    "train": 0.8,
    "valid": 0.1,
    "test": 0.1,
}


def clean_text(text: str) -> str:
    """
    Minimal cleaning:
    - preserve Vietnamese diacritics
    - normalize whitespace
    - fix spacing before punctuation
    We intentionally avoid heavy normalization since TTS depends on raw text.
    """
    if text is None:
        return ""

    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text


def is_valid_row(row, min_duration: float, max_duration: float) -> bool:
    """
    Filter unusable samples in case that data isn't clean:
    - empty text
    - missing region/speaker/audio
    - duration outside bounds
    This keeps training stable and avoids noisy data.
    """
    text = clean_text(row.get("text", ""))
    region = row.get("region")
    speaker_id = row.get("speakerID")
    duration = row.get("length")
    audio = row.get("audio")

    if not text:
        return False
    if region not in REGIONS:
        return False
    if speaker_id is None:
        return False
    if duration is None:
        return False
    if not (min_duration <= float(duration) <= max_duration):
        return False
    if audio is None:
        return False

    return True


def ensure_dirs(output_root: Path):
    """
    Create output folders:
    - audio/
    - manifests/
    """
    (output_root/"audio").mkdir(parents=True, exist_ok=True)
    (output_root/"manifests").mkdir(parents=True, exist_ok=True)


def save_audio_from_hf_audio(audio_info: dict, output_path: Path):
    """
    Save audio to .wav file.
    We only save selected subset audio to avoid huge storage usage.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_bytes = audio_info.get("bytes")
    source_path = audio_info.get("path")

    if source_bytes is not None:
        audio_array, sample_rate = sf.read(io.BytesIO(source_bytes))
    elif source_path is not None:
        audio_array, sample_rate = sf.read(source_path)
    else:
        raise ValueError("Audio source is missing both path and bytes.")

    sf.write(output_path, audio_array, sample_rate)


def build_targets(hours_per_region: float):
    """
    Compute how many seconds we want per:
    - split (train/valid/test)
    - region (North/Central/South)

    Example:
        3h per region →
        train: 2.4h, valid: 0.3h, test: 0.3h
    """
    total_seconds = hours_per_region * 3600

    targets = {
        split: {
            region: total_seconds * DEFAULT_SPLIT_RATIOS[split]
            for region in REGIONS
        }
        for split in SPLITS
    }

    return targets


def extract_balanced_subset_per_split(
    dataset_name: str,
    output_root: Path,
    hours_per_region: float,
    min_duration: float,
    max_duration: float,
    max_utts_per_speaker: int,
):
    """
    Main extraction logic.

    We stream the dataset and:
    - accept samples until each (split, region) reaches its target duration
    - enforce speaker caps
    - maintain balance across dialects and splits
    """
    ensure_dirs(output_root)

    targets = build_targets(hours_per_region)

    # Track how much audio we have collected
    accepted_seconds = {
        split: {region: 0.0 for region in REGIONS}
        for split in SPLITS
    }

    # Track speaker usage (prevents one speaker dominating)
    speaker_counts = {
        split: {region: defaultdict(int) for region in REGIONS}
        for split in SPLITS
    }

    rows = []

    print("\n[INFO] Per-split extraction targets:")
    for split in SPLITS:
        for region in REGIONS:
            hrs = targets[split][region] / 3600
            print(f"  - {split:<5} | {region:<7}: {hrs:.2f}h")

    # Process each split separately
    for split_name in SPLITS:
        print(f"\n[INFO] Streaming split: {split_name}")
        ds_stream = load_dataset(
            dataset_name,
            split=split_name,
            streaming=True)
        ds_stream = ds_stream.decode(False)
        ds_stream = ds_stream.with_format("python")

        for idx, row in enumerate(ds_stream):
            if not is_valid_row(row, min_duration, max_duration):
                continue

            region = row["region"]
            speaker_id = row["speakerID"]
            duration = float(row["length"])

            # Stop if this split-region already reached target
            if all(
                accepted_seconds[split_name][region] >= targets[split_name][region]
                for region in REGIONS
            ):
                print(f"[INFO] Finished split: {split_name}")
                break

            # Cap per speaker
            if speaker_counts[split_name][region][speaker_id] >= max_utts_per_speaker:
                continue

            text = clean_text(row["text"])
            filename = row.get("filename", f"{split_name}_{idx}")
            # remove extension if already exists
            base_filename = Path(filename).stem
            wav_filename = f"{split_name}_{base_filename}.wav"
            wav_path = output_root / "audio" / split_name / region / wav_filename

            try:
                save_audio_from_hf_audio(row["audio"], wav_path)
            except Exception as e:
                print(f"[WARN] Failed to save {filename}: {e}")
                continue

            rows.append({
                "audio_path": str(wav_path.resolve()),
                "text": text,
                "duration_sec": duration,
                "region": region,
                "speaker_id": speaker_id,
                "split": split_name,
                "filename": filename,
            })

            accepted_seconds[split_name][region] += duration
            speaker_counts[split_name][region][speaker_id] += 1

            if len(rows) % 200 == 0:
                print(f"[INFO] Accepted {len(rows)} samples so far...")

    return rows, accepted_seconds, speaker_counts


def main():
    parser = argparse.ArgumentParser(description="Extract balanced ViMD subset per split for Vietnamese TTS.")
    parser.add_argument("--dataset_name", type=str, default="nguyendv02/ViMD_Dataset")
    parser.add_argument("--output_dir", type=str, default="data/vimd_subset")
    parser.add_argument("--hours_per_region", type=float, default=1.0)
    parser.add_argument("--min_duration", type=float, default=3.0)
    parser.add_argument("--max_duration", type=float, default=25.0)
    parser.add_argument("--max_utts_per_speaker", type=int, default=8)
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    manifest_dir = output_root / "manifests"

    rows, accepted_seconds, speaker_counts = extract_balanced_subset_per_split(
        dataset_name=args.dataset_name,
        output_root=output_root,
        hours_per_region=args.hours_per_region,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        max_utts_per_speaker=args.max_utts_per_speaker,
    )

    if not rows:
        print("\n[WARN] No samples were selected, so no manifest files were written.")
        print("[WARN] Try loosening the filters or checking whether audio loading is still failing.")
        print(
            f"[WARN] Current filters: min_duration={args.min_duration}, "
            f"max_duration={args.max_duration}, max_utts_per_speaker={args.max_utts_per_speaker}"
        )
        print("\n[INFO] Accepted duration summary:")
        for split in SPLITS:
            for region in REGIONS:
                hrs = accepted_seconds[split][region] / 3600
                unique_speakers = len(speaker_counts[split][region])
                print(
                    f"  - {split:<5} | {region:<7}: {hrs:.2f}h accepted "
                    f"across {unique_speakers} speakers"
                )
        return

    # Save outputs
    with open(manifest_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(manifest_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n[INFO] Saved {len(rows)} samples.")


if __name__ == "__main__":
    main()
