"""
Streams and extracts a balanced subset of the ViMD dataset for Vietnamese TTS.
Balances hours across 3 regional dialects (North/Central/South) and 3 splits.
 
Note: I used an agentic AI assistant(Claude + ChatGPT) during development to help debug
the HuggingFace streaming decode issues and figure out the audio loading logic.
Overall structure and design decisions are my own.
 
TODO: add --dry-run flag to preview targets without writing files
TODO: improve error logging, right now everything just prints to stdout
"""
 
import argparse
import csv
import io
import json
import re
from pathlib import Path

import librosa
import soundfile as sf
from datasets import load_dataset


REGIONS = ["North", "Central", "South"]
SPLITS = ["train", "valid", "test"]
TARGET_SAMPLE_RATE = 24_000
 
SPLIT_RATIOS = {
    "train": 0.8,
    "valid": 0.1,
    "test": 0.1,
}
 
 
def clean_text(text: str) -> str:
    # keeping normalization minimal - TTS is sensitive to text changes
    if text is None:
        return ""
 
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text
 
 
def get_duration(audio: dict):
    """
    Compute duration from the waveform directly.
    row["length"] is unreliable in streaming mode so we can't use it.
    """
    if audio is None:
        return None
 
    try:
        source_bytes = audio.get("bytes")
        source_path = audio.get("path")
 
        if source_bytes is not None:
            audio_array, sample_rate = sf.read(io.BytesIO(source_bytes))
        elif source_path is not None:
            audio_array, sample_rate = sf.read(source_path)
        else:
            return None
 
        return len(audio_array) / sample_rate
 
    except Exception as e:
        print(f"[WARN] audio read failed: {e}")
        return None
 
 
def is_valid_row(row, min_dur, max_dur) -> bool:
    text = clean_text(row.get("text", ""))
    region = row.get("region")
    audio = row.get("audio")
 
    if not text:
        return False
    if region not in REGIONS:
        return False
    if audio is None:
        return False
 
    duration = get_duration(audio)
    if duration is None:
        return False
    if not (min_dur <= duration <= max_dur):
        return False
 
    return True
 
 
def ensure_dirs(output_root: Path):
    (output_root / "audio").mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)
 
 
def save_audio(audio_info: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
 
    source_bytes = audio_info.get("bytes")
    source_path = audio_info.get("path")
 
    if source_bytes is not None:
        audio_array, sample_rate = sf.read(io.BytesIO(source_bytes))
    elif source_path is not None:
        audio_array, sample_rate = sf.read(source_path)
    else:
        raise ValueError("audio source is missing both bytes and path")

    if sample_rate != TARGET_SAMPLE_RATE:
        audio_array = librosa.resample(
            y=audio_array.T if getattr(audio_array, "ndim", 1) > 1 else audio_array,
            orig_sr=sample_rate,
            target_sr=TARGET_SAMPLE_RATE,
        )
        if getattr(audio_array, "ndim", 1) > 1:
            audio_array = audio_array.T

    sf.write(output_path, audio_array, TARGET_SAMPLE_RATE)


def build_manifest_rows(rows):
    return [
        {
            "audio_file": row["audio_path"],
            "text": f"[{row['region']}] {row['text']}",
        }
        for row in rows
    ]
 
 
def build_targets(hours_per_region: float):
    """
    Compute per-split, per-region duration targets in seconds.
    e.g. 1h per region -> train: 0.8h, valid: 0.1h, test: 0.1h
    """
    total_sec = hours_per_region * 3600
    return {
        split: {region: total_sec * SPLIT_RATIOS[split] for region in REGIONS}
        for split in SPLITS
    }
 
 
def extract_subset(dataset_name, output_root, hours_per_region, min_dur, max_dur):
    ensure_dirs(output_root)
    targets = build_targets(hours_per_region)
 
    accepted_sec = {split: {region: 0.0 for region in REGIONS} for split in SPLITS}
    rows = []
 
    print("\n[INFO] Targets:")
    for split in SPLITS:
        for region in REGIONS:
            print(f"  {split} | {region}: {targets[split][region]/3600:.2f}h")
 
    for split_name in SPLITS:
        print(f"\n[INFO] Streaming split: {split_name}")
 
        # .decode(False) keeps audio as raw bytes rather than decoding upfront
        # needed for streaming mode to avoid loading everything into memory
        ds_stream = load_dataset(dataset_name, split=split_name, streaming=True)
        ds_stream = ds_stream.decode(False).with_format("python")
 
        for idx, row in enumerate(ds_stream):
 
            if all(accepted_sec[split_name][r] >= targets[split_name][r] for r in REGIONS):
                print(f"[INFO] Finished split: {split_name}")
                break
 
            if not is_valid_row(row, min_dur, max_dur):
                continue
 
            region = row["region"]
            audio = row["audio"]
 
            if accepted_sec[split_name][region] >= targets[split_name][region]:
                continue
 
            duration = get_duration(audio)
            if duration is None:
                continue
 
            text = clean_text(row["text"])
            filename = row.get("filename", f"{split_name}_{idx}")
            base = Path(filename).stem
            wav_path = output_root / "audio" / split_name / region / f"{split_name}_{base}.wav"
 
            try:
                save_audio(audio, wav_path)
            except Exception as e:
                print(f"[WARN] failed to save {filename}: {e}")
                continue
 
            rows.append({
                "audio_path": str(wav_path.resolve()),
                "text": text,
                "duration_sec": round(duration, 3),
                "region": region,
                "split": split_name,
                "filename": filename,
            })
 
            accepted_sec[split_name][region] += duration
 
            if len(rows) % 200 == 0:
                print(f"[INFO] {len(rows)} samples accepted so far...")
 
    return rows, accepted_sec
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Extract a balanced ViMD subset for Vietnamese TTS."
    )
    parser.add_argument("--dataset_name", type=str, default="nguyendv02/ViMD_Dataset")
    parser.add_argument("--output_dir", type=str, default="data/vimd_subset")
    parser.add_argument("--hours_per_region", type=float, default=10.0)
    parser.add_argument("--min_duration", type=float, default=3.0)
    parser.add_argument("--max_duration", type=float, default=15.0)
    args = parser.parse_args()
 
    output_root = Path(args.output_dir)
    manifest_dir = output_root / "manifests"
 
    rows, accepted_sec = extract_subset(
        dataset_name=args.dataset_name,
        output_root=output_root,
        hours_per_region=args.hours_per_region,
        min_dur=args.min_duration,
        max_dur=args.max_duration,
    )
 
    if not rows:
        print("\n[WARN] No samples selected. Check filters:")
        print(f"       min_duration={args.min_duration}, max_duration={args.max_duration}")
        print("\n[INFO] Accepted so far:")
        for split in SPLITS:
            for region in REGIONS:
                print(f"  {split} | {region}: {accepted_sec[split][region]/3600:.2f}h")
        return
 
    manifest_rows = build_manifest_rows(rows)

    with open(manifest_dir / "manifest_original.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(manifest_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["audio_file", "text"],
            delimiter="|",
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    with open(manifest_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n[INFO] Done. {len(rows)} samples saved.")
    print(f"       CSV:   {manifest_dir / 'manifest_original.csv'} (original extracted rows)")
    print(f"       CSV:   {manifest_dir / 'manifest.csv'} (pipe-delimited training format)")
    print(f"       JSONL: {manifest_dir / 'manifest.jsonl'}")
    print(f"       Audio: resampled to {TARGET_SAMPLE_RATE} Hz")
 
    print("\n[INFO] Accepted duration summary:")
    for split in SPLITS:
        for region in REGIONS:
            print(f"  {split} | {region}: {accepted_sec[split][region]/3600:.2f}h")
 
 
if __name__ == "__main__":
    main()
