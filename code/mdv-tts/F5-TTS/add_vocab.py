import pandas as pd
import os

manifest_path = "data/viet_tts/manifest.csv"
pretrained_vocab_path = "data/Emilia_ZH_EN_pinyin/vocab.txt"

# 1. Read all text from your manifest
df = pd.read_csv(manifest_path, sep='|')
text_data = df['text'].dropna().tolist()

# 2. Get unique characters from your dataset
viet_chars = set()
for text in text_data:
    viet_chars.update(list(text))

# 3. Read the original F5-TTS characters
with open(pretrained_vocab_path, "r", encoding="utf-8") as f:
    pretrained_chars = set(line.strip() for line in f.readlines() if line.strip())

# 4. Merge and save back to the original file
merged_chars = pretrained_chars.union(viet_chars)
with open(pretrained_vocab_path, "w", encoding="utf-8") as f:
    for char in sorted(merged_chars):
        f.write(char + "\n")
        
print("Successfully synced Vietnamese characters into the F5-TTS vocab list!")