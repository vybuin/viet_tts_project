import pandas as pd
import os

# 1. Load your existing manifest
input_csv = "../data/vimd_subset/manifests/manifest.csv"
df = pd.read_csv(input_csv)

# 2. Fix the incorrect absolute path.
# This extracts the trailing part of the path starting from 'vimd_subset/audio/...'
extracted_paths = df['audio_path'].str.extract(r'(vimd_subset/audio/.*)')[0]

# Prepend the correct absolute base path for your current machine
base_dir = "/Users/quentonni/Desktop/Spring2026/5541/viet-tts-test/data/"
df['audio_path'] = base_dir + extracted_paths

# 3. Add dialect tags to the text 
df['formatted_text'] = '[' + df['region'] + '] ' + df['text']

# 4. Create the expected structure
f5_df = pd.DataFrame({
    'audio_file': df['audio_path'],
    'text': df['formatted_text']
})

# 5. Save to your new location using pipe '|' separator
output_csv = "../data/viet_tts/manifest.csv" 
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
f5_df.to_csv(output_csv, sep='|', index=False)

print(f"Created new manifest at {output_csv}")