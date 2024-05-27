import json
import os
import shutil
from tqdm import tqdm

ROOT = '/datasets/mms/transcribed_lid_clearml'
SPLIT='test'
OUTPUT_DIR = os.path.join(ROOT, SPLIT)
INPUT_MANIFEST_DIR = os.path.join(ROOT, f'{SPLIT}_manifest_raw.json')
OUTPUT_MANIFEST_DIR = os.path.join(ROOT, f'{SPLIT}_manifest.json')

dict_list = []
with open(INPUT_MANIFEST_DIR, 'r+') as f:
    for line in f:
        dict_list.append(json.loads(line))

output_manifest_list = []

for entry in tqdm(dict_list):
    shutil.copy(os.path.join(ROOT, entry['audio_filepath']), OUTPUT_DIR)
    temp = {
        'audio_filepath': os.path.join('mms', SPLIT, os.path.basename(entry['audio_filepath'])),
        'duration': entry['duration'],
        'language': entry['language']
    }

    output_manifest_list.append(temp)

# export manifest
with open(OUTPUT_MANIFEST_DIR, 'w+', encoding='utf-8') as f:
    for data in output_manifest_list:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')