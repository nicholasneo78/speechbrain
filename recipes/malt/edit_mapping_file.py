import json
import os
from tqdm import tqdm

mapping_file_dir = '/models/lang-id-voxlingua107-ecapa/label_encoder_full.txt'
mapping_file_dir_output = '/models/lang-id-voxlingua107-ecapa/label_encoder.txt'

with open(mapping_file_dir, 'r+') as f:
    lines = [
        line.strip('\r\n').replace("\'", '').replace(' => ', ': ').split(': ') for line in f.readlines()[:-2]
    ]

final_list = []

for line in lines:
    processed_line = f"'{line[0]}' => {line[2]}"
    final_list.append(processed_line)

print(final_list)

# export final text file
with open(mapping_file_dir_output, 'w') as f:
    for line in final_list:
        f.write(f"{line}\n")