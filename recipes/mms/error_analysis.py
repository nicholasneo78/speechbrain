import os
import json
from typing import List

def load_nemo_manifest(manifest_dir: str) -> List[str]:

    with open(manifest_dir, 'r+', encoding='utf-8') as fr:
        lines = fr.readlines()
        items = [json.loads(line.strip('\r\n')) for line in lines]

    return items


def get_misclassification(items: List[str], mode: str):

    count = 0
    count_short_duration = 0

    if mode == 'incorrect':

        for item in items:
            if item['ref_language'] != item['pred_language'] and item['pred_language'] != "en: English":
                count += 1

                if item['duration'] <= 3:
                    count_short_duration += 1

    elif mode == 'correct':
        
        for item in items:
            if item['ref_language'] == item['pred_language']:
                count += 1

                if item['duration'] <= 3:
                    count_short_duration += 1

    print(f"Total misclassified count: {count}")
    print(f"Total misclassified count with short timestamp: {count_short_duration}")

    print(f"% short timestamp misclassified: {count_short_duration/count}")


if __name__ == '__main__':

    ROOT = '/datasets/mms/transcribed_bibm_final/processed_BI/test_split'
    # ROOT = '/datasets/mms/transcribed/test_for_D5/EN_test_set'
    OUTPUT_MANIFEST_DIR = os.path.join(ROOT, 'pred_test_manifest_lid_2024072401.json')
    mode = 'incorrect'

    items = load_nemo_manifest(manifest_dir=OUTPUT_MANIFEST_DIR)
    
    get_misclassification(
        items=items,
        mode=mode
    )

