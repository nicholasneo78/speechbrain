import json
import os
from tqdm import tqdm

def get_label_encoder_lang_id(mapping_file_dir: str) -> int:

    with open(mapping_file_dir, 'r+') as f:
        lines = [
            line.strip('\r\n').replace("\'", '').replace(' => ', ': ').split(': ') for line in f.readlines()[:-2]
        ]

    mapping_dict = {}

    for line in lines:
        # mapping_dict[line[0]] = {
        #     'lang_id': int(line[2]),
        #     'language': line[1],
        # }
        mapping_dict[int(line[2])] = f'{line[0]}: {line[1]}'

    return mapping_dict

if __name__ == "__main__":

    ROOT = '/models/lang-id-ecapa-mms-finetuned/1779'
    LABEL_ENCODER = os.path.join(ROOT, 'save/label_encoder.txt')
    CARTOGRAPHY_DIR = 'cartography'
    CARTOGRAPHY_DIR_LIST = [
        os.path.join(ROOT, CARTOGRAPHY_DIR, file) for file in os.listdir(os.path.join(ROOT, CARTOGRAPHY_DIR))
    ]
    OUTPUT_DIR = os.path.join(ROOT, 'cartography_processed')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mapping_dict = get_label_encoder_lang_id(
        mapping_file_dir=LABEL_ENCODER
    )

    print(mapping_dict)
    
    for json_file in tqdm(CARTOGRAPHY_DIR_LIST):

        with open(json_file, 'r+') as fr:
            lines = fr.readlines()
            items = [json.loads(line.strip('\r\n')) for line in lines]

        for item in items:
            temp_dict = {}
            for key in mapping_dict:
                temp_dict[mapping_dict[key]] = item['confidence_full'][key]

            item['confidence_full'] = temp_dict

        output_filepath = os.path.join(OUTPUT_DIR, os.path.basename(json_file))

        with open(output_filepath, 'w+', encoding='utf-8') as f:
            for data in items:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')