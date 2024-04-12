from typing import List, Dict
import os
import json
from tqdm import tqdm

class ConvertToSBJson:

    """
    converts the NeMo json format to the speechbrain json format
    """

    def __init__(self, root_dir: str, input_manifest_dir: str, output_manifest_dir: str) -> None:

        self.root_dir = root_dir
        self.input_manifest_dir = input_manifest_dir
        self.output_manifest_dir = output_manifest_dir

    def load_manifest(self) -> List[Dict[str, str]]:

        with open(os.path.join(self.root_dir, self.input_manifest_dir), 'r+') as fr:
            lines = fr.readlines()
            items = [json.loads(line.strip('\r\n')) for line in lines]

        return items

    
    def convert(self) -> None:

        items = self.load_manifest()

        final_output_dict = {}

        language_mapping = {
            'en': 'en: English',
            'id': 'id: Indonesian',
            'ms': 'ms: Malay'
        }

        for item in tqdm(items):

            final_output_dict[
            os.path.basename(item['audio_filepath'])
            .split('.')[0]
            ] = {
                'wav': "{data_root}" + "/" + item['audio_filepath'],
                'wav_format': ".wav",
                'duration': item['duration'],
                'language': language_mapping[item['language']]
            }

        # save the updated json file
        with open(os.path.join(self.root_dir, self.output_manifest_dir), 'w', encoding='utf-8') as fw:
            json.dump(final_output_dict, fw, indent=2)

    
    def __call__(self) -> None:

        return self.convert()
    
if __name__ == "__main__": 

    # ROOT_DIR = '/datasets/mms/transcribed_lid'
    # INPUT_MANIFEST_DIR = 'train_manifest.json'
    # OUTPUT_MANIFEST_DIR = 'train_manifest_sb.json'

    # c = ConvertToSBJson(
    #     root_dir=ROOT_DIR,
    #     input_manifest_dir=INPUT_MANIFEST_DIR,
    #     output_manifest_dir=OUTPUT_MANIFEST_DIR
    # )()

    # ROOT_DIR = '/datasets/mms/transcribed_lid'
    # INPUT_MANIFEST_DIR = 'dev_manifest.json'
    # OUTPUT_MANIFEST_DIR = 'dev_manifest_sb.json'

    # c = ConvertToSBJson(
    #     root_dir=ROOT_DIR,
    #     input_manifest_dir=INPUT_MANIFEST_DIR,
    #     output_manifest_dir=OUTPUT_MANIFEST_DIR
    # )()

    ROOT_DIR = '/datasets/mms/transcribed_lid'
    INPUT_MANIFEST_DIR = 'test_manifest.json'
    OUTPUT_MANIFEST_DIR = 'test_manifest_sb.json'

    c = ConvertToSBJson(
        root_dir=ROOT_DIR,
        input_manifest_dir=INPUT_MANIFEST_DIR,
        output_manifest_dir=OUTPUT_MANIFEST_DIR
    )()