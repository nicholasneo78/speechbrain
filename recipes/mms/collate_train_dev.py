from typing import List, Dict
import os
import json
import random
from tqdm import tqdm

class CollateTrainDev:

    """
    Collate the train and dev set, make sure the train set is balanced
    """

    def __init__(
        self,
        root_dir: str,
        train_manifest_list: List[str],
        dev_manifest_list: List[str],
        output_train_manifest_dir: str,
        output_dev_manifest_dir: str,
    ) -> None:
        
        self.root_dir = root_dir
        self.train_manifest_list = train_manifest_list
        self.dev_manifest_list = dev_manifest_list
        self.output_train_manifest_dir = output_train_manifest_dir
        self.output_dev_manifest_dir = output_dev_manifest_dir
        self.SEED = 7


    def load_manifest(self, manifest_dir: str) -> List[Dict[str, str]]:

        with open(os.path.join(self.root_dir, manifest_dir), 'r+') as fr:
            lines = fr.readlines()
            items = [json.loads(line.strip('\r\n')) for line in lines]

        return items

    
    def get_min_duration(self) -> int:

        """
        compare datasets of different languages with different duration in the train set, get the minimum
        """
        
        train_duration_list = []

        for train_manifest_dir in self.train_manifest_list:

            items = self.load_manifest(manifest_dir=os.path.join(self.root_dir, train_manifest_dir))

            duration_count = 0

            for item in items:
                duration_count += item['duration']

            train_duration_list.append(duration_count)

        return min(train_duration_list)
    

    def collate(self) -> None:

        min_duration_threshold = self.get_min_duration()

        final_train_manifest_list = []
        final_dev_manifest_list = []

        for train_manifest_dir in self.train_manifest_list:

            sub_train_manifest_list = []
            duration_counter = 0

            items = self.load_manifest(manifest_dir=os.path.join(self.root_dir, train_manifest_dir))

            # shuffle the list
            random.Random(self.SEED).shuffle(items)

            for item in tqdm(items):
                if duration_counter < min_duration_threshold:
                    sub_train_manifest_list.append(item)
                    duration_counter += item['duration']
                else:
                    break

            final_train_manifest_list.extend(sub_train_manifest_list)

        for dev_manifest_dir in self.dev_manifest_list:

            items = self.load_manifest(manifest_dir=os.path.join(self.root_dir, dev_manifest_dir))

            final_dev_manifest_list.extend(items)

        # shuffle the final list
        random.Random(self.SEED).shuffle(final_train_manifest_list)
        random.Random(self.SEED).shuffle(final_dev_manifest_list)

        # export the manifest files
        with open(os.path.join(self.root_dir , self.output_train_manifest_dir), 'w', encoding='utf-8') as fw:
            for item in final_train_manifest_list:
                fw.write(json.dumps(item)+'\n')

        with open(os.path.join(self.root_dir , self.output_dev_manifest_dir), 'w', encoding='utf-8') as fw:
            for item in final_dev_manifest_list:
                fw.write(json.dumps(item)+'\n')


    def __call__(self) -> None:

        return self.collate()
    

if __name__ == "__main__":

    ROOT_DIR = '/datasets/mms/transcribed_lid'
    TRAIN_MANIFEST_LIST = [
        'train_manifest_en.json',
        'train_manifest_id.json',
        'train_manifest_ms.json',
    ]
    DEV_MANIFEST_LIST = [
        'dev_manifest_en.json',
        'dev_manifest_id.json',
        'dev_manifest_ms.json',
    ]
    OUTPUT_TRAIN_MANIFEST_DIR = 'train_manifest.json'
    OUTPUT_DEV_MANIFEST_DIR = 'dev_manifest.json'

    c = CollateTrainDev(
        root_dir=ROOT_DIR,
        train_manifest_list=TRAIN_MANIFEST_LIST,
        dev_manifest_list=DEV_MANIFEST_LIST,
        output_train_manifest_dir=OUTPUT_TRAIN_MANIFEST_DIR,
        output_dev_manifest_dir=OUTPUT_DEV_MANIFEST_DIR,
    )()