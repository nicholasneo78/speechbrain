from typing import List, Dict
import os
import json
import random
from tqdm import tqdm

class CollateTest:

    """
    Collate the train and dev set, make sure the train set is balanced
    """

    def __init__(
        self,
        root_dir: str,
        test_manifest_list: List[str],
        output_test_manifest_dir: str,
    ) -> None:
        
        self.root_dir = root_dir
        self.test_manifest_list = test_manifest_list
        self.output_test_manifest_dir = output_test_manifest_dir
        self.SEED = 7


    def load_manifest(self, manifest_dir: str) -> List[Dict[str, str]]:

        with open(os.path.join(self.root_dir, manifest_dir), 'r+') as fr:
            lines = fr.readlines()
            items = [json.loads(line.strip('\r\n')) for line in lines]

        return items
    

    def collate(self) -> None:

        final_test_manifest_list = []

        for test_manifest_dir in self.test_manifest_list:

            items = self.load_manifest(manifest_dir=os.path.join(self.root_dir, test_manifest_dir))

            final_test_manifest_list.extend(items)

        # shuffle the final list
        random.Random(self.SEED).shuffle(final_test_manifest_list)

        # export the manifest files
        with open(os.path.join(self.root_dir , self.output_test_manifest_dir), 'w', encoding='utf-8') as fw:
            for item in final_test_manifest_list:
                fw.write(json.dumps(item)+'\n')


    def __call__(self) -> None:

        return self.collate()
    
if __name__ == "__main__":

    ROOT_DIR = '/datasets/mms/transcribed_lid'

    TEST_MANIFEST_LIST = [
        'test_manifest_en.json',
        'test_manifest_id.json',
        'test_manifest_ms.json',
        'test_long_manifest_id.json',
        'test_long_manifest_ms.json',
    ]

    OUTPUT_TEST_MANIFEST_DIR = 'test_manifest.json'

    c = CollateTest(
        root_dir=ROOT_DIR,
        test_manifest_list=TEST_MANIFEST_LIST,
        output_test_manifest_dir=OUTPUT_TEST_MANIFEST_DIR,
    )()