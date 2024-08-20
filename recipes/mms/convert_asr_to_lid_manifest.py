"""
For audio's test set, to convert ASR manifest to lid one, for KNOWN audio language
"""

import json
import os
from typing import List, Dict

class ConvertASRtoLIDManifest:

    """
    main class to convert ASR to LID manifest
    requires: "audio_filepath" and "duration" from the ASR manifest
    """

    def __init__(
            self,
            input_manifest_path: str,
            output_manifest_path: str, 
            language: str
        ) -> None:

        """
        input_manifest_path: input ASR manifest
        language: ground truth language to be appended
        """

        self.input_manifest_path = input_manifest_path
        self.output_manifest_path = output_manifest_path
        self.language = language

    
    def load_manifest_nemo(self) -> List[Dict[str, str]]:

        '''
        loads the manifest file in Nvidia NeMo format to process the entries and store them into a list of dictionaries

        the manifest file would contain entries in this format:

        {"audio_filepath": "subdir1/xxx1.wav", "duration": 3.0, "text": "shan jie is an orange cat"}
        {"audio_filepath": "subdir1/xxx2.wav", "duration": 4.0, "text": "shan jie's orange cat is chonky"}
        ---

        input_manifest_path: the manifest path that contains the information of the audio clips of interest
        ---
        returns: a list of dictionaries of the information in the input manifest file
        '''

        dict_list = []

        with open(self.input_manifest_path, 'r+') as f:
            for line in f:
                dict_list.append(json.loads(line))

        return dict_list
    

    def export_splits(self, data_list: List[Dict[str, str]]) -> None:

        '''
        outputs the respective (train, dev or test) manifest with the split data entries from a list 
        ---

        manifest_dir: the output manifest directory to be exported (train, dev or test)
        data_list: the list of splitted entries 
        '''
        
        with open(self.output_manifest_path, 'w+', encoding='utf-8') as f:
            for data in data_list:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

    
    def convert(self) -> None:

        """
        main method to convert the ASR manifest to an LID one, that is inference ready
        """

        dict_list = self.load_manifest_nemo()
        lid_dict_list = []

        for entry in dict_list:

            temp = {
                'audio_filepath': entry['audio_filepath'],
                'duration': entry['duration'],
                'language': self.language
            }

            lid_dict_list.append(temp)

        self.export_splits(data_list=lid_dict_list)


    def __call__(self) -> None:

        return self.convert()

        
if __name__ == "__main__":

    # ROOT = '/datasets/mms/transcribed_bibm_final/processed_BM/test_split'
    ROOT = '/datasets/mms/transcribed/test_for_D5/EN_test_set/set_3'
    
    INPUT_MANIFEST = os.path.join(ROOT, 'test_manifest.json')
    OUTPUT_MANIFEST = os.path.join(ROOT, 'test_manifest_lid.json')
    LANGUAGE = 'en'

    c = ConvertASRtoLIDManifest(
        input_manifest_path=INPUT_MANIFEST,
        output_manifest_path=OUTPUT_MANIFEST,
        language=LANGUAGE
    )()
