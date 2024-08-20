import os
import json
from tqdm import tqdm
from typing import List, Dict
import torch

from speechbrain.inference import EncoderClassifier
from sklearn.metrics import classification_report, confusion_matrix

class LIDInference:

    def __init__(
        self,
        root_dir: str,
        model_dir: str, 
        input_manifest_dir: str,
        output_manifest_dir: str,
    ) -> None:

        self.root_dir = root_dir
        self.model_dir = model_dir
        self.input_manifest_dir = input_manifest_dir
        self.output_manifest_dir = output_manifest_dir

        self.MAPPING = {
            'en': "en: English",
            'id': "id: Indonesian",
            'ms': "ms: Malay",
        }

    
    def language_mapping(self, language: str) -> str:

        return self.MAPPING[language]

    
    def load_nemo_manifest(self):

        with open(self.input_manifest_dir, 'r+', encoding='utf-8') as fr:
            lines = fr.readlines()
            items = [json.loads(line.strip('\r\n')) for line in lines]

        return items


    def load_sb_model(self) -> EncoderClassifier:
        
        classifier = EncoderClassifier.from_hparams(
            source=self.model_dir,
            hparams_file=os.path.join(self.model_dir, 'hyperparams.yaml'),
        )

        return classifier


    def infer(self) -> None:

        items = self.load_nemo_manifest()
        classifier = self.load_sb_model()

        for item in tqdm(items):
            item['ref_language'] = self.language_mapping(language=item['language'])
            signal = classifier.load_audio(os.path.join(self.root_dir, item['audio_filepath']), savedir='/temp')
            logits, max_logits, pred_lang_id, pred = classifier.classify_batch(signal)

            confidence = round((torch.exp(max_logits[0]) / sum(torch.exp(logits[0]))).item(), 8)

            item['pred_language'] = pred[0]
            item['confidence'] = confidence

            print(f"Reference: {item['ref_language']} | Pred: {pred[0]} | Confidence: {confidence}")

        # export manifest
        with open(self.output_manifest_dir, 'w+', encoding='utf-8') as fw:
            for item in items:
                fw.write(json.dumps(item) + '\n')


    def __call__(self) -> None:

        return self.infer()
    

class LIDEvaluation:

    def __init__(self, manifest_dir: str, language_list: List[str]) -> None:

        self.manifest_dir = manifest_dir
        self.language_list = language_list


    def load_nemo_manifest(self):

        with open(self.manifest_dir, 'r+', encoding='utf-8') as fr:
            lines = fr.readlines()
            items = [json.loads(line.strip('\r\n')) for line in lines]

        return items


    def compute_metrics(self) -> None:
        
        items = self.load_nemo_manifest()

        ref_list = [item['ref_language'] for item in items]
        pred_list = [item['pred_language'] for item in items]

        duration = sum([item['duration'] for item in items])

        print(f"Total duration: {duration/3600} h")
        print(confusion_matrix(ref_list, pred_list))
        print(classification_report(ref_list, pred_list, target_names=self.language_list, digits=4))


    def __call__(self) -> None:

        return self.compute_metrics()


if __name__ == '__main__':

    # ensure that the hyparams and the label encoder file is in the directory stated
    MODE = 'evaluation'
    # MODEL = '/models/lang-id-ecapa-mms-finetuned/778/save/CKPT+2024-04-09+11-13-05+00'
    MODEL = '/models/lang-id-ecapa-mms-finetuned/2024072401/save/CKPT+2024-07-25+01-28-10+00'
    # ROOT = '/datasets/mms/transcribed_lid/'
    # raw test set
    ROOT = '/datasets/mms/transcribed_bibm_final/processed_BM/test_split'
    # ROOT = '/datasets/mms/transcribed/test_for_D5/EN_test_set'

    INPUT_MANIFEST_DIR = os.path.join(ROOT, 'test_manifest_lid.json')
    OUTPUT_MANIFEST_DIR = os.path.join(ROOT, 'pred_test_manifest_lid_2024072401.json')

    LANGUAGE_LIST = ['en: English', 'id: Indonesian', 'ms: Malay']
    if MODE == "inference":
        c = LIDInference(
            root_dir=ROOT,
            model_dir=MODEL,
            input_manifest_dir=INPUT_MANIFEST_DIR,
            output_manifest_dir=OUTPUT_MANIFEST_DIR
        )()
    else:
        ev = LIDEvaluation(
            manifest_dir=OUTPUT_MANIFEST_DIR,
            language_list=LANGUAGE_LIST,
        )()