from typing import List, Dict
import os
import json
import subprocess
from tqdm import tqdm

from speechbrain.inference.VAD import VAD

class FilterBestAudio:

    """
    Filter and only get the most suitable audio length after passing through VAD
    """

    def __init__(
        self, 
        input_root_dir: str,
        input_manifest_dir: str,
        min_audio_duration: float,
        long_audio_multiplier: float,
        output_root_dir: str,
        output_manifest_dir: str,
        split: str,
        language: str
    ) -> None:

        self.vad = VAD.from_hparams(source='/models/vad-crdnn-libriparty')
        self.DATASET = 'mms'

        self.input_root_dir = input_root_dir
        self.input_manifest_dir = input_manifest_dir
        self.min_audio_duration = min_audio_duration
        self.long_audio_multiplier = long_audio_multiplier
        self.output_root_dir = output_root_dir
        self.output_manifest_dir = output_manifest_dir
        self.split = split
        self.language = language


    def load_manifest(self) -> List[Dict[str, str]]:

        with open(os.path.join(self.input_root_dir, self.input_manifest_dir), 'r+') as fr:
            lines = fr.readlines()
            items = [json.loads(line.strip('\r\n')) for line in lines]

        return items


    def vad_pipeline(self, audio_filepath: str) -> List[List[str]]:

        prob_chunks = self.vad.get_speech_prob_file(audio_filepath)
        prob_th = self.vad.apply_threshold(prob_chunks, activation_th=0.5, deactivation_th=0.25).float()
        boundaries = self.vad.get_boundaries(prob_th)
        boundaries = self.vad.energy_VAD(audio_filepath, boundaries, activation_th=0.8, deactivation_th=0.0)
        boundaries = self.vad.merge_close_segments(boundaries, close_th=0.250)
        boundaries = self.vad.remove_short_segments(boundaries, len_th=0.250)
        boundaries = self.vad.double_check_speech_segments(boundaries, audio_filepath,  speech_th=0.5)

        # only return those timestamps that are more than the min audio duration
        return [
            [round(value, 5) for value in value_list]
            for value_list in boundaries.tolist()
            if value_list[1]-value_list[0] >= self.min_audio_duration
        ]


    def duration_length_check(self, duration: float) -> bool:

        return duration >= self.min_audio_duration


    def split_audio_based_on_timing(self, main_filename: str, split_filename: str, start: str, end: str) -> None:

        """
        sox implementation of splitting the long audio file into utterance level
        ---
        main_filename: raw audio filepath
        split_filename: split audio filepath
        start: start timestamp in string
        end: end timestamp in string
        """

        subprocess.run(['sox', main_filename, split_filename, 'trim', start, str(float(end)-float(start))])


    def split_long_audio_boundary(self, boundaries_raw: List[List[str]]) -> List[List[str]]:

        """
        takes in raw boundaries from the VAD, if detects a particular boundary is long, split the boundary 
        """

        boundaries = []

        for boundary in boundaries_raw:
            boundary_duration = boundary[1] - boundary[0]
            if boundary_duration <= self.long_audio_multiplier:
                boundaries.append(boundary)
            else:
                # idea is to divide into equal audio length
                # e.g multiplier=10, audio duration=21, your split should be 3 (21//10 + bool(21%10) = 3)
                # e.g multiplier=10, audio duration=20, your split should be 2 (20//10 + bool(20%10) = 2)

                excess_count = 1 if (boundary_duration % self.long_audio_multiplier) != 0 else 0
                split_count = int(boundary_duration // self.long_audio_multiplier + excess_count)
                interval = boundary_duration / split_count
                subsplit = []
                for count in range(split_count):
                    subsplit.append([boundary[0]+count*interval, boundary[0]+(count+1)*interval])
                
                boundaries.extend(subsplit)
        
        return boundaries


    def filter_best_audio(self):

        """
        main method to filter and do the vad on the final audio
        """

        os.makedirs(os.path.join(self.output_root_dir, self.DATASET), exist_ok=True)
        os.makedirs(os.path.join(self.output_root_dir, self.DATASET, self.language), exist_ok=True)
        os.makedirs(os.path.join(self.output_root_dir, self.DATASET, self.language, self.split), exist_ok=True)

        output_manifest_list = []
        entries = self.load_manifest()

        for entry in tqdm(entries):

            if not self.duration_length_check(entry['duration']):
                continue
            
            try:
                boundaries_raw = self.vad_pipeline(audio_filepath=os.path.join(self.input_root_dir, entry['audio_filepath']))
            except RuntimeError:
                print(f'File Error: {entry["audio_filepath"]}')
                continue

            boundaries = self.split_long_audio_boundary(boundaries_raw=boundaries_raw)

            # loop boundaries
            for idx, boundary in enumerate(boundaries):

                audio_filepath = os.path.join(self.DATASET, self.language, self.split, f"{os.path.basename(entry['audio_filepath']).split('.')[0]}-{idx:02}.wav")

                temp = {
                    "audio_filepath": audio_filepath,
                    "duration": round(boundary[1] - boundary[0], 5),
                    "language": self.language
                }

                self.split_audio_based_on_timing(
                    main_filename=os.path.join(self.input_root_dir, entry['audio_filepath']),
                    split_filename=os.path.join(self.output_root_dir, self.DATASET, self.language, self.split, f"{os.path.basename(entry['audio_filepath']).split('.')[0]}-{idx:02}.wav"),
                    start=str(boundary[0]),
                    end=str(boundary[1]),
                )

                output_manifest_list.append(temp)

        # export the manifest file
        with open(os.path.join(self.output_root_dir, self.output_manifest_dir), 'w', encoding='utf-8') as fw:
            for item in output_manifest_list:
                fw.write(json.dumps(item)+'\n')


    def __call__(self) -> None:
        
        return self.filter_best_audio()


if __name__ == "__main__":

    SPLIT = 'test'
    LANGUAGE = 'ms'
    LANGUAGE_ = 'BM'
    INPUT_ROOT_DIR = f'/datasets/mms/transcribed_bibm_final/processed_{LANGUAGE_}/{SPLIT}_split'
    INPUT_MANIFEST_DIR = f'{SPLIT}_manifest.json' # f'test_manifest.json' # f'{SPLIT}_manifest.json'
    MIN_AUDIO_DURATION = 3
    LONG_AUDIO_MULTIPLIER = 10
    OUTPUT_ROOT_DIR = '/datasets/mms/transcribed_lid'
    OUTPUT_MANIFEST_DIR = f'{SPLIT}_manifest_{LANGUAGE}.json'
    
    # SPLIT = 'test'
    # LANGUAGE = 'en'
    # INPUT_ROOT_DIR = f'/datasets/mms/transcribed/mms_transcribed_batch_2/{SPLIT}_split'
    # INPUT_MANIFEST_DIR = f'{SPLIT}_manifest.json'
    # MIN_AUDIO_DURATION = 3
    # LONG_AUDIO_MULTIPLIER = 10
    # OUTPUT_ROOT_DIR = '/datasets/mms/transcribed_lid'
    # OUTPUT_MANIFEST_DIR = f'{SPLIT}_manifest_{LANGUAGE}.json'

    f = FilterBestAudio(
        input_root_dir=INPUT_ROOT_DIR,
        input_manifest_dir=INPUT_MANIFEST_DIR,
        min_audio_duration=MIN_AUDIO_DURATION,
        long_audio_multiplier=LONG_AUDIO_MULTIPLIER,
        output_root_dir=OUTPUT_ROOT_DIR,
        output_manifest_dir=OUTPUT_MANIFEST_DIR,
        split=SPLIT,
        language=LANGUAGE
    )()