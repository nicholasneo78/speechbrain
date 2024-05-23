from speechbrain.inference.VAD import VAD

audio_filepath = '/datasets/mms/transcribed/mms_transcribed_batch_2/train_split/splits/CHDIR_497_2022-04-30-00603352-00653981.wav'

VAD = VAD.from_hparams(source='/models/vad-crdnn-libriparty')

boundaries = VAD.get_speech_segments(audio_filepath)
print(type(boundaries))
print(boundaries)
print()

prob_chunks = VAD.get_speech_prob_file(audio_filepath)
prob_th = VAD.apply_threshold(prob_chunks, activation_th=0.5, deactivation_th=0.25).float()
boundaries = VAD.get_boundaries(prob_th)
print(boundaries)
print()

boundaries = VAD.energy_VAD(audio_filepath, boundaries, activation_th=0.8, deactivation_th=0.0)
boundaries = VAD.merge_close_segments(boundaries, close_th=0.250)
boundaries = VAD.remove_short_segments(boundaries, len_th=0.250)

boundaries = VAD.double_check_speech_segments(boundaries, audio_filepath,  speech_th=0.5)
print(boundaries)

boundaries_list = [[round(value, 5) for value in value_list] for value_list in boundaries.tolist() if value_list[1]-value_list[0] >= 3]

duration_list = [round(value[1]-value[0], 5) for value in boundaries_list]
print(boundaries_list)
print(duration_list)