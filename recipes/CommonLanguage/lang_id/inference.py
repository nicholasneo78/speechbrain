import torchaudio
from speechbrain.inference import EncoderClassifier

classifier = EncoderClassifier.from_hparams(
    source="/datasets/CommonLanguage/results/ECAPA-TDNN/1986/save/CKPT+2024-03-31+13-53-01+00", 
    hparams_file="/datasets/CommonLanguage/results/ECAPA-TDNN/1986/save/CKPT+2024-03-31+13-53-01+00/hyperparams.yaml",#"/datasets/CommonLanguage/models",#"/models/lang-id-voxlingua107-ecapa",#"/datasets/CommonLanguage/models",
    # savedir="/datasets/CommonLanguage/results/ECAPA-TDNN/1987/save/CKPT+2024-04-01+23-48-35+00",  #"/datasets/CommonLanguage/models",#"/models/lang-id-voxlingua107-ecapa",#"/datasets/CommonLanguage/models"
)

# japanese
signal = classifier.load_audio('/datasets/CommonLanguage/common_voice_kpd/Japanese/test/jpn_tst_sp_1/common_voice_ja_20721115.wav', savedir='/temp')
prediction = classifier.classify_batch(
    signal
)
print(prediction)
print()
print()

# english
signal = classifier.load_audio('/datasets/test_2.wav', savedir='/temp')
prediction = classifier.classify_batch(
    signal
)
print(prediction)
print()
print()

# indo
signal = classifier.load_audio('/datasets/test_3.wav', savedir='/temp')
prediction = classifier.classify_batch(
    signal
)
print(prediction)
