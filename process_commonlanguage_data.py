from recipes.CommonLanguage.common_language_prepare import prepare_common_language

data_folder = '/datasets/CommonLanguage/common_voice_kpd'
save_folder = '/datasets/CommonLanguage'

prepare_common_language(
    data_folder=data_folder,
    save_folder=save_folder,
    skip_prep=False
)