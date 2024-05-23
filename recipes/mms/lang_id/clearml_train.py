from clearml import Task, Dataset, Model
import hydra
from hyperpyyaml import load_hyperpyyaml

@hydra.main(version_base=None, config_path='hparams', config_name='clearml')
def main(cfg) -> None:

    """
    main function to do the training of LID model
    """

    import speechbrain as sb
    import sys

    # load the speechbrain hyperpyyaml config file
    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    task = Task.init(
        project_name=cfg.task_config.project_name,
        task_name=cfg.finetuning.task_name,
        output_uri=cfg.task_config.output_url,
    )

    task.set_base_docker(
        docker_image=cfg.task_config.docker_image,
    )

    task.execute_remotely(
        queue_name=cfg.task_config.finetuning.queue,
        exit_process=True
    )

    from train import LID, dataio_prep
    import os 
    import sys
    import logging
    import shutil
    import speechbrain as sb

    # Setup logging in a nice readable format
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # obtain all the required datasets
    dataset_train_path = Dataset.get(
        dataset_id=cfg.finetuning.data.train.dataset_task_id
    ). get_local_copy()

    dataset_dev_path = Dataset.get(
        dataset_id=cfg.finetuning.data.dev.dataset_task_id
    ). get_local_copy()

    dataset_test_path = Dataset.get(
        dataset_id=cfg.finetuning.data.test.dataset_task_id
    ). get_local_copy()

    dataset_noise_path = Dataset.get(
        dataset_id=cfg.finetuning.augmentation.noise.dataset_task_id
    ). get_local_copy()

    dataset_rir_path = Dataset.get(
        dataset_id=cfg.finetuning.augmentation.rir.dataset_task_id
    ). get_local_copy()

    # load and unzip the model
    model_pretrained_path_zip = Model(model_id=cfg.finetuning.model.model_task_id).get_local_copy()
    shutil.unpack_archive(model_pretrained_path_zip, os.path.dirname(model_pretrained_path_zip))
    model_pretrained_path = os.path.dirname(model_pretrained_path_zip)

    logging.getLogger('INFO').info(f'Model pretrained path: {model_pretrained_path}')
    logging.getLogger('INFO').info(f'Files in model pretrained path: {os.listdir(model_pretrained_path)}')

    # load all the speechbrain dependencies here

    # some overriding of config for the clearml path
    hparams['train_data_folder'] = dataset_train_path
    hparams['dev_data_folder'] = dataset_dev_path
    hparams['test_data_folder'] = dataset_test_path
    hparams['data_folder_noise'] = dataset_noise_path
    hparams['data_folder_rir'] = dataset_rir_path

    hparams['embedding_model_folder'] = model_pretrained_path
    
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation for augmentation
    sb.utils.distributed.run_on_main(hparams["prepare_noise_data"])
    sb.utils.distributed.run_on_main(hparams["prepare_rir_data"])

    # Create dataset objects "train", "dev", and "test" and language_encoder
    datasets, label_encoder = dataio_prep(hparams)

    # Fetch and load pretrained modules
    sb.utils.distributed.run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected()

    # Initialize the Brain object to prepare for mask training.
    lid_brain = LID(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    lid_brain.fit(
        epoch_counter=lid_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["dev"],
        train_loader_kwargs=hparams["train_dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    test_stats = lid_brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )

    # save the finetuned model as a task artifact
    shutil.make_archive(
        base_name=hparams['output_folder'],
        format='zip',
        root_dir='.',
        base_dir=hparams['output_folder']
    )

    # upload the model
    task.update_output_model(
        model_path=hparams['output_folder']+'.zip',
        name=os.path.dirname(hparams['output_folder']),
        model_name=os.path.dirname(hparams['output_folder'])
    )

    logging.getLogger('STATUS').info(f'ClearML Finetuned Model ID: {task.id}')
    logging.getLogger('STATUS').info('DONE!')

    task.close()

if __name__ == '__main__':
    main()
