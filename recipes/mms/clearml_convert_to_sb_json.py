from clearml import Task, Dataset
import hydra

@hydra.main(version_base=None, config_path='lang_id/hparams', config_name='clearml')
def main(cfg) -> None:

    task = Task.init(
        project_name=cfg.task_config.project_name,
        task_name=cfg.convert_nemo_to_sb.task_name,
        output_uri=cfg.task_config.output_url,
    )

    task.set_base_docker(
        docker_image=cfg.task_config.docker_image,
    )

    task.execute_remotely(
        queue_name=cfg.task_config.convert_nemo_to_sb.queue,
        exit_process=True
    )

    from convert_to_sb_json import ConvertToSBJson
    import os
    import logging

    # Setup logging in a nice readable format
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
        datefmt='%H:%M:%S'
    )

    # load all the datasets
    dataset_train_path = Dataset.get(
        dataset_id=cfg.convert_nemo_to_sb.data.split.train.input_dataset_task_id
    ). get_local_copy()

    dataset_dev_path = Dataset.get(
        dataset_id=cfg.convert_nemo_to_sb.data.split.dev.input_dataset_task_id
    ). get_local_copy()

    dataset_test_path = Dataset.get(
        dataset_id=cfg.convert_nemo_to_sb.data.split.test.input_dataset_task_id
    ). get_local_copy()

    _ = ConvertToSBJson(
        root_dir=dataset_train_path,
        input_manifest_dir=cfg.convert_nemo_to_sb.split.train.input_manifest_path,
        output_manifest_dir=cfg.convert_nemo_to_sb.split.train.output_manifest_path,
    )()

    _ = ConvertToSBJson(
        root_dir=dataset_dev_path,
        input_manifest_dir=cfg.convert_nemo_to_sb.split.dev.input_manifest_path,
        output_manifest_dir=cfg.convert_nemo_to_sb.split.dev.output_manifest_path,
    )()

    _ = ConvertToSBJson(
        root_dir=dataset_test_path,
        input_manifest_dir=cfg.convert_nemo_to_sb.split.test.input_manifest_path,
        output_manifest_dir=cfg.convert_nemo_to_sb.split.test.output_manifest_path,
    )()

    logging.getLogger('INFO').info(f'Train Path: {dataset_train_path}')
    logging.getLogger('INFO').info(f'Train Path: {dataset_dev_path}')
    logging.getLogger('INFO').info(f'Train Path: {dataset_test_path}')

    dataset_train = Dataset.create(
        dataset_project=cfg.dataset_config.dataset_project,
        dataset_name=cfg.convert_nemo_to_sb.split.train.output_dataset_name,
        parent_datasets=[cfg.convert_nemo_to_sb.split.train.input_dataset_task_id],
        output_uri=cfg.dataset_config.output_url
    )

    dataset_dev = Dataset.create(
        dataset_project=cfg.dataset_config.dataset_project,
        dataset_name=cfg.convert_nemo_to_sb.split.dev.output_dataset_name,
        parent_datasets=[cfg.convert_nemo_to_sb.split.dev.input_dataset_task_id],
        output_uri=cfg.dataset_config.output_url
    )

    dataset_test = Dataset.create(
        dataset_project=cfg.dataset_config.dataset_project,
        dataset_name=cfg.convert_nemo_to_sb.split.test.output_dataset_name,
        parent_datasets=[cfg.convert_nemo_to_sb.split.test.input_dataset_task_id],
        output_uri=cfg.dataset_config.output_url
    )

    # save artifacts
    dataset_train_task = Task.get_task(dataset_train.id)
    dataset_dev_task = Task.get_task(dataset_dev.id)
    dataset_test_task = Task.get_task(dataset_test.id)

    # upload artifacts and files
    dataset_train_task.upload_artifacts(name='train_manifest_sb.json', artifact_object=cfg.convert_nemo_to_sb.split.train.output_manifest_path)
    dataset_dev_task.upload_artifacts(name='dev_manifest_sb.json', artifact_object=cfg.convert_nemo_to_sb.split.dev.output_manifest_path)
    dataset_test_task.upload_artifacts(name='test_manifest_sb.json', artifact_object=cfg.convert_nemo_to_sb.split.test.output_manifest_path)

    dataset_train.add_files(path=cfg.convert_nemo_to_sb.split.train.output_manifest_path, local_base_folder='root/')
    dataset_dev.add_files(path=cfg.convert_nemo_to_sb.split.dev.output_manifest_path, local_base_folder='root/')
    dataset_test.add_files(path=cfg.convert_nemo_to_sb.split.test.output_manifest_path, local_base_folder='root/')

    dataset_train.upload(output_url=cfg.dataset_config.output_url)
    dataset_train.finalize()
    dataset_dev.upload(output_url=cfg.dataset_config.output_url)
    dataset_dev.finalize()
    dataset_test.upload(output_url=cfg.dataset_config.output_url)
    dataset_test.finalize()

    # get the created dataset id
    logging.getLogger('INFO').info(f"ClearML Dataset ID - Train: {dataset_train.id}")
    logging.getLogger('INFO').info(f"ClearML Dataset ID - Dev: {dataset_dev.id}")
    logging.getLogger('INFO').info(f"ClearML Dataset ID - Test: {dataset_test.id}")

    logging.getLogger('INFO').info("DONE!")
