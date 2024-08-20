#!/usr/bin/env python3

"""Recipe for training a LID system with MMS Maritime data.

To run this recipe, do the following:
> python train.py hparams/train_ecapa_tdnn.yaml

Author
------
 * Mirco Ravanelli 2021
 * Pavlo Ruban 2021
 * Nicholas Neo 2024
"""

import os
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

# Brain class for Language ID training
class LID(sb.Brain):

    def __init__(self, modules, opt_class, hparams, run_opts, checkpointer):
        super(LID, self).__init__(modules, opt_class, hparams, run_opts, checkpointer)
        self.writer = SummaryWriter(log_dir="logdir/")

    def prepare_features(self, wavs, stage):
        """Prepare the features for computation, including augmentation.

        Arguments
        ---------
        wavs : tuple
            Input signals (tensor) and their relative lengths (tensor).
        stage : sb.Stage
            The current stage of training.

        Returns
        -------
        feats : torch.Tensor
            Computed features.
        lens : torch.Tensor
            The length of the corresponding features.
        """
        wavs, lens = wavs

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, lens = self.hparams.wav_augment(wavs, lens)

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm_input(feats, lens)

        return feats, lens
    
    def compute_forward(self, batch, stage):
        """Runs all the computation of that transforms the input into the
        output probabilities over the N classes.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : torch.Tensor
            torch.Tensor that contains the posterior probabilities over the N classes.
        """

        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)

        # Compute features, embeddings and output
        feats, lens = self.prepare_features(batch.sig, stage)
        embeddings = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)

        return outputs, lens
    
    def compute_objectives(self, inputs, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        inputs : tensors
            The output tensors from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """

        predictions, lens = inputs

        targets = batch.language_encoded.data

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "wav_augment"):
                targets = self.hparams.wav_augment.replicate_labels(targets)
                if hasattr(self.hparams.lr_annealing, "on_batch_end"):
                    self.hparams.lr_annealing.on_batch_end(self.optimizer)

        loss = self.hparams.compute_cost(predictions, targets)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, targets, lens)

        return loss
    
    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            self.writer.add_scalar('Loss/train', stage_loss, epoch)

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error": self.error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:
            self.writer.add_scalar('Loss/valid', stage_loss, epoch)
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {
                    "Epoch": epoch, 
                    "lr": old_lr
                },
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["error"])

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def on_evaluate_end(self, test_data):
        """Hook to log metrics at the end of evaluation."""
        test_loss = self.evaluate(test_data)
        self.writer.add_scalar('Loss/test', test_loss, 0)
        # Add any other test metrics you want to log here
        
        # Close the writer
        self.writer.close()

def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    the `train.json`, `valid.json` manifest files are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """
    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'spk01': 0, 'spk02': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`.
        """
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("language")
    @sb.utils.data_pipeline.provides("language", "language_encoded")
    def label_pipeline(language):
        """Defines the pipeline to process the input language."""
        yield language
        language_encoded = label_encoder.encode_label_torch(language)
        yield language_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "dev": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }

    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "language_encoded"],
        )

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mapping.
    language_encoder_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=language_encoder_file,
        from_didatasets=[datasets["train"]],
        output_key="language",
    )

    return datasets, label_encoder

# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments.
    # hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    hparams_file, run_opts, overrides = sb.parse_arguments(["/speechbrain/recipes/mms/lang_id/hparams/train_ecapa_tdnn.yaml"])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation for augmentation
    print("Noise and RIR")
    print(hparams["prepare_noise_data"])
    print(type(hparams["prepare_noise_data"]))

    print(hparams["prepare_rir_data"])
    print(type(hparams["prepare_rir_data"]))

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

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
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
