#!/usr/bin/env python3
"""Recipe for training a digit recognition system. The proposed task classifies
10 digits using AudioMNIST.
To run this recipe, do the following:
> python train.py train.yaml
To read the code, first scroll to the bottom to see the "main" code.
This gives a high-level overview of what is going on, while the
Brain class definition provides the details of what happens
for each batch during training. The dataset for AudioMNIST is available in the file
recordings.tar.gz in this directory
Authors
 * Ada Tur, 2023
 * Mirco Ravanelli, 2023
 * Cem Subakan, 2023
"""
import os
import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from prepare_audiomnist import prepare_audiomnist
import torch.nn.functional as F


# Brain class for speech enhancement training
class MNISTBrain(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""

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
                predictions : Tensor
                    Tensor that contains the posterior probabilities over the N classes.
                """
        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)

        # Compute features, embeddings, and predictions
        feats = self.prepare_features(batch.sig, stage)
        embeddings = self.modules.embedding_model(feats)
        predictions = self.modules.classifier(embeddings)

        return predictions

    def prepare_features(self, wavs, stage):
        """Prepare the features for computation, including augmentation.
               Arguments
               ---------
               wavs : tuple
                   Input signals (tensor) and their relative lengths (tensor).
               stage : sb.Stage
                   The current stage of training.
               """

        wavs, lens = wavs
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)
        # Reshaping features to fit ResNet input shape of 3x224x224
        size = feats.shape[1]
        feats_pad = F.pad(feats, pad=(0, 224 - self.n_mels, 0, 224 - size, 0, 0))
        feats_pad = torch.unsqueeze(feats_pad, dim=1).repeat(1, 3, 1, 1)
        return feats_pad, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.
                Arguments
                ---------
                predictions : tensor
                    The output tensor from `compute_forward`.
                batch : PaddedBatch
                    This batch object contains all the relevant tensors for computation.
                stage : sb.Stage
                    One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
                Returns
                -------
                loss : torch.Tensor
                    A one-element tensor used for backpropagating the gradient.
                """
        predictions = torch.unsqueeze(predictions, 1)
        _, lens = batch.sig
        digit, _ = batch.digit_encoded

        # Compute the cost function
        loss = sb.nnet.losses.nll_loss(predictions, digit, lens)
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, digit)
        if stage == sb.Stage.TEST:
            print(predictions)
            print(digit)
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

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

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

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error": self.error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
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


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
        It also defines the data processing pipeline through user-defined functions.
        We expect `prepare_audiomnist` to have been called before this,
        so that the `train.json`, `valid.json`,  and `valid.json` manifest files
        are available.
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
    # of the observed label a unique index (e.g, '0': 0, '1': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        return (sig)

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("digit")
    @sb.utils.data_pipeline.provides("digit", "digit_encoded")
    def label_pipeline(digit):
        """Defines the pipeline to process the input digit."""
        yield digit
        digit_encoded = label_encoder.encode_label_torch(digit)
        yield digit_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    hparams["dataloader_options"]["shuffle"] = False
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "digit_encoded"],
        )

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mapping.
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="digit",
    )

    return datasets


# Recipe begins!
if __name__ == "__main__":
    # Reading command line arguments.

    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

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

    # Data preparation, to be run on only one process.
    sb.utils.distributed.run_on_main(
        prepare_audiomnist,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "save_json_test": hparams["test_annotation"],
            "split_ratio": [80, 10, 10],
        },
    )

    # Create dataset objects "train", "valid", and "test".
    datasets = dataio_prep(hparams)

    # Initialize the Brain object to prepare for mask training.
    mnist_brain = MNISTBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    print(datasets["valid"][0]["sig"].shape)
    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    mnist_brain.fit(
        epoch_counter=mnist_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    test_stats = mnist_brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["dataloader_options"],
    )
