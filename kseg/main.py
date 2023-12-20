import os

import click
import pytorch_lightning as pl
import ray

from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from typing import Dict, Any


from kseg.data.lightning import (
    DataModuleBase,
    KneeDataModule,
    CNSLymphomaSSDataModule,
    CNSLymphomaTissueDataModule,
    OasisTissueDataModule,
    UCSF51LesionDataModule,
    UPennGBMSSDataModule,
    UPennGBMTumorDataModule,
)
from kseg.model.lightning import LitModel
from kseg.utils.config_handler import ConfigHandler


@click.group()
def main():
    pass


@main.command()
@click.argument('model-name', default='MLP')
@click.argument('data-name', default='UPennGBMSS')
@click.option('--batch-size', default=4)
@click.option('--epochs', default=100)
@click.option('--learning-rate', '--lr', default=0.01)
@click.option('--step-size', default=300)
@click.option('--input_domain', '--id', default='kspace')
@click.option('--label-domain', '--ld', default='kspace')
@click.option('--num-cpus', '--cpus', default=32)
@click.option('--num_gpus', '--gpus', default=1)
@click.option('--skip-checkpoints', '--sc', is_flag=True, default=False)
@click.option(
    '--output_dir', '--out', default='.', type=click.Path(exists=True)
)
@click.option(
    '--datasets-path', default='./datasets', type=click.Path(exists=True)
)
@click.option(
    '--config-path', default='./config.yml', type=click.Path(exists=True)
)
@click.option('--resume', is_flag=True, default=False)
@click.option('--tuning', is_flag=True, default=False)
def train(
    model_name: str,
    data_name: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    step_size: int,
    input_domain: str,
    label_domain: str,
    num_cpus: int = 32,
    num_gpus: int = 1,
    skip_checkpoints: bool = False,
    output_dir: str = '.',
    datasets_path: str = './datasets/',
    config_path: str = './config.yml',
    resume: bool = False,
    tuning: bool = False,
) -> None:
    """Starts the training routine using the passed parameters.

    Args:
        model_name: Name of the deep learning model to be trained.
        data_name: Name of the data to be used for training and validation.
        batch_size: Batch size used for training.
        epochs: Number of training epochs.
        learning_rate: Learning rate used for training.
        step_size: Step size for the scheduler.
        input_domain: Target domain of inputs ('kspace', 'wavelet' or 'pixel').
        label_domain: Target domain of labels ('kspace', 'wavelet' or 'pixel').
        num_cpus: Number of CPUs used for the training or whole tuning task.
            Defaults to 32.
        num_gpus: Number of GPUs used for the training or whole tuning task.
            Defaults to 1.
        skip_checkpoints: Whether to skip creating model checkpoints or not.
            Defaults to False.
        output_dir: Directory of results and checkpoints. Defaults to '.'.
        datasets_path: Path to the datasets directory.
            Defaults to './datasets/'.
        config_path: Path to the config file. Defaults to './config.yml'.
        resume: Whether the training shall be resumed or not.
            Defaults to False.
        tuning: Whether to tune hyper parameter or not. Defaults to False.

    Note:
        If the ray hparams scheduler or an early stopping criterium terminates
        the training, the given epoch may be superseded.

    Raises:
        NotImplementedError: If the resume flag is set.
        ValueError: If the specified model or data is not supported.
    """
    if resume:
        raise NotImplementedError('Resume currently not supported.')

    data_modules = {
        'Knee': KneeDataModule,
        'CNSLymphomaSS': CNSLymphomaSSDataModule,
        'CNSLymphomaTissue': CNSLymphomaTissueDataModule,
        'OasisTissue': OasisTissueDataModule,
        'UCSF51Lesion': UCSF51LesionDataModule,
        'UPennGBMSS': UPennGBMSSDataModule,
        'UPennGBMTumor': UPennGBMTumorDataModule,
    }

    if data_name in data_modules:
        data_module = data_modules[data_name](
            batch_size=batch_size,
            input_domain=input_domain,
            label_domain=label_domain,
            datasets_dir=datasets_path,
        )
    else:
        raise ValueError(f'Data module for {data_name} is not defined.')

    data_module.prepare_data()
    data_module.setup()

    # Get config dict from YAML file
    config_handler = ConfigHandler(config_path)
    config = config_handler.get_config(
        model_name,
        epochs,
        learning_rate,
        step_size,
        data_module.input_shape,
        data_module.label_shape,
        data_module.input_domain,
        data_module.label_domain,
        data_module.class_weights,
        resume,
        tuning,
    )

    # Initialize the ray cluster with defined resources
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus)

    # We're using grid search only, we sample each value once
    num_samples = 1

    scheduler = ASHAScheduler(
        max_t=epochs, grace_period=100, reduction_factor=2
    )

    # Display either hparams or standard parameter in CLIReporter
    if tuning:
        # Replace grid search array by ray grid search object
        for key, value in config.items():
            if isinstance(value, list):
                config[key] = tune.grid_search(value)
        parameter_columns = [
            k
            for k, v in config.items()
            if isinstance(v, dict) and 'grid_search' in v
        ]
    else:
        parameter_columns = ['lr', 'criterion']

    reporter = CLIReporter(
        parameter_columns=parameter_columns,
        metric_columns=['val_loss', 'val_avg_dice', 'training_iteration'],
        max_report_frequency=15,  # Report maximum every 15 sec
    )

    train_fn_with_parameters = tune.with_parameters(
        train_worker,
        model_name=model_name,
        data_module=data_module,
        num_epochs=epochs,
        num_gpus=1,
        skip_checkpoints=skip_checkpoints,
    )

    # Train worker run not in parallel so assign all resources to each trail
    resources_per_trial = {'cpu': num_cpus, 'gpu': num_gpus}

    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_parameters, resources=resources_per_trial
        ),
        tune_config=tune.TuneConfig(
            metric='val_avg_dice',
            mode='max',
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            name='pytorch_logs',
            progress_reporter=reporter,
            storage_path=output_dir,
        ),
        param_space=config,
    )
    results = tuner.fit()

    print('Best hyperparameters found were: ', results.get_best_result().config)


@main.command()
@click.argument('data-name', default='UPennGBMSS')
@click.argument('checkpoint_path', type=click.Path(exists=True))
@click.option('--batch-size', default=4)
@click.option('--num_gpus', '--gpus', default=1)
@click.option(
    '--config-path', default='./config.yml', type=click.Path(exists=True)
)
def test(
    data_name: str,
    checkpoint_path: str,
    batch_size: int,
    num_gpus: int = 1,
    config_path: str = './config.yml',
) -> None:
    """Starts the testing routine using the passed parameters.

    Args:
        data_name: Name of the data to be used for testing.
        checkpoint_path: Path to the saved model checkpoint.
        batch_size: Batch size used for testing.
        num_gpus: Number of GPUs used for the testing. Defaults to 1.

    Raises:
        ValueError: Raised if specified data is unknown or given checkpoint
            path is invalid.
    """
    # Get absolute path, convert it to string and split at '/'
    checkpoint_absolute_path = Path(checkpoint_path).resolve()
    checkpoint_path_parts = str(checkpoint_absolute_path).split('/')

    # Find the index of 'pytorch_logs' and take all parts up to worker dir
    try:
        idx = checkpoint_path_parts.index('pytorch_logs')
        train_worker_path = '/'.join(checkpoint_path_parts[: idx + 2]) + '/'
    except ValueError:
        raise ValueError(f'Given checkpoint path is invalid.')

    # Load model and weights from checkpoint and set model to eval mode
    model = LitModel.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Get config dict from YAML file without class substitutions
    config_handler = ConfigHandler(config_path)
    unparsed_config = config_handler.get_unparsed_config()

    # Initialize data module with domain config from checkpoint
    data_modules = {
        'Knee': KneeDataModule,
        'CNSLymphomaSS': CNSLymphomaSSDataModule,
        'CNSLymphomaTissue': CNSLymphomaTissueDataModule,
        'OasisTissue': OasisTissueDataModule,
        'UCSF51Lesion': UCSF51LesionDataModule,
        'UPennGBMSS': UPennGBMSSDataModule,
        'UPennGBMTumor': UPennGBMTumorDataModule,
    }
    if data_name in data_modules:
        data_module = data_modules[data_name](
            batch_size=batch_size,
            input_domain=model.input_domain,
            label_domain=model.label_domain,
            dataset_dir=unparsed_config['datasets'][data_name],
        )
    else:
        raise ValueError(f'Data module for {data_name} is not defined.')

    data_module.prepare_data()
    data_module.setup()

    # Initialize the Trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=num_gpus,
        logger=TensorBoardLogger(
            save_dir=train_worker_path,  # Store test results in train worker
            name='',
            version='.',
            default_hp_metric=False,
        ),
        enable_progress_bar=False,
    )

    # Test the model
    trainer.test(model, datamodule=data_module)


def train_worker(
    config: Dict[str, Any],
    model_name: str,
    data_module: DataModuleBase,
    num_epochs: int = 100,
    num_gpus: int = 1,
    skip_checkpoints: bool = False,
    checkpoint_path: str = './checkpoints',
) -> None:
    """Worker process for training.

    Args:
        config: Model and training config.
        model_name: Model name.
        data_module: Data Module which provides data for training.
        num_epochs: Maximum number of training epochs. Defaults to 10.
        num_gpus: Number of GPUs to be used. Defaults to 1.
        skip_checkpoints: Whether to skip creating model checkpoints or not.
            Defaults to False.
        checkpoint_path: Path for saving checkpoints.
            Defaults to './checkpoints'.
    """
    model = LitModel(model_name, config)

    # Monitor the average validation dice score for multi-class tasks,
    # otherwise use the dice sore of the object class (class 1)
    if data_module.num_classes > 2:
        monitor_metric = 'val_avg_dice'
    else:
        monitor_metric = 'val_dice_class_1'

    # Set up callbacks
    callbacks = [
        TuneReportCallback(
            {'val_loss': 'val_loss', 'val_avg_dice': 'val_avg_dice'},
            on='validation_end',
        ),
        pl.callbacks.early_stopping.EarlyStopping(
            monitor=monitor_metric, patience=10, mode='max'
        ),
    ]
    # Skip checkpointing if the corresponding flag is set
    if not skip_checkpoints:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=checkpoint_path,
                filename='best_{epoch}',
                monitor=monitor_metric,
                save_last=True,
                save_top_k=1,
                mode='max',
                save_weights_only=False,
            )
        )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=num_gpus,
        max_epochs=num_epochs,
        logger=TensorBoardLogger(
            save_dir=os.getcwd(),
            name='',
            version='.',
            default_hp_metric=False,
        ),
        enable_progress_bar=False,
        enable_checkpointing=not (skip_checkpoints),
        callbacks=[*callbacks],
    )
    trainer.fit(model=model, datamodule=data_module)


if __name__ == '__main__':
    main()
