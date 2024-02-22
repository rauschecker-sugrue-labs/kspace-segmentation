import torch
import torch.nn as nn
import torch.optim as optim
import torch_optimizer
import yaml

from importlib import metadata
from typing import Dict, Any, Optional, List, Tuple

from kseg.model.loss import (
    DiceLoss,
    FocalLoss,
    HighFreqMSELoss,
    WeightedMSELoss,
)


class ConfigHandler:
    def __init__(self, config_path) -> None:
        """Initializaton of the config handler.

        Args:
            config_path: Path to the config file.
        """
        self.config_path = config_path

    def get_config(
        self,
        model_name: str,
        epochs: int,
        learning_rate: float,
        step_size: int,
        input_shape: Tuple[int],
        label_shape: Tuple[int],
        input_domain: str,
        label_domain: str,
        class_weights: List[float],
        resume: bool,
        tuning: bool,
    ) -> Dict[Any, Any]:
        """Get training configuration from config file.

        Args:
            model_name: Name of the model to train and test.
            epochs: Number of training epochs.
            learning_rate: Learning rate.
            step_size: Step size.
            input_shape: Shape of the input.
            label_shape: Shape of the label.
            input_domain: Domain of the input.
            label_domain: Domain of the label.
            class_weights: Class weights for the loss.
            resume: Whether to resume an already existing training.
            tuning: Whether hyperparameter tuning shall be used.

        Raises:
            DeprecationWarning: Raised if config version does not equal kseg
                version.
            ValueError: Raised if a model name is given for which there is no
                configuration specified.

        Returns:
            Config.
        """
        with open(self.config_path, 'r') as file:
            unparsed_config = yaml.safe_load(file)

        # Check if the config file is compatible with current kseg version
        try:
            if unparsed_config['version'] != metadata.version('kseg'):
                raise DeprecationWarning(
                    'The given config file version may not be compatible!'
                )
        except metadata.PackageNotFoundError:
            # If the kseg package is not installed, this check cannot be done
            pass

        config = self._parse_config(unparsed_config, class_weights)
        config = self._get_trial_specific_config(
            config,
            model_name,
            input_shape,
            label_shape,
            input_domain,
            label_domain,
            epochs,
            learning_rate,
            step_size,
            resume,
            tuning,
        )
        return config

    def _parse_config(
        self,
        unparsed_config: Dict[Any, Any],
        class_weights: Optional[List[float]] = None,
    ) -> Dict[Any, Any]:
        """Substitute strings with the actual corresponding classes.

        Args:
            unparsed_config: Unparsed config.
            class_weights: Class weights for weighted loss. Defaults to None.

        Returns:
            Parsed config.
        """
        replace_pattern = {
            # Loss fucntions
            'BCELoss': nn.BCELoss(),
            'BCEWithLogitsLoss': nn.BCEWithLogitsLoss(),
            'CrossEntropyLoss': nn.CrossEntropyLoss(),
            'DiceLoss': DiceLoss(),
            'FocalLoss': FocalLoss(),
            'HighFreqMSELoss': HighFreqMSELoss(),
            'MSELoss': nn.MSELoss(),
            # Optimizer
            'Adam': optim.Adam,
            'Adagrad': optim.Adagrad,
            'Lamb': torch_optimizer.Lamb,
            'SGD': optim.SGD,
            # Scheduler
            'StepLR': optim.lr_scheduler.StepLR,
        }
        # Handle weighted loss functions if class_weights are provided
        if class_weights is not None:
            weighted_losses = {
                'WeightedCrossEntropyLoss': nn.CrossEntropyLoss(
                    weight=torch.Tensor(class_weights)
                ),
                'WeightedMSELoss': WeightedMSELoss(weight=torch.Tensor(class_weights)),
            }
            replace_pattern.update(weighted_losses)

        parsed_config = self._parse_objects(unparsed_config, replace_pattern)
        return parsed_config

    def _get_trial_specific_config(
        self,
        parsed_config: Dict[Any, Any],
        model_name: str,
        input_shape: Tuple[int],
        label_shape: Tuple[int],
        input_domain: str,
        label_domain: str,
        epochs: int,
        learning_rate: float,
        step_size: int,
        resume: bool,
        tuning: bool,
    ) -> Dict[Any, Any]:
        """Get the trial-specific configuration for training and model
        creation considering the passed parameters, chosen data module and
        the config file.

        Args:
            parsed_config: Parsed config.
            model_name: Name of the deep learning model for which the config should
                be returned.
            epochs: Number of training epochs.
            learning_rate: Learning rate for the training.
            step_size: Step size for the training.
            resume: Whether to resume an existing training progress or not.
            tuning: Whether to use hyperparameter tuning or not.

        Raises:
            ValueError: Raised if unknown model name was given or there is no
                (tuning) config for this model available.

        Returns:
            Configuration for training and model creation.
        """
        config = {}

        # Load general training config
        config.update(parsed_config['training'])

        # Check if model name exists in models config and get its configuration
        if model_name in list(parsed_config['models'].keys()):
            config.update(parsed_config['models'][model_name])
            # Check if model name exists in tuning config and get its configuration
            if tuning and model_name in list(parsed_config['tuning'].keys()):
                config.update(parsed_config['tuning'][model_name])
        else:
            raise ValueError(
                f'No config for model \'{model_name}\' available. ' 'Check config file.'
            )

        # Add information about shapes and domains. Write command line parameters
        # only if they were not set yet.
        defaults = {
            'input_shape': input_shape,
            'output_shape': label_shape,
            'input_domain': input_domain,
            'label_domain': label_domain,
            'epochs': epochs,
            'lr': learning_rate,
            'step_size': step_size,
            'resume': resume,
        }
        config = defaults | config  # Merges two dicts (right overwrites left)

        return config

    def _parse_objects(
        self, dictionary: Dict[Any, Any], replace_pattern: Dict[Any, Any]
    ) -> Dict[Any, Any]:
        """Replaces strings or numbers with given objects or classes.

        Args:
            dictionary: Dictionary containing the elements to be replaced.
            replace_pattern: Rules to replace elements.

        Returns:
            Config containing pytorch-related objects or classes.
        """
        for old, new in replace_pattern.items():
            for key, value in dictionary.items():
                if value == old:
                    dictionary[key] = new
                # If the value is a list, iterate through
                elif isinstance(value, list):
                    dictionary[key] = [
                        new if item == old else item for item in dictionary[key]
                    ]
                # If the value is another dict, recursively search inside it
                elif isinstance(value, dict):
                    self._parse_objects(value, replace_pattern)
        return dictionary
