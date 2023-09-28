import torch
import torch.nn as nn
import torch.optim as optim
import torch_optimizer
import yaml

from importlib import metadata
from ray import tune
from typing import Dict, Any, Optional, List

from kseg.data.lightning import DataModuleBase
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

    def get_unparsed_config(self) -> Dict[Any, Any]:
        """Get the unsubstituted config.

        Raises:
            DeprecationWarning: Raised if config version does not equal kseg
                version.

        Returns:
            Unparsed config.
        """
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Check if the config file is compatible with current kseg version
        try:
            if config['version'] != metadata.version('kseg'):
                raise DeprecationWarning(
                    'The given config file version may not be compatible!'
                )
        except metadata.PackageNotFoundError:
            # If the kseg package is not installed, this check cannot be done
            pass

        return config

    def parse_config(
        self,
        unparsed_config: Dict[Any, Any],
        class_weights: Optional[List[float]] = None,
    ) -> Dict[Any, Any]:
        """Substitute strings with the actual corresponding class.

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
                'WeightedMSELoss': WeightedMSELoss(
                    weight=torch.Tensor(class_weights)
                ),
            }
            replace_pattern.update(weighted_losses)

        parsed_config = self._parse_objects(unparsed_config, replace_pattern)
        parsed_config = self._parse_hparam_space(unparsed_config)
        return parsed_config

    def get_trial_specific_config(
        self,
        parsed_config: Dict[Any, Any],
        model_name: str,
        data_module: DataModuleBase,
        epochs: int,
        learning_rate: float,
        step_size: int,
        resume: bool,
        tuning: bool,
    ) -> Dict[Any, Any]:
        """Get the trail-specific configuration for training and model
            creation considering the passed parameters, chosen data module and
            the config file.

        Args:
            parsed_config: Parsed config.
            model_name: Name of the deep learning model for which the config should
                be returned.
            data_module: Data module which is used for the training.
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
                f'No config for model \'{model_name}\' available. '
                'Check config file.'
            )

        # Add information about shapes and domains. Write command line parameters
        # only if they were not set yet.
        defaults = {
            'input_shape': data_module.input_shape,
            'output_shape': data_module.label_shape,
            'input_domain': data_module.input_domain,
            'label_domain': data_module.label_domain,
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
                        new if item == old else item
                        for item in dictionary[key]
                    ]
                # If the value is another dict, recursively search inside it
                elif isinstance(value, dict):
                    self._parse_objects(value, replace_pattern)
        return dictionary

    def _parse_hparam_space(
        self, dictionary: Dict[Any, Any]
    ) -> Dict[Any, Any]:
        """Parse hyperparameter spaces into ray tune grid search functions.

        Args:
            dictionary: Dictionary which contains hyperparameter spaces.

        Note:
            All lists will be treated as tune grid_search function arguments.

        Returns:
            Config containing ray tune grid search functions.
        """
        for key, value in dictionary.items():
            if isinstance(value, list):
                dictionary[key] = tune.grid_search(value)
            # If the value is another dict, recursively search inside it
            elif isinstance(value, dict):
                self._parse_hparam_space(value)
        return dictionary
