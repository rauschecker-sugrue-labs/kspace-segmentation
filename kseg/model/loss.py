import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from einops import reduce
from typing import Union, Tuple


class HighFreqMSELoss(nn.Module):
    def __init__(
        self,
        reduction: str = 'mean',
        fourier_plane_enc: str = 'sagittal',
    ) -> None:
        """Initialization of the high frequency MSE loss.

        Args:
            reduction: Method to combine loss of each matrix entry.
                Defaults to 'mean'.
            fourier_plane_enc: Determines the plane along which the FFT is
                done. Defaults to 'sagittal'.
        """
        super(HighFreqMSELoss, self).__init__()
        self.reduction = reduction
        self.fourier_plane_enc = fourier_plane_enc

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Union[float, torch.Tensor]:
        """Calculates the high frequency MSE loss.

        Args:
            inputs: Predicted values.
            targets: Ground truth values.

        Returns:
            Combined loss or loss for each matrix entry.
        """
        if not (inputs.size() == targets.size()):
            print(
                f"""Using a prediction size ({inputs.size()}) that is different
                to the ground truth size ({targets.size()}). This will likely
                lead to incorrect results due to broadcasting."""
            )
        # Compute the squared error
        diff = torch.square(inputs - targets)

        # Generate frequency-dependent weights
        # Create weights plane with shape of Fourier encoded plane and copy it
        # to get same shape as diff tensor
        if self.fourier_plane_enc == 'sagittal':
            enc_shape = diff.shape[-3:-1]
            freq_weights = self.get_freq_weights(enc_shape, diff.device.type)
            freq_weights = freq_weights[np.newaxis, np.newaxis, :, :, np.newaxis]
        elif self.fourier_plane_enc == 'coronal':
            enc_shape = (diff.shape[1], diff.shape[3])
            freq_weights = self.get_freq_weights(enc_shape, diff.device.type)
            freq_weights = freq_weights[np.newaxis, :, np.newaxis, :, np.newaxis]
        elif self.fourier_plane_enc == 'axial':
            enc_shape = diff.shape[1:3]
            freq_weights = self.get_freq_weights(enc_shape, diff.device.type)
            freq_weights = freq_weights[np.newaxis, :, :, np.newaxis, np.newaxis]
        else:
            raise ValueError('Unknown Fourier plane encoding')

        # Multiply the differences by the weights
        if self.reduction == 'mean':
            return torch.mean(diff * freq_weights)

        return diff * freq_weights

    def get_freq_weights(self, shape: Tuple[int], device: str) -> torch.Tensor:
        """Creates a grid to weight the frequency components in k-space.

        Args:
            shape: Shape of the k-space which will be weighted.
            device: Device on which a tensor will be allocated
                ('cpu', 'cuda' or 'mps').

        Returns:
            Grid with high weights for high frequency components.
        """
        # Generate a grid
        cy, cx = (shape[0] - 1) / 2, (shape[1] - 1) / 2  # center coordinates
        y = torch.arange(0, shape[0]).to(device)
        x = torch.arange(0, shape[1]).to(device)
        y, x = torch.meshgrid(y, x)

        # Compute the frequency magnitude from the center
        freq_magnitude = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        # Normalize the frequency magnitude to [0, 1]
        max_distance = (
            np.sqrt((shape[0] - 1) ** 2 + (shape[1] - 1) ** 2) / 2
        )  # Maximum possible distance
        freq_magnitude = freq_magnitude / max_distance

        # Compute the weights as the square of the frequency magnitude
        # This will give higher weights to high frequencies
        weights = freq_magnitude**2

        return weights


class FocalLoss(nn.Module):
    def __init__(self) -> None:
        """Initialization of the focal loss."""
        super(FocalLoss, self).__init__()

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.8,
        gamma: float = 2,
    ) -> float:
        """Calculates the focal loss.

        Args:
            inputs: Predicted values.
            targets: Ground truth values.
            alpha: Weight for class imbalance. Defaults to 0.8.
            gamma: Weight for 'hard-to-classify' examples. Defaults to 2.

        Returns:
            Focal loss.
        """
        assert inputs.size() == targets.size()

        # Flatten label and prediction tensors
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        # First compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        """Initialization of the Dice loss.

        Args:
            smooth: Factor to ensure differentiability. Defaults to 1.0.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculates the Dice loss.

        Args:
            inputs: Predicted values of shape (b, c, v, x, y, z).
            targets: Ground truth values of shape (b, c, v, x, y, z).

        Returns:
            Dice loss.
        """
        assert inputs.size() == targets.size()

        # Calculate intersection and union
        intersection = reduce(inputs * targets, 'b c v x y z -> c', 'sum')
        union = reduce(inputs + targets, 'b c v x y z -> c', 'sum')

        # Calculate Dice score for each class
        dice_scores = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_scores.mean()


class WeightedMSELoss(nn.Module):
    def __init__(self, weight: torch.Tensor):
        """Initaliziation of the weighted MSE loss.

        Args:
            weight:  Weighting for each class, list of size n_classes.
        """
        super(WeightedMSELoss, self).__init__()
        self.weight = weight

    def forward(self, y_hat, y):
        """Calculates the weighted MSE loss.

        Args:
            y_hat: Prediction tensor.
            y: Ground truth tensor.

        Raises:
            ValueError: Raised if the weight list size is not equal to the
                number of classes.

        Returns:
            Loss.
        """
        if len(self.weight) != y_hat.shape[1]:
            raise ValueError(
                'Length of weight list must be the same as the number of classes.'
            )

        loss = 0
        for i, w in enumerate(self.weight):
            mse_class = F.mse_loss(y_hat[:, i, ...], y[:, i, ...], reduction='none')
            weighted_mse_class = w * mse_class
            loss += weighted_mse_class.sum()

        return loss / y_hat.numel()
