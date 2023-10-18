import cv2
import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms.functional as vfunctional

from einops import rearrange
from pathlib import Path
from typing import Tuple, Dict, List, Any
from torch.optim import lr_scheduler, Optimizer
from torchmetrics import Specificity, Recall


from kseg.data.transforms import InverseKSpace, Vec2Complex
from kseg.model.modules import DiceScore
from kseg.model.modules import MLP, PerceiverIO, SkipMLP, ResMLP, Transformer


class LitModel(pl.LightningModule):
    def __init__(self, model_name: str, config: Dict[str, Any]) -> None:
        """Initialization of the custom Lightning Module.

        Args:
            model: Neural network model name.
            config: Neural network model and training config.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.lr = config['lr']
        self.criterion = config['criterion']
        self.optimizer_class = config['optimizer_class']
        self.step_size = config['step_size']
        self.scheduler_class = config['scheduler_class']
        self.input_domain = config['input_domain']
        self.label_domain = config['label_domain']
        self.dice_score = DiceScore()

        if self.model_name == 'MLP':
            self.net = MLP(
                config['input_shape'],
                config['output_shape'],
                config['hidden_factor'],
                config['depth'],
            )
        elif self.model_name == 'PerceiverIO':
            self.net = PerceiverIO(
                config['input_shape'],
                config['output_shape'],
                config['num_frequency_bands'],
                config['num_latents'],
                config['num_latent_channels'],
                config['num_cross_attention_heads'],
                config['num_cross_attention_layers'],
                config['num_self_attention_heads'],
                config['num_self_attention_layers_per_block'],
                config['num_self_attention_blocks'],
                config['dropout'],
            )
        elif self.model_name == 'SkipMLP':
            self.net = SkipMLP(
                config['input_shape'],
                config['output_shape'],
                config['hidden_factor'],
                config['depth_per_block'],
                config['depth'],
            )
        elif self.model_name == 'ResMLP':
            self.net = ResMLP(
                config['input_shape'],
                config['output_shape'],
                config['hidden_factor'],
                config['depth'],
                config['layerscale_init'],
            )
        elif self.model_name == 'Transformer':
            self.net = Transformer(
                config['input_shape'],
                config['output_shape'],
                config['hidden_factor'],
                config['depth'],
            )
        else:
            raise ValueError(f'Model {self.model} is not defined')

    @property
    def name(self) -> str:
        """Model name property.

        Returns:
            Model name.
        """
        return self.model_name

    def configure_optimizers(
        self,
    ) -> Tuple[Optimizer, lr_scheduler.LRScheduler]:
        """Configures the optimizer and scheduler based on the learning rate
            and step size.

        Returns:
            Configured optimizer and scheduler.
        """
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        scheduler = self.scheduler_class(optimizer, self.step_size)
        return [optimizer], [scheduler]

    def infer_batch(
        self, batch: Dict[str, dict]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate given batch through the Lightning Module.

        Args:
            batch: Batch containing the subjects.

        Returns:
            Model output and corresponding ground truth.
        """
        x, y = batch['input']['data'], batch['label']['data']
        y = y.float()
        y_hat = self.net(x)
        return y_hat, y

    def training_step(self, batch: Dict[str, dict], batch_idx: int) -> float:
        """Infer batch on training data, log metrics and retrieve loss.

        Args:
            batch: Batch containing the subjects.
            batch_idx: Number displaying index of this batch.

        Returns:
            Calculated loss.
        """
        y_hat, y = self.infer_batch(batch)

        # Calculate loss
        loss = self.criterion(y_hat, y)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, dict], batch_idx: int) -> None:
        """Infer batch on validation data, log metrics and retrieve loss.

        Args:
            batch: Batch containing the subjects.
            batch_idx: Number displaying index of this batch.

        Returns:
            Calculated loss.
        """
        y_hat, y = self.infer_batch(batch)
        x = batch['input']['data']

        # Calculate loss
        loss = self.criterion(y_hat, y)

        # Convert input and label to pixel domain to calculate dice score and
        # display qualitative results
        if self.label_domain == 'kspace':
            y_hat, y = self.evaluation_transform([y_hat, y])
        if self.input_domain == 'kspace':
            x = self.evaluation_transform([x])[0]
        # After iFFT there are imprecisions which result in numbers with
        # fractional parts. Use round() and convert to integer {0,1}.
        y = y.round().int()

        # Convert logits to one hot encoded segmentation mask
        y_hat = self.logits_to_mask(y_hat, y.shape[1])

        avg_dice, per_class_dice = self.dice_score(y_hat, y)
        (
            avg_recall,
            per_class_recall,
            avg_specificity,
            per_class_specificity,
        ) = self.calculate_recall_specificity(y_hat, y, y.shape[1])

        # Log metrics on epoch level
        self.log('step', float(self.current_epoch))  # Abuse steps as epochs
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_recall', avg_recall, on_epoch=True)
        self.log('val_specificity', avg_specificity, on_epoch=True)
        self.log('val_avg_dice', avg_dice, on_epoch=True)
        self.log_per_class_metrics('val_dice', per_class_dice)
        self.log_per_class_metrics('val_recall', per_class_recall)
        self.log_per_class_metrics('val_specificity', per_class_specificity)

        # Display qualitative results
        if batch_idx == 0:
            self.log_qualitativ_results(
                x, y, y_hat, self.logger, batch_idx, self.current_epoch
            )

    def test_step(self, batch: Dict[str, dict], batch_idx: int) -> None:
        """Infer batch on test data, log metrics and retrieve loss.

        Args:
            batch: Batch containing the subjects.
            batch_idx: Number displaying index of this batch.

        Returns:
            None.
        """
        y_hat, y = self.infer_batch(batch)
        x = batch['input']['data']

        # Calculate loss
        loss = self.criterion(y_hat, y)

        # Convert input and label to pixel domain to calculate dice score and
        # display qualitative results
        if self.label_domain == 'kspace':
            y_hat, y = self.evaluation_transform([y_hat, y])
        if self.input_domain == 'kspace':
            x = self.evaluation_transform([x])[0]
        # After iFFT there are imprecisions which result in numbers with
        # fractional parts. Use round() and convert to integer {0,1}.
        y = y.round().int()

        # Convert logits to one hot encoded segmentation mask
        y_hat = self.logits_to_mask(y_hat, y.shape[1])

        avg_dice, per_class_dice = self.dice_score(y_hat, y)
        (
            avg_recall,
            per_class_recall,
            avg_specificity,
            per_class_specificity,
        ) = self.calculate_recall_specificity(y_hat, y, y.shape[1])

        # Log metrics on epoch level
        self.log('step', float(self.current_epoch))  # Abuse steps as epochs
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_recall', avg_recall, on_epoch=True)
        self.log('test_specificity', avg_specificity, on_epoch=True)
        self.log('test_avg_dice', avg_dice, on_epoch=True)
        self.log_per_class_metrics('test_dice', per_class_dice)
        self.log_per_class_metrics('test_recall', per_class_recall)
        self.log_per_class_metrics('test_specificity', per_class_specificity)

        # Display qualitative results
        if batch_idx == 0:
            self.log_qualitativ_results(
                x, y, y_hat, self.logger, batch_idx, self.current_epoch, 'test'
            )

        # Save tensors as NIfTI files
        output_dir = Path(
            f'{self.logger.save_dir}/test_samples/batch_{batch_idx}/'
        )
        self.tensors_to_nifti(x, y, y_hat, output_dir)

    def logits_to_mask(self, logits: torch.Tensor, num_classes: int):
        """Converts logits to a one-hot encoded segmentation mask.

        Args:
            logits: Logits of shape (b, c, v, x, y, z).
            num_classes: Number of segmentation classes. Used for one-hot
                encoding.

        Returns:
            Segmentation mask of same shape as logits.
        """
        y_hat = F.softmax(logits, dim=1)
        y_hat = torch.argmax(y_hat, dim=1)
        # Argmax converted probs to ordinal encoded so we need one hot encoding
        return rearrange(
            F.one_hot(y_hat, num_classes=num_classes),
            'b v x y z c -> b c v x y z',
        )

    def evaluation_transform(
        self, variables: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Transform variables into pixel domain for evaluation.

        Args:
            variables: List of input or output variables.

        Returns:
            Variables in pixel domain.
        """
        vec2complex = Vec2Complex()
        inv_kspace = InverseKSpace(exclude_label=(self.label_domain == 'pixel'))
        transformed_variables = []
        for variable in variables:
            variable = [
                inv_kspace(vec2complex(b))
                for b in torch.unbind(variable, dim=0)
            ]
            variable = torch.stack(variable, dim=0)
            transformed_variables.append(variable)
        return transformed_variables

    def log_per_class_metrics(self, prefix: str, metrics: List):
        """Logs each class metric value separately.

        Args:
            prefix: Prefix for the name of the metric.
            metrics: List containing the values per class of the metric.


        Note:
            If metrics is a single scalar value, this function assumes a
            binary class segmentation was performed so there is no need to log
            the per class metrics.
        """
        try:
            metric_dict = {
                f"{prefix}_class_{i}": score.item()
                for i, score in enumerate(metrics)
            }
            self.log_dict(metric_dict, on_epoch=True)
        except TypeError:
            pass

    def log_qualitativ_results(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        logger: Any,
        batch_idx: int,
        epoch: int,
        stage: str = 'val',
    ) -> Any:
        # Extract middle slice along the z-axis
        middle_z = x.shape[-1] // 2
        x_slice = x[batch_idx, 0, 0, :, :, middle_z].cpu().numpy()
        y_slice = y[batch_idx, :, 0, :, :, middle_z].cpu().numpy()
        y_hat_slice = y_hat[batch_idx, :, 0, :, :, middle_z].cpu().numpy()

        # Convert one-hot to ordinal encoded ground truth and prediction
        y_slice = np.argmax(y_slice, axis=0)
        y_hat_slice = np.argmax(y_hat_slice, axis=0)

        # Define a colormap where each class ID maps to an RGB color
        color_map = {
            0: [0, 0, 0],  # Black for class 0 (background)
            1: [0, 255, 0],  # Green for class 1 (CSF / femoral cartilage)
            2: [255, 0, 0],  # Red for class 2 (cortical GM / tibial cartilage)
            3: [0, 0, 255],  # Blue for class 3 (WM / patellar cartilage)
            4: [255, 255, 0],  # Yellow for class 4 (GM / femur)
            5: [0, 255, 255],  # Cyan for class 5 (brain stem / tibia)
            6: [255, 0, 255],  # Magenta for class 6 (cerebellum / patella)
        }

        # Transform x
        x_slice = cv2.normalize(
            x_slice, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U
        )
        x_slice = cv2.cvtColor(x_slice, cv2.COLOR_GRAY2RGB)
        x_slice = np.transpose(torch.from_numpy(x_slice), axes=[2, 0, 1])
        x_slice = vfunctional.rotate(x_slice, 90)

        # Transform y
        output_image = np.zeros(
            (y_slice.shape[0], y_slice.shape[1], 3), dtype=np.uint8
        )
        for value, color in color_map.items():
            mask = y_slice == value
            output_image[mask] = color
        y_slice = np.transpose(torch.from_numpy(output_image), axes=[2, 0, 1])
        y_slice = vfunctional.rotate(y_slice, 90)
        y_slice = torch.from_numpy(
            cv2.addWeighted(x_slice.numpy(), 0.5, y_slice.numpy(), 0.5, 0)
        )

        # Transform y_hat
        output_image = np.zeros(
            (y_hat_slice.shape[0], y_hat_slice.shape[1], 3), dtype=np.uint8
        )
        for value, color in color_map.items():
            mask = y_hat_slice == value
            output_image[mask] = color
        y_hat_slice = np.transpose(
            torch.from_numpy(output_image), axes=[2, 0, 1]
        )
        y_hat_slice = vfunctional.rotate(y_hat_slice, 90)
        y_hat_slice = torch.from_numpy(
            cv2.addWeighted(x_slice.numpy(), 0.5, y_hat_slice.numpy(), 0.5, 0)
        )

        # Create grid and log to TensorBoard
        seg_grid = vutils.make_grid([x_slice, y_slice, y_hat_slice])
        logger.experiment.add_image(
            f'{stage}_sample_{batch_idx}', seg_grid, float(epoch)
        )

    def tensors_to_nifti(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        output_dir: Path,
    ) -> None:
        """Save a PyTorch tensor as a NIfTI file.

        Args:
            x: Input tensor to save.
            y: Ground truth tensor to save.
            y_hat: Prediction tensor to save.
            output_dir: The directory to save the tensors.

        Note:
            Saves only the first sample of each batch.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        y_hat = y_hat.detach().cpu().numpy()

        # Tranform shape from (b, c, v, x, y, z) to (b, x, y, z)
        x = x.squeeze(axis=(1, 2)).astype('float')
        y = np.argmax(y, axis=1).squeeze(axis=1).astype('float')
        y_hat = np.argmax(y_hat, axis=1).squeeze(axis=1).astype('float')

        nifti_img_x = nib.Nifti1Image(x[0], affine=np.eye(4))
        nib.save(nifti_img_x, output_dir / 'input.nii.gz')
        nifti_img_y = nib.Nifti1Image(y[0], affine=np.eye(4))
        nib.save(nifti_img_y, output_dir / 'gt.nii.gz')
        nifti_img_y_hat = nib.Nifti1Image(y_hat[0], affine=np.eye(4))
        nib.save(nifti_img_y_hat, output_dir / 'pred.nii.gz')

    def calculate_recall_specificity(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        num_classes: int,
    ) -> Tuple[float]:
        """Calculates the recall and specificity of a prediction given
            the target.

        Args:
            pred: Prediction.
            gt: Target / Ground truth.

        Returns:
            Average and per class recall and specificity scores.
        """
        # Convert one-hot to ordinal encoded tensor and flatten
        pred = torch.flatten(torch.argmax(pred, dim=1)).to('cpu')
        gt = torch.flatten(torch.argmax(gt, dim=1)).to('cpu')

        if num_classes == 2:
            num_classes -= 1

        # Calculate recall and specificity
        recall_metric = Recall(
            num_classes=num_classes,
            average='none',
            multiclass=(num_classes > 1),
        )
        specificity_metric = Specificity(
            num_classes=num_classes,
            average='none',
            multiclass=(num_classes > 1),
        )
        per_class_recall = recall_metric(pred, gt)
        avg_recall = per_class_recall.mean()
        per_class_specificity = specificity_metric(pred, gt)
        avg_specificity = per_class_specificity.mean()
        return (
            avg_recall,
            per_class_recall,
            avg_specificity,
            per_class_specificity,
        )
