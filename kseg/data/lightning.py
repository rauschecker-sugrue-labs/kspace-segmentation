import glob
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torchio

from kseg.data.custom_torchio import NDLabelMap, NDScalarImage
from kseg.data.transforms import Complex2Vec, Compress, KSpace, Unsqueeze


class DataModuleBase(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        input_domain: str,
        label_domain: str,
        dataset_dir: str,
        num_classes: int,
        train_val_ratio: float,
        resampling_target_size: int,
        crop_size: Tuple,
        class_weights: Optional[List] = None,
    ) -> None:
        """Initialization of the Base Data Module.

        Args:
            batch_size: Batch size used for training.
            input_domain: Target domain of inputs ('kspace' or 'pixel').
            label_domain: Target domain of label ('kspace' or 'pixel').
            dataset_dir: Directory of the dataset.
            train_val_ratio: Ratio between train dataset and val+test dataset.
            resampling_target_size: Size to which the input shall be resampled.
            crop_size: Size to which the input shall be cropped.
            class_weights: Weights for each class which are used to calculate
                the loss. Defaults to None.
        """
        super().__init__()
        self.batch_size = batch_size
        self.input_domain = input_domain
        self.label_domain = label_domain
        self.dataset_dir = dataset_dir
        self.num_classes = num_classes
        self.train_val_ratio = train_val_ratio
        self.resampling_target_size = resampling_target_size
        self.crop_size = crop_size
        self.class_weights = class_weights
        self.subject_list = None

    @property
    def input_shape(self) -> Tuple:
        """Input shape property.

        Returns:
            Input shape.
        """
        if self.input_domain == 'kspace':
            return (
                1,
                2,
                self.crop_size[0],
                self.crop_size[1],
                self.crop_size[2] // 2,
            )
        return 1, 1, self.crop_size[0], self.crop_size[1], self.crop_size[2]

    @property
    def label_shape(self) -> Tuple:
        """Label shape property.

        Returns:
            Label shape.
        """
        if self.label_domain == 'kspace':
            return (
                self.num_classes,
                2,
                self.crop_size[0],
                self.crop_size[1],
                self.crop_size[2] // 2,
            )
        return (
            self.num_classes,
            1,
            self.crop_size[0],
            self.crop_size[1],
            self.crop_size[2],
        )

    def _flatten(self, nested_list: List[List]) -> List:
        """Flattens a single-nested list.

        Args:
            nested_list: Single-nested list.

        Returns:
            Flattened list or unmodified list if the given list was not nested.
        """
        if all(not isinstance(item, (list, tuple)) for item in nested_list):
            return nested_list
        return [item for sublist in nested_list for item in sublist]

    def get_preprocessing_transform(self) -> torchio.Compose:
        """Composes the transformations for the preprocessing step.

        Returns:
            Transformations for the preprocessing step.
        """
        preprocess = torchio.Compose(
            [
                torchio.ToCanonical(),
                torchio.Resample('input'),
                torchio.Resample(self.resampling_target_size),
                torchio.CropOrPad(self.crop_size),
                torchio.transforms.ZNormalization(),
            ]
        )
        return preprocess

    def get_augmentation_transform(self) -> torchio.Compose:
        """Composes the transformations for the augmentation step.

        Returns:
            Transformations for the augmentation step.
        """
        augment = torchio.Compose(
            [
                torchio.RandomAffine(),
                torchio.RandomGamma(p=0.1),
                torchio.RandomNoise(p=0.1),
                torchio.RandomMotion(p=0.1),
                torchio.RandomBiasField(p=0.1),
            ]
        )
        return augment

    def setup(self, stage: str = None) -> None:
        """Creates processed datasets for training, validation and testing.

        Args:
            stage: Current stage of data propagation. Defaults to None.
        """
        train_val_test_ratio = [
            self.train_val_ratio,
            (1 - self.train_val_ratio) / 2,
            (1 - self.train_val_ratio) / 2,
        ]
        (
            train_subjects,
            val_subjects,
            test_subjects,
        ) = torch.utils.data.random_split(
            self.subject_list,
            train_val_test_ratio,
            torch.Generator().manual_seed(42),
        )

        preprocess = self.get_preprocessing_transform()
        augment = self.get_augmentation_transform()

        # If the input domain is not in k-space, neither is the label domain
        if self.input_domain == 'kspace':
            domain_transform = torchio.Compose(
                [
                    KSpace(exclude_label=(self.label_domain == 'pixel')),
                    Complex2Vec(),
                    Compress(exclude_label=(self.label_domain == 'pixel')),
                ]
            )
            self.train_transform = torchio.Compose(
                [
                    preprocess,
                    augment,
                    torchio.OneHot(num_classes=self.num_classes),
                    domain_transform,
                ]
            )
            self.val_transform = torchio.Compose(
                [
                    preprocess,
                    torchio.OneHot(num_classes=self.num_classes),
                    domain_transform,
                ]
            )

            self.test_transform = self.val_transform
        else:
            self.train_transform = torchio.Compose(
                [
                    preprocess,
                    augment,
                    torchio.OneHot(num_classes=self.num_classes),
                    Unsqueeze(position=1),
                ]
            )
            self.val_transform = torchio.Compose(
                [
                    preprocess,
                    torchio.OneHot(num_classes=self.num_classes),
                    Unsqueeze(position=1),
                ]
            )
            self.test_transform = self.val_transform

        # Flatten all lists in case they were stored subject-wise
        self.train_subjects = self._flatten(train_subjects)
        self.val_subjects = self._flatten(val_subjects)
        self.test_subjects = self._flatten(test_subjects)

        self.train_set = torchio.SubjectsDataset(
            self.train_subjects, transform=self.train_transform
        )
        self.val_set = torchio.SubjectsDataset(
            self.val_subjects, transform=self.val_transform
        )
        self.test_set = torchio.SubjectsDataset(
            self.test_subjects, transform=self.test_transform
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Creates Dataloader for training phase.

        Returns:
            Dataloader for training phase.
        """
        return torch.utils.data.DataLoader(
            self.train_set, self.batch_size, num_workers=32
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Creates Dataloader for validation phase.

        Returns:
            Dataloader for validation phase.
        """
        return torch.utils.data.DataLoader(
            self.val_set, self.batch_size, num_workers=10
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Creates Dataloader for testing phase.

        Returns:
            Dataloader for testing phase.
        """
        return torch.utils.data.DataLoader(
            self.test_set, self.batch_size, num_workers=10
        )

    def save_preprocessed_data(self, path: Union[str, Path]) -> None:
        """Save preprocessed inputs and labels of data module.

        Args:
            path: Path to store the preprocessed inputs and labels for train,
                validation and test stage.
        """
        base_path = Path(path)

        for dir_name, dataset in [
            ('train', self.train_subjects),
            ('val', self.val_subjects),
            ('test', self.test_subjects),
        ]:
            dir_path = base_path / dir_name
            dir_path.mkdir(exist_ok=True)

            for idx, subject in enumerate(dataset):
                transformed_subject = (
                    self.get_preprocessing_transform().apply_transform(subject)
                )

                transformed_subject['input'].save(dir_path / f'input_{idx}.nii.gz')
                transformed_subject['label'].save(dir_path / f'label_{idx}.nii.gz')


class UCSF51LesionDataModule(DataModuleBase):
    def __init__(
        self,
        batch_size: int,
        input_domain: str,
        label_domain: str,
        datasets_dir: str,
        train_val_ratio: float = 0.8,
        resampling_target_size: int = 3,
        crop_size: Tuple = (64, 64, 64),
    ) -> None:
        """Initialization for the UCSF51 lesion Data Module.

        Args:
            batch_size: Batch size used for training.
            input_domain: Target domain of inputs ('kspace' or 'pixel').
            label_domain: Target domain of label ('kspace' or 'pixel').
            datasets_dir: Path to the datasets directory.
            train_val_ratio: Ratio between train dataset and val+test dataset.
                Defaults to 0.8.
            resampling_target_size: Size to which the input shall be resampled.
                Defaults to 3.
            crop_size: Size to which the input shall be cropped.
                Defaults to (64, 64, 64).
        """
        super().__init__(
            batch_size=batch_size,
            input_domain=input_domain,
            label_domain=label_domain,
            dataset_dir=os.path.join(os.getcwd(), datasets_dir, 'UCSF51/'),
            num_classes=2,
            train_val_ratio=train_val_ratio,
            resampling_target_size=resampling_target_size,
            crop_size=crop_size,
        )

    @property
    def name(self) -> str:
        """Data module name property.

        Returns:
            Data module name.
        """
        return 'UCSF51Lesion'

    def prepare_data(self) -> None:
        """Creates the subject list based on the UCSF51 dataset dir."""
        self.subject_list = []
        image_files = glob.glob(os.path.join(self.dataset_dir, '*_seg.nii.gz'))
        for image in image_files:
            accession = os.path.basename(image).split('_')[0]
            image_path = os.path.join(self.dataset_dir, f'{accession}.nii.gz')
            seg_path = os.path.join(self.dataset_dir, f'{accession}_seg.nii.gz')

            subject = torchio.Subject(
                input=NDScalarImage(image_path),
                label=NDLabelMap(seg_path),
            )
            self.subject_list.append(subject)


class CNSLymphomaSSDataModule(DataModuleBase):
    def __init__(
        self,
        batch_size: int,
        input_domain: str,
        label_domain: str,
        datasets_dir: str,
        train_val_ratio: float = 0.8,
        resampling_target_size: int = 3,
        crop_size: Tuple = (64, 64, 64),
    ) -> None:
        """Initialization for the CNS lymphoma skull stripping Data Module.

        Args:
            batch_size: Batch size used for training.
            input_domain: Target domain of inputs ('kspace' or 'pixel').
            label_domain: Target domain of labels ('kspace' or 'pixel').
            datasets_dir: Path to the datasets directory.
            train_val_ratio: Ratio between train dataset and val+test dataset.
                Defaults to 0.8.
            resampling_target_size: Size to which the input shall be resampled.
                Defaults to 3.
            crop_size: Size to which the input shall be cropped.
                Defaults to (64, 64, 64).
        """
        super().__init__(
            batch_size=batch_size,
            input_domain=input_domain,
            label_domain=label_domain,
            dataset_dir=os.path.join(os.getcwd(), datasets_dir, 'CNS_Lymphoma/'),
            num_classes=2,
            train_val_ratio=train_val_ratio,
            resampling_target_size=resampling_target_size,
            crop_size=crop_size,
        )

    @property
    def name(self):
        return 'CNSLymphomaSS'

    def prepare_data(self) -> None:
        """Creates the subject list based on the SkullStripping dataset dir."""
        self.subject_list = []
        image_files = glob.glob(str(Path(self.dataset_dir) / 'imagesTr' / '*.nii.gz'))

        current_identifier = None
        current_subjects = []

        for image in sorted(image_files):
            accession = Path(image).name.split('_')[0]
            identifier = re.split(r'(\D)', accession, maxsplit=1)[0]

            # If identifier is different from the current one, start a new list
            if identifier != current_identifier:
                if current_subjects:
                    self.subject_list.append(current_subjects)
                current_identifier = identifier
                current_subjects = []

            image_path = (
                Path(self.dataset_dir) / 'imagesTr' / f'{accession}_0000.nii.gz'
            )
            seg_path = Path(self.dataset_dir) / 'labelsTr' / f'{accession}.nii.gz'

            subject = torchio.Subject(
                input=NDScalarImage(image_path),
                label=NDLabelMap(seg_path),
            )
            current_subjects.append(subject)

        if current_subjects:
            self.subject_list.append(current_subjects)


class CNSLymphomaTissueDataModule(DataModuleBase):
    def __init__(
        self,
        batch_size: int,
        input_domain: str,
        label_domain: str,
        datasets_dir: str,
        train_val_ratio: float = 0.8,
        resampling_target_size: int = 3,
        crop_size: Tuple = (64, 64, 64),
    ) -> None:
        """Initialization for the CNS lymphoma tissue Data Module.
        Args:
            batch_size: Batch size used for training.
            input_domain: Target domain of inputs ('kspace' or 'pixel').
            label_domain: Target domain of labels ('kspace' or 'pixel').
            datasets_dir: Path to the datasets directory.
            train_val_ratio: Ratio between train dataset and val+test dataset.
                Defaults to 0.8.
            resampling_target_size: Size to which the input shall be resampled.
                Defaults to 3.
            crop_size: Size to which the input shall be cropped.
                Defaults to (64, 64, 64).
        """
        super().__init__(
            batch_size=batch_size,
            input_domain=input_domain,
            label_domain=label_domain,
            dataset_dir=os.path.join(os.getcwd(), datasets_dir, 'CNS_Lymphoma/'),
            num_classes=7,
            train_val_ratio=train_val_ratio,
            resampling_target_size=resampling_target_size,
            crop_size=crop_size,
            class_weights=[0.2, 2.37, 1.54, 1.55, 18.43, 30.97, 4.43],
        )

    @property
    def name(self):
        return 'CNSLymphomaTissue'

    def prepare_data(self) -> None:
        """Creates the subject list based on the Tissue dataset dir."""
        self.subject_list = []
        image_files = glob.glob(os.path.join(self.dataset_dir, 'imagesTr', '*.nii.gz'))
        for image in image_files:
            accession = os.path.basename(image).split('_')[0]
            image_path = os.path.join(
                self.dataset_dir, 'imagesTr', f'{accession}_0000.nii.gz'
            )
            seg_path = os.path.join(self.dataset_dir, 'labelsTr', f'{accession}.nii.gz')
            subject = torchio.Subject(
                input=NDScalarImage(image_path),
                label=NDLabelMap(seg_path),
            )
            self.subject_list.append(subject)


class KneeDataModule(DataModuleBase):
    def __init__(
        self,
        batch_size: int,
        input_domain: str,
        label_domain: str,
        datasets_dir: str,
        train_val_ratio: float = 0.8,
        resampling_target_size: int = 3,
        crop_size: Tuple = (64, 64, 64),
    ) -> None:
        """Initialization for the Knee Data Module.
        Args:
            batch_size: Batch size used for training.
            input_domain: Target domain of inputs ('kspace' or 'pixel').
            label_domain: Target domain of labels ('kspace' or 'pixel').
            datasets_dir: Path to the datasets directory.
            train_val_ratio: Ratio between train dataset and val+test dataset.
                Defaults to 0.8.
            resampling_target_size: Size to which the input shall be resampled.
                Defaults to 3.
            crop_size: Size to which the input shall be cropped.
                Defaults to (64, 64, 64).
        """
        super().__init__(
            batch_size=batch_size,
            input_domain=input_domain,
            label_domain=label_domain,
            dataset_dir=os.path.join(os.getcwd(), datasets_dir, 'Knee/'),
            num_classes=7,
            train_val_ratio=train_val_ratio,
            resampling_target_size=resampling_target_size,
            crop_size=crop_size,
        )

    @property
    def name(self):
        return 'Knee'

    def prepare_data(self):
        self.subject_list = []
        for image in glob.glob(os.path.join(self.dataset_dir, '*_seg.nii.gz')):
            accession = os.path.basename(image).split('_')[0]
            image_path = os.path.join(self.dataset_dir, f'{accession}.nii.gz')
            seg_path = os.path.join(self.dataset_dir, f'{accession}_seg.nii.gz')

            subject = torchio.Subject(
                input=NDScalarImage(image_path),
                label=NDLabelMap(seg_path),
            )
            self.subject_list.append(subject)

    def get_preprocessing_transform(self) -> torchio.Compose:
        """Composes the transformations for the preprocessing step.
        Returns:
            Transformations for the preprocessing step.
        """
        preprocess = torchio.Compose(
            [
                torchio.ToCanonical(),
                torchio.Resample('input'),
                torchio.Resample(self.resampling_target_size),
                torchio.Resize(64),
                torchio.transforms.ZNormalization(),
            ]
        )
        return preprocess


class UPennGBMSSDataModule(DataModuleBase):
    def __init__(
        self,
        batch_size: int,
        input_domain: str,
        label_domain: str,
        datasets_dir: str,
        train_val_ratio: float = 0.8,
        resampling_target_size: int = 3,
        crop_size: Tuple = (64, 64, 64),
    ) -> None:
        """Initialization for the UPenn GBM skull stripping Data Module.

        Args:
            batch_size: Batch size used for training.
            input_domain: Target domain of inputs ('kspace' or 'pixel').
            label_domain: Target domain of label ('kspace' or 'pixel').
            datasets_dir: Path to the datasets directory.
            train_val_ratio: Ratio between train dataset and val+test dataset.
                Defaults to 0.8.
            resampling_target_size: Size to which the input shall be resampled.
                Defaults to 3.
            crop_size: Size to which the input shall be cropped.
                Defaults to (64, 64, 64).
        """
        super().__init__(
            batch_size=batch_size,
            input_domain=input_domain,
            label_domain=label_domain,
            dataset_dir=os.path.join(os.getcwd(), datasets_dir, 'UPENN_GBM/'),
            num_classes=2,
            train_val_ratio=train_val_ratio,
            resampling_target_size=resampling_target_size,
            crop_size=crop_size,
        )

    @property
    def name(self) -> str:
        """Data module name property.

        Returns:
            Data module name.
        """
        return 'UPennGBMSS'

    def prepare_data(self) -> None:
        """Creates the subject list based on the UPennGBM dataset dir."""
        self.subject_list = []
        image_dir = os.path.join(self.dataset_dir, 'images_structural_unstripped')
        stripped_dir = os.path.join(self.dataset_dir, 'images_structural')

        # Go through each subject's folder
        for subject_dir in glob.glob(os.path.join(image_dir, '*')):
            # Skip the directories ending with 21
            if subject_dir.endswith('21'):
                continue

            # Find the unstripped T1 image path
            t1_unstripped_path = glob.glob(
                os.path.join(subject_dir, '*_T1_unstripped.nii.gz')
            )[0]
            base_name = os.path.basename(t1_unstripped_path).replace(
                '_T1_unstripped.nii.gz', ''
            )

            # Determine the corresponding stripped T1 image path
            t1_stripped_path = os.path.join(
                stripped_dir, base_name, f'{base_name}_T1.nii.gz'
            )

            # Create and append subject object to list
            subject = torchio.Subject(
                input=NDScalarImage(t1_unstripped_path),
                label=NDLabelMap(t1_stripped_path),
            )
            self.subject_list.append(subject)

    def get_preprocessing_transform(self) -> torchio.Compose:
        """Composes the transformations for the preprocessing step.

        Returns:
            Transformations for the preprocessing step.
        """
        preprocess = torchio.Compose(
            [
                # Convert stripped images to binary segmentation maps
                torchio.Lambda(
                    lambda x: (x != 0).float(), types_to_apply=[torchio.LABEL]
                ),
                torchio.ToCanonical(),
                torchio.Resample('input'),
                torchio.Resample(self.resampling_target_size),
                torchio.CropOrPad(self.crop_size),
                torchio.transforms.ZNormalization(),
            ]
        )
        return preprocess


class UPennGBMTumorDataModule(DataModuleBase):
    def __init__(
        self,
        batch_size: int,
        input_domain: str,
        label_domain: str,
        datasets_dir: str,
        train_val_ratio: float = 0.8,
        resampling_target_size: int = 3,
        crop_size: Tuple = (64, 64, 64),
    ) -> None:
        """Initialization for the UPenn GBM tumor Data Module.

        Args:
            batch_size: Batch size used for training.
            input_domain: Target domain of inputs ('kspace' or 'pixel').
            label_domain: Target domain of label ('kspace' or 'pixel').
            datasets_dir: Path to the datasets directory.
            train_val_ratio: Ratio between train dataset and val+test dataset.
                Defaults to 0.8.
            resampling_target_size: Size to which the input shall be resampled.
                Defaults to 3.
            crop_size: Size to which the input shall be cropped.
                Defaults to (64, 64, 64).
        """
        super().__init__(
            batch_size=batch_size,
            input_domain=input_domain,
            label_domain=label_domain,
            dataset_dir=os.path.join(os.getcwd(), datasets_dir, 'UPENN_GBM/'),
            num_classes=4,
            train_val_ratio=train_val_ratio,
            resampling_target_size=resampling_target_size,
            crop_size=crop_size,
            class_weights=[0.25, 196.26, 42.66, 114.3],
        )

    @property
    def name(self) -> str:
        """Data module name property.

        Returns:
            Data module name.
        """
        return 'UPennGBMTumor'

    def prepare_data(self) -> None:
        """Creates the subject list based on the UPennGBM dataset dir."""
        self.subject_list = []
        image_paths = []
        segmentation_paths = []
        image_dir = os.path.join(self.dataset_dir, 'images_structural')
        manual_segm_dir = os.path.join(self.dataset_dir, 'images_segm')
        auto_segm_dir = os.path.join(self.dataset_dir, 'automated_segm')

        # Go through each subject's folder
        for subject_dir in glob.glob(os.path.join(image_dir, '*')):
            # Skip the directories ending with 21
            if subject_dir.endswith('21'):
                continue

            # Look for the T1 image
            t1_image_path = glob.glob(os.path.join(subject_dir, '*_T1.nii.gz'))[
                0
            ]  # Taking the first match
            image_paths.append(t1_image_path)
            base_name = os.path.basename(t1_image_path).split('_T1.nii.gz')[0]

            # Check for manually segmented labels
            manual_segm = os.path.join(manual_segm_dir, f'{base_name}_segm.nii.gz')
            if os.path.exists(manual_segm):
                segmentation_paths.append(manual_segm)
            else:
                # If no manually segmented label, check for automatically
                # segmented label
                auto_segm = os.path.join(
                    auto_segm_dir, f'{base_name}_automated_approx_segm.nii.gz'
                )
                segmentation_paths.append(auto_segm)

        # Create subject objects
        for image_path, seg_path in zip(image_paths, segmentation_paths):
            subject = torchio.Subject(
                input=NDScalarImage(image_path),
                label=NDLabelMap(seg_path),
            )
            self.subject_list.append(subject)

    def get_preprocessing_transform(self) -> torchio.Compose:
        """Composes the transformations for the preprocessing step.

        Returns:
            Transformations for the preprocessing step.
        """
        preprocess = torchio.Compose(
            [
                # Change segmentation class 4 to 3 since 3 is never used.
                torchio.Lambda(
                    lambda x: torch.where(x == 4, torch.tensor(3), x),
                    types_to_apply=[torchio.LABEL],
                ),
                torchio.ToCanonical(),
                torchio.Resample('input'),
                torchio.Resample(self.resampling_target_size),
                torchio.CropOrPad(self.crop_size),
                torchio.transforms.ZNormalization(),
            ]
        )
        return preprocess


class OasisTissueDataModule(DataModuleBase):
    def __init__(
        self,
        batch_size: int,
        input_domain: str,
        label_domain: str,
        datasets_dir: str,
        train_val_ratio: float = 0.8,
        resampling_target_size: int = 3,
        crop_size: Tuple = (64, 64, 64),
    ) -> None:
        """Initialization for the Oasis Tissue Data Module.

        Args:
            batch_size: Batch size used for training.
            input_domain: Target domain of inputs ('kspace' or 'pixel').
            label_domain: Target domain of label ('kspace' or 'pixel').
            datasets_dir: Path to the datasets directory.
            train_val_ratio: Ratio between train dataset and val+test dataset.
                Defaults to 0.8.
            resampling_target_size: Size to which the input shall be resampled.
                Defaults to 3.
            crop_size: Size to which the input shall be cropped.
                Defaults to (64, 64, 64).
        """
        super().__init__(
            batch_size=batch_size,
            input_domain=input_domain,
            label_domain=label_domain,
            dataset_dir=os.path.join(os.getcwd(), datasets_dir, 'OASIS/'),
            num_classes=7,
            train_val_ratio=train_val_ratio,
            resampling_target_size=resampling_target_size,
            crop_size=crop_size,
            class_weights=[0.15, 74.13, 5.27, 6.0, 43.65, 121.28, 19.23],
        )

    @property
    def name(self) -> str:
        """Data module name property.

        Returns:
            Data module name.
        """
        return 'OasisTissue'

    def prepare_data(self) -> None:
        """Creates the subject list based on the Oasis dataset dir."""
        self.subject_list = []

        # Iterate over each directory inside the root directory
        for subject_dir in os.listdir(self.dataset_dir):
            if subject_dir.startswith('OAS1_') and os.path.isdir(
                os.path.join(self.dataset_dir, subject_dir)
            ):
                # Build the mri directory path
                mri_dir_path = os.path.join(self.dataset_dir, subject_dir, 'mri')

                # Check if both files exist in the mri directory
                aseg_path = os.path.join(mri_dir_path, 'aseg.nii.gz')
                brain_mask_path = os.path.join(mri_dir_path, 'brainmask.nii.gz')

                if os.path.exists(aseg_path) and os.path.exists(brain_mask_path):
                    subject = torchio.Subject(
                        input=NDScalarImage(brain_mask_path),
                        label=NDLabelMap(aseg_path),
                    )
                    self.subject_list.append(subject)

    def get_preprocessing_transform(self) -> torchio.Compose:
        """Composes the transformations for the preprocessing step.

        Returns:
            Transformations for the preprocessing step.
        """

        def custom_mapping(x):
            # Map FreeSurfer label values to kseg tissue label values
            mapping = {
                0: 0,
                2: 3,
                3: 2,
                4: 1,
                5: 1,
                7: 6,
                8: 6,
                10: 4,
                11: 4,
                12: 4,
                13: 4,
                14: 1,
                15: 1,
                16: 5,
                17: 4,
                18: 4,
                24: 1,
                26: 4,
                28: 4,
                30: 1,
                41: 3,
                42: 2,
                43: 1,
                44: 1,
                46: 6,
                47: 6,
                49: 4,
                50: 4,
                51: 4,
                52: 4,
                53: 4,
                54: 4,
                58: 4,
                60: 4,
                62: 1,
                72: 1,
                78: 3,
                79: 3,
                81: 4,
                82: 4,
                85: 5,
            }
            # This is used to prevent replacing replaced values
            replaced_mask = torch.zeros_like(x).bool()

            for original, new in mapping.items():
                mask = (x == original) & (~replaced_mask)
                x[mask] = new
                replaced_mask[mask] = True
            return x

        preprocess = torchio.Compose(
            [
                # Map FreeSurfer label classes to custom label classes
                torchio.Lambda(custom_mapping, types_to_apply=[torchio.LABEL]),
                torchio.ToCanonical(),
                torchio.Resample('input'),
                torchio.Resample(self.resampling_target_size),
                torchio.CropOrPad(self.crop_size),
                torchio.transforms.ZNormalization(),
            ]
        )
        return preprocess
