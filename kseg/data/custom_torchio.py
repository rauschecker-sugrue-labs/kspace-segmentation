import copy
import warnings
from pathlib import Path
from typing import List, Optional

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torchio
from torchio.data.image import Image
from torchio.data.subject import Subject


class NDScalarImage(torchio.ScalarImage):
    """Extends torchio ScalarImage class to support n-dimensional tensors."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialization of the NDScalarImage."""
        super().__init__(*args, **kwargs)

    def _parse_tensor(
        self,
        tensor,
        none_ok: bool = True,
    ) -> Optional[torch.Tensor]:
        """Built on the torchio.data.Image._parse_tensor function but omits
        the check if the number of dimensions is 4.

        Args:
            tensor: N-dim Tensor.
            none_ok: Whether the tensor argument can be None. Defaults to True.

        Raises:
            RuntimeError: Raised if none_ok is set to False and Tensor is None.
            TypeError: Raised if tensor is not a PyTorch tensor or NumPy array.

        Returns:
            Parsed tensor.
        """
        if tensor is None:
            if none_ok:
                return None
            else:
                raise RuntimeError('Input tensor cannot be None')
        if isinstance(tensor, np.ndarray):
            tensor = torchio.check_uint_to_int(tensor)
            tensor = torch.as_tensor(tensor)
        elif not isinstance(tensor, torch.Tensor):
            message = (
                'Input tensor must be a PyTorch tensor or NumPy array,'
                f' but type "{type(tensor)}" was found'
            )
            raise TypeError(message)
        if tensor.dtype == torch.bool:
            tensor = tensor.to(torch.uint8)
        if self.check_nans and torch.isnan(tensor).any():
            warnings.warn('NaNs found in tensor', RuntimeWarning, stacklevel=2)
        return tensor

    @property
    def shape(self):
        """Built on the torchio.data.Inage.shape property but directly gets
        the shape from PyTorch tensor or NumPy array property instead of
        parsing each shape dimension into its own variable."""
        custom_reader = self.reader is not torchio.data.io.read_image
        multipath = self._is_multipath()
        if isinstance(self.path, Path):
            is_dir = self.path.is_dir()
        if self._loaded or custom_reader or multipath or is_dir:
            shape = self.data.shape
        else:
            assert isinstance(self.path, (str, Path))
            shape = torchio.data.io.read_shape(self.path)
        return shape


class NDLabelMap(torchio.LabelMap):
    """Extends torchio LabelMap class to support n-dimensional tensors."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialization of the NDLabelMap."""
        super().__init__(*args, **kwargs)

    def _parse_tensor(
        self,
        tensor,
        none_ok: bool = True,
    ) -> Optional[torch.Tensor]:
        """Built on the torchio.data.Image._parse_tensor function but omits
        the check if the number of dimensions is 4.

        Args:
            tensor: N-dim Tensor.
            none_ok: Whether the tensor argument can be None. Defaults to True.

        Raises:
            RuntimeError: Raised if none_ok is set to False and Tensor is None.
            TypeError: Raised if tensor is not a PyTorch tensor or NumPy array.

        Returns:
            Parsed tensor.
        """
        if tensor is None:
            if none_ok:
                return None
            else:
                raise RuntimeError('Input tensor cannot be None')
        if isinstance(tensor, np.ndarray):
            tensor = torchio.check_uint_to_int(tensor)
            tensor = torch.as_tensor(tensor)
        elif not isinstance(tensor, torch.Tensor):
            message = (
                'Input tensor must be a PyTorch tensor or NumPy array,'
                f' but type "{type(tensor)}" was found'
            )
            raise TypeError(message)
        if tensor.dtype == torch.bool:
            tensor = tensor.to(torch.uint8)
        if self.check_nans and torch.isnan(tensor).any():
            warnings.warn('NaNs found in tensor', RuntimeWarning, stacklevel=2)
        return tensor

    @property
    def shape(self):
        """Built on the torchio.data.Inage.shape property but directly gets
        the shape from PyTorch tensor or NumPy array property instead of
        parsing each shape dimension into its own variable."""
        custom_reader = self.reader is not torchio.data.io.read_image
        multipath = self._is_multipath()
        if isinstance(self.path, Path):
            is_dir = self.path.is_dir()
        if self._loaded or custom_reader or multipath or is_dir:
            shape = self.data.shape
        else:
            assert isinstance(self.path, (str, Path))
            shape = torchio.data.io.read_shape(self.path)
        return shape


class NDDataParser(torchio.transforms.data_parser.DataParser):
    """Extends torchio DataParser class to support n-dimensional tensors."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialization of the NDDataParser."""
        super().__init__(*args, **kwargs)

    def get_subject(self):
        """Built on the torchio.transforms.data_parser.DataParser but stores
        Nifti1 images as a NDScalarImage.

        Raises:
            RuntimeError: Raised if input is a dictionary but value for
                "include" is not set.
            ValueError: Raised if input type it not recognized.

        Returns:
            Subject.
        """
        if isinstance(self.data, nib.Nifti1Image):
            tensor = self.data.get_fdata(dtype=np.float32)
            if tensor.ndim == 3:
                tensor = tensor[np.newaxis]
            elif tensor.ndim == 5:
                tensor = tensor.transpose(3, 4, 0, 1, 2)
                # Assume a unique timepoint
                tensor = tensor[0]
            data = NDScalarImage(tensor=tensor, affine=self.data.affine)
            subject = self._get_subject_from_image(data)
            self.is_nib = True
        elif isinstance(self.data, (np.ndarray, torch.Tensor)):
            subject = self._parse_tensor(self.data)
            self.is_array = isinstance(self.data, np.ndarray)
            self.is_tensor = True
        elif isinstance(self.data, Image):
            subject = self._get_subject_from_image(self.data)
            self.is_image = True
        elif isinstance(self.data, Subject):
            subject = self.data
        elif isinstance(self.data, sitk.Image):
            subject = self._get_subject_from_sitk_image(self.data)
            self.is_sitk = True
        elif isinstance(self.data, dict):  # e.g. Eisen or MONAI dicts
            if self.keys is None:
                message = (
                    'If the input is a dictionary, a value for "include" must'
                    ' be specified when instantiating the transform. See the'
                    ' docs for Transform:'
                    ' https://torchio.readthedocs.io/transforms/transforms.html#torchio.transforms.Transform'  # noqa: B950
                )
                raise RuntimeError(message)
            subject = self._get_subject_from_dict(
                self.data,
                self.keys,
                self.label_keys,
            )
            self.is_dict = True
        else:
            raise ValueError(f'Input type not recognized: {type(self.data)}')
        assert isinstance(subject, Subject)
        return subject

    def _parse_tensor(self, data) -> Subject:
        """Built on the torchio.transforms.data_parser._parse_tensor function
        but omits the check if the number of dimensions of data is 4.

        Args:
            data: Tensor.

        Returns:
            Parsed Tensor.
        """
        return self._get_subject_from_tensor(data)

    def _get_subject_from_tensor(self, tensor) -> Subject:
        """Built on the torchio.transforms.data_parser._get_subject_from_tensor
        function but stores the tensor as a NDScalarImage instead of a
        ScalarImage.

        Args:
            tensor: Tensor, which contains the subject.

        Returns:
            Return of the _get_subject_from_image function using the tensor as
                argument.
        """
        image = NDScalarImage(tensor=tensor)
        return self._get_subject_from_image(image)


class NDTransform(torchio.Transform):
    """Extends torchio Transform class to support n-dimensional tensors."""

    def __init__(self, exclude_label: bool = False) -> None:
        """Initialization of the NDTransform class.

        Args:
            exclude_label: Whether to exlcude the label for the transformation.
                Defaults to False.

        Returns:
            None.
        """
        super().__init__()
        self.exclude_label = exclude_label

    def __call__(self, data):
        """Overrides the torchio.Transform __call__ method."""
        if torch.rand(1).item() > self.probability:
            return data

        # Some transforms such as Compose should not modify the input data
        if self.parse_input:
            data_parser = NDDataParser(
                data,
                keys=self.include,
                label_keys=self.label_keys,
            )
            subject = data_parser.get_subject()
        else:
            subject = data

        if self.keep is not None:
            images_to_keep = {}
            for name, new_name in self.keep.items():
                images_to_keep[new_name] = copy.copy(subject[name])
        if self.copy:
            subject = copy.copy(subject)
        with np.errstate(all='raise', under='ignore'):
            transformed = self.apply_transform(subject)
        if self.keep is not None:
            for name, image in images_to_keep.items():
                transformed.add_image(image, name)

        if self.parse_input:
            self.add_transform_to_subject_history(transformed)
            output = data_parser.get_output(transformed)
        else:
            output = transformed

        return output

    def get_images(self, subject: torchio.Subject) -> List[torchio.Image]:
        images = subject.get_images(
            intensity_only=self.exclude_label,
            include=self.include,
            exclude=self.exclude,
        )
        return images
