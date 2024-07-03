from typing import Generator, Optional

import nibabel as nib
import numpy as np
import torch
from torchio.constants import LOCATION
from torchio.data.subject import Subject
from torchio.typing import TypeSpatialShape, TypeTripletInt
from torchio.utils import to_tuple

from kseg.data.custom_torchio import NDSpatialTransform


class BoundsTransform(NDSpatialTransform):
    """Base class for transforms that change image bounds.

    Args:
        bounds_parameters: The meaning of this argument varies according to the
            child class.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """

    def __init__(self, bounds_parameters, **kwargs):
        super().__init__(**kwargs)
        self.bounds_parameters = bounds_parameters

    def is_invertible(self):
        return True


class Crop(BoundsTransform):
    r"""Crop an image.

    Args:
        cropping: Tuple
            :math:`(w_{ini}, w_{fin}, h_{ini}, h_{fin}, d_{ini}, d_{fin})`
            defining the number of values cropped from the edges of each axis.
            If the initial shape of the image is
            :math:`W \times H \times D`, the final shape will be
            :math:`(- w_{ini} + W - w_{fin}) \times (- h_{ini} + H - h_{fin})
            \times (- d_{ini} + D - d_{fin})`.
            If only three values :math:`(w, h, d)` are provided, then
            :math:`w_{ini} = w_{fin} = w`,
            :math:`h_{ini} = h_{fin} = h` and
            :math:`d_{ini} = d_{fin} = d`.
            If only one value :math:`n` is provided, then
            :math:`w_{ini} = w_{fin} = h_{ini} = h_{fin}
            = d_{ini} = d_{fin} = n`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    .. seealso:: If you want to pass the output shape instead, please use
        :class:`~torchio.transforms.CropOrPad` instead.
    """

    def __init__(self, cropping, **kwargs):
        super().__init__(cropping, **kwargs)
        self.cropping = cropping
        self.args_names = ['cropping']

    def apply_transform(self, sample) -> Subject:
        assert self.bounds_parameters is not None
        i0, i1 = self.cropping
        for image in self.get_images(sample):
            new_origin = nib.affines.apply_affine(image.affine, (i0, 0, 0))
            new_affine = image.affine.copy()
            new_affine[:3, 3] = new_origin
            image.set_data(image.data[:, :, i0:i1, :, :].clone())
            image.affine = new_affine
        return sample

    def inverse(self):
        from torchio.transforms.preprocessing.spatial.pad import Pad

        return Pad(self.cropping)


class PatchSampler:
    r"""Base class for TorchIO samplers.

    Args:
        patch_size: Tuple of integers :math:`(w, h, d)` to generate patches
            of size :math:`w \times h \times d`.
            If a single number :math:`n` is provided, :math:`w = h = d = n`.

    .. warning:: This is an abstract class that should only be instantiated
        using child classes such as :class:`~torchio.data.UniformSampler` and
        :class:`~torchio.data.WeightedSampler`.
    """

    def __init__(self, patch_size: TypeSpatialShape):
        patch_size_array = np.array(to_tuple(patch_size, length=3))
        for n in patch_size_array:
            if n < 1 or not isinstance(n, (int, np.integer)):
                message = (
                    'Patch dimensions must be positive integers,'
                    f' not {patch_size_array}'
                )
                raise ValueError(message)
        self.patch_size = patch_size_array.astype(np.uint16)

    def extract_patch(
        self,
        subject: Subject,
        index_ini: TypeTripletInt,
    ) -> Subject:

        cropped_subject = self.crop(subject, index_ini, self.patch_size)  # type: ignore[arg-type]  # noqa: B950
        return cropped_subject

    def crop(
        self,
        subject: Subject,
        index_ini: TypeTripletInt,
        patch_size: TypeTripletInt,
    ) -> Subject:
        transform = self._get_crop_transform(index_ini, patch_size)
        cropped_subject = transform(subject)
        patch_size_array = np.asarray(patch_size)
        location = [index_ini, 0, 0] + (
            patch_size_array + [index_ini, 0, 0]
        ).tolist()
        cropped_subject[LOCATION] = torch.tensor(location)
        cropped_subject.update_attributes()
        return cropped_subject

    @staticmethod
    def _get_crop_transform(
        index_ini: int,
        patch_size: TypeSpatialShape,
    ):
        patch_size_array = np.array(patch_size, dtype=np.uint16)
        index_fin = index_ini + patch_size_array[0]
        return Crop((index_ini, index_fin))  # type: ignore[arg-type]

    def __call__(
        self,
        subject: Subject,
        num_patches: Optional[int] = None,
    ) -> Generator[Subject, None, None]:
        kwargs = {} if num_patches is None else {'num_patches': num_patches}
        return self._generate_patches(subject, **kwargs)

    def _generate_patches(
        self,
        subject: Subject,
        num_patches: Optional[int] = None,
    ) -> Generator[Subject, None, None]:
        raise NotImplementedError


class RandomSampler(PatchSampler):
    r"""Base class for random samplers.

    Args:
        patch_size: Tuple of integers :math:`(w, h, d)` to generate patches
            of size :math:`w \times h \times d`.
            If a single number :math:`n` is provided, :math:`w = h = d = n`.
    """

    def get_probability_map(self, subject: Subject):
        raise NotImplementedError


class UniformSampler(RandomSampler):
    """Randomly extract patches from a volume with uniform probability.

    Args:
        patch_size: See :class:`~torchio.data.PatchSampler`.

    Note:
        For the patch_size only first element is considered as it represents
        the sagittal slice.
    """

    def get_probability_map(self, subject: Subject) -> torch.Tensor:
        return torch.ones(1, *subject.spatial_shape)

    def _generate_patches(
        self,
        subject: Subject,
        num_patches: Optional[int] = None,
    ) -> Generator[Subject, None, None]:
        valid_range = subject.spatial_shape[0] - self.patch_size[0]
        patches_left = num_patches if num_patches is not None else True
        while patches_left:
            index_ini = int(torch.randint(valid_range + 1, (1,)).item())
            yield self.extract_patch(subject, index_ini)
            if num_patches is not None:
                patches_left -= 1
