import torch
import torchio

from einops import rearrange
from typing import List

from kseg.data.custom_torchio import NDTransform

from typing import List


class DomainTransform(NDTransform):
    """Parent class for all domain transforms."""

    def __init__(self, exclude_label: bool) -> None:
        """Initalization of the Domain Transform class.

        Args:
            exclude_label: Whether to exclude the label for the transformation.
        """
        super().__init__()
        self.exclude_label = exclude_label

    def get_images(self, subject: torchio.Subject) -> List[torchio.Image]:
        images = subject.get_images(
            intensity_only=self.exclude_label,
            include=self.include,
            exclude=self.exclude,
        )
        return images


class KSpace(DomainTransform, torchio.FourierTransform):
    def __init__(self, exclude_label: bool) -> None:
        """Initialization of the KSpace transformation.

        Args:
            exclude_label: Whether to exlcude the label for the transformation.
        """
        super().__init__(exclude_label)

    def apply_transform(self, subject: torchio.Subject) -> torchio.Subject:
        """Applies the 2D real FFT on the given subject.

        Args:
            subject: Subject to be transformed. The data must be in RAS+
                orientation (c, x, y, z).

        Returns:
            Transformed subject.
        """
        for image in self.get_images(subject):
            transformed = torch.fft.rfft2(image.data)
            image.set_data(transformed)
        return subject

    @staticmethod
    def is_invertible() -> bool:
        """Shows whether the KSpace transformation is invertible.

        Returns:
            Whether the KSpace transformation is invertible.
        """
        return True

    def inverse(self):
        """Gives the inverse transformation for the KSpace transformation.

        Returns:
            Inverse KSpace transformation.
        """
        return InverseKSpace()


class InverseKSpace(DomainTransform, torchio.FourierTransform):
    def __init__(self, exclude_label: bool) -> None:
        """Initialization of the inverse KSpace Transformation.

        Args:
            exclude_label: Whether to exlcude the label for the transformation.
        """
        super().__init__(exclude_label)

    def apply_transform(self, subject: torchio.Subject) -> torchio.Subject:
        """Applies the inverse 2D real FFT on the given subject.

        Args:
            subject: Subject to be transformed. The data must be in RAS+
                orientation (c, x, y, z).

        Returns:
            2D real FFT transformed subject.
        """
        for image in self.get_images(subject):
            inverse_transformed = self._inverse_transform(image)
            image.set_data(inverse_transformed)
        return subject

    @staticmethod
    def _inverse_transform(image: torchio.Image) -> torchio.Image:
        """Inverse 2D real FFT transformation.

        Args:
            image: Image to be transformed. Image data must be in RAS+
                orientation (c, v, x, y, z).

        Returns:
            Transformed image.
        """
        return torch.fft.irfft2(image.data)

    @staticmethod
    def is_invertible() -> bool:
        """Shows whether the inverse KSpace transformation is invertible.

        Returns:
            Whether the invserse KSpace transformation is invertible.
        """
        return True

    def inverse(self):
        """Gives the inverse transformation for the inverse KSpace
            transformation.

        Returns:
            KSpace transformation.
        """
        return KSpace()


class Complex2Vec(NDTransform, torchio.SpatialTransform):
    def __init__(self) -> None:
        """Initialization for the complex to vector transformation."""
        super().__init__()

    def apply_transform(self, subject: torchio.Subject) -> torchio.Subject:
        """Transforms complex numbers to a vector on the given subject.

        Args:
            subject: Subject which contains complex values.

        Returns:
            Subject with at a new dimension for storing the real and imaginary
                part of the image. In case there are not complex numbers, this
                dimension has a lnegth of 1.
        """
        for image in self.get_images(subject):
            if 'complex' in str(image.data.dtype):
                image.set_data(torch.stack([image.data.real, image.data.imag], dim=1))
            else:
                image.set_data(torch.unsqueeze(image.data, dim=1))
        return subject

    @staticmethod
    def is_invertible() -> bool:
        """Shows whether the Complex2Vec transformation is invertible.

        Returns:
            Whether the Complex2Vec transformation is invertible.
        """
        return True

    def inverse(self):
        """Gives the inverse transformation for the Complex2Vec transformation.

        Returns:
            Vec2Complex transformation.
        """
        return Vec2Complex()


class Vec2Complex(NDTransform, torchio.SpatialTransform):
    def __init__(self) -> None:
        """Initialization for the vector to complex transformation."""
        super().__init__()

    def apply_transform(self, subject: torchio.Subject) -> torchio.Subject:
        """Transforms vectors to complex numbers on the given subject.

        Args:
            subject: Subject to be transformed. Image data must be in RAS+
                orientation (c, v, x, y, z) where the channel dim contains
                seg classes and vector dim the real and imaginary values of a
                complex number.

        Returns:
            Subject with complex numbers.
        """
        for image in self.get_images(subject):
            image.set_data(self._inverse_transform(image.data))
        return subject

    @staticmethod
    def _inverse_transform(data: torchio.Image) -> torchio.Image:
        """Inverse Vec2Complex transformation.

        Args:
            image: Image to be transformed. Image data must be in RAS+
                orientation (c, v, x, y, z).

        Returns:
            Transformed image.
        """
        # Reorder the axis since view_as_complex expects the last dimension
        # represents the real and imaginary components of complex numbers.
        data = rearrange(data, 'c v x y z -> c x y z v').contiguous()

        return torch.unsqueeze(torch.view_as_complex(data), dim=1)

    @staticmethod
    def is_invertible() -> bool:
        """Shows whether the Vec2Complex transformation is invertible.

        Returns:
            Whether the Vec2Complex transformation is invertible.
        """
        return True

    def inverse(self):
        """Gives the inverse transformation for the Vec2Complex transformation.

        Returns:
            Complex2Vec transformation.
        """
        return Complex2Vec()


class Unsqueeze(NDTransform, torchio.SpatialTransform):
    def __init__(self, position: int) -> None:
        """Initialization for the unsqueeze transformation.

        Args:
            position: Position at which the tensor should be unsqueezed.
        """
        super().__init__()
        self.position = position

    def apply_transform(self, subject: torchio.Subject) -> torchio.Subject:
        """Transformation that unsqueezes the tensor at the specified position.

        Args:
            subject: Subject containing data to be unsqueezed.

        Returns:
            Subject with one additional dimension at the specified position.
        """
        for image in self.get_images(subject):
            image.set_data(torch.unsqueeze(image.data, dim=self.position))
        return subject

    @staticmethod
    def is_invertible() -> bool:
        """Shows whether the unsqueeze transformation is invertible.

        Returns:
            Whether the unsqueeze transformation is invertible.
        """
        return True

    def inverse(self):
        """Gives the inverse transformation for the unsqueeze
            transformation.

        Returns:
            Squeeze transformation.
        """
        return Squeeze()


class Squeeze(NDTransform, torchio.SpatialTransform):
    def __init__(self, position: int) -> None:
        """Initialization for the squeeze transformation.

        Args:
            position: Position at which the tensor should be squeezed.
        """
        super().__init__()
        self.position = position

    def apply_transform(self, subject: torchio.Subject) -> torchio.Subject:
        """Transformation that squeezes the tensor.

        Args:
            subject: Subject containing data to be squeezed.

        Returns:
            Subject with one dimension less at the specified position.
        """
        for image in self.get_images(subject):
            image.set_data(torch.squeeze(image.data, dim=self.position))
        return subject

    @staticmethod
    def is_invertible() -> bool:
        """Shows whether the squeeze transformation is invertible.

        Returns:
            Whether the squeeze transformation is invertible.
        """
        return True

    def inverse(self):
        """Gives the inverse transformation for the squeeze
            transformation.

        Returns:
            Unsqueeze transformation.
        """
        return Unsqueeze()


class SimpleFreqSpace(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, img):
        return torch.fft.rfft2(img)


class SimpleComplex2Vec(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.view_as_real(x)
