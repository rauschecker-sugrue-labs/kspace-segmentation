import torch
import torchio

from unittest import TestCase

from kseg.data.custom_torchio import NDScalarImage
from kseg.data.transforms import (
    KSpace,
    InverseKSpace,
    Complex2Vec,
    Compress,
    Decompress,
    Vec2Complex,
)


class TestKSpace(TestCase):
    def test_apply_transform(self):
        transform = KSpace(exclude_label=True)

        xs = torch.tensor([-1.0, -0.5, 0.5, 1])
        ys = torch.tensor([-1.0, -0.5, 0.5, 1])
        x, y = torch.meshgrid(xs, ys, indexing='xy')
        image = NDScalarImage(tensor=torch.stack([x + y, x, y]).unsqueeze(0))

        subject = torchio.Subject(one_image=image)
        transform.apply_transform(subject)

        self.assertEqual([1, 3, 4, 3], list(image.data.shape))
        torch.testing.assert_close(
            torch.tensor(
                [
                    [
                        [
                            [0.0 + 0.0j, -6.0 + 6.0j, -4.0 + 0.0j],
                            [-6.0 + 6.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                            [-4.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                            [-6.0 - 6.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        ],
                        [
                            [0.0 + 0.0j, -6.0 + 6.0j, -4.0 + 0.0j],
                            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        ],
                        [
                            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                            [-6.0 + 6.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                            [-4.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                            [-6.0 - 6.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        ],
                    ]
                ]
            ),
            image.data,
        )

    def test_is_invertible(self):
        self.assertTrue(KSpace(exclude_label=True).is_invertible())

    def test_inverse(self):
        self.assertEqual(type(KSpace(exclude_label=True).inverse()), InverseKSpace)


class TestInverseKSpace(TestCase):
    def test_apply_transform(self):
        transform = KSpace(exclude_label=True)
        inverse_transform = InverseKSpace(exclude_label=True)

        xs = torch.tensor([-1.0, -0.5, 0.5, 1])
        ys = torch.tensor([-1.0, -0.5, 0.5, 1])
        x, y = torch.meshgrid(xs, ys, indexing='xy')
        data = torch.stack([x + y, x, y]).unsqueeze(0)

        image = NDScalarImage(tensor=data)
        subject = torchio.Subject(one_image=image)
        transform.apply_transform(subject)
        inverse_transform.apply_transform(subject)

        torch.testing.assert_close(subject.one_image.data, data)

    def test_is_invertible(self):
        self.assertTrue(InverseKSpace(exclude_label=True).is_invertible())

    def test_inverse(self):
        self.assertEqual(type(InverseKSpace(exclude_label=True).inverse()), KSpace)


class TestComplex2Vec(TestCase):
    def test_apply_transform(self):
        transform = Complex2Vec()
        data = torch.tensor(
            [
                [
                    [
                        [1.0 + 0.0j, 1.0 - 0.0j],
                        [1.0 + 1.0j, 3.0 + 1.0j],
                        [1.0 - 1.0j, 3.0 - 1.0j],
                    ],
                    [
                        [2.0 - 0.0j, 1.0 + 0.0j],
                        [3.0 + 1.0j, -1.0 - 2.0j],
                        [2.0 - 1.0j, -1.0 + 2.0j],
                    ],
                ]
            ]
        )
        image = NDScalarImage(tensor=data)
        subject = torchio.Subject(one_image=image)
        transform.apply_transform(subject)
        torch.testing.assert_close(
            image.data[:, 0, :, :, :],
            torch.tensor([[[[1.0], [1.0], [1.0]], [[2.0], [3.0], [1.0]]]]),
        )
        torch.testing.assert_close(
            image.data[:, 1, :, :, :],
            torch.tensor([[[[1.0], [3.0], [1.0]], [[1.0], [-1.0], [-2.0]]]]),
        )

    def test_is_invertible(self):
        self.assertTrue(Complex2Vec().is_invertible())

    def test_inverse(self):
        self.assertEqual(type(Complex2Vec().inverse()), Vec2Complex)


class TestVec2Complex(TestCase):
    def test_apply_transform(self):
        transform = Complex2Vec()
        inverse_transform = Vec2Complex()
        data = torch.tensor(
            [
                [
                    [
                        [1.0 + 0.0j, 1.0 - 0.0j],
                        [1.0 + 1.0j, 3.0 + 1.0j],
                        [1.0 - 1.0j, 3.0 - 1.0j],
                    ],
                    [
                        [2.0 - 0.0j, 1.0 + 0.0j],
                        [3.0 + 1.0j, -1.0 - 2.0j],
                        [2.0 - 1.0j, -1.0 + 2.0j],
                    ],
                ]
            ]
        )
        image = NDScalarImage(tensor=data)
        subject = torchio.Subject(one_image=image)
        transform.apply_transform(subject)
        inverse_transform.apply_transform(subject)
        # Why does the inverse transform unsqueeze the tensor?
        torch.testing.assert_close(image.data, torch.unsqueeze(data, dim=1))

    def test_is_invertible(self):
        self.assertTrue(Vec2Complex().is_invertible())

    def test_inverse(self):
        self.assertEqual(type(Vec2Complex().inverse()), Complex2Vec)


class TestCompress(TestCase):
    def test_apply_transform(self):
        transform = Compress()
        data = torch.tensor(
            [
                [
                    [
                        [[1.0, 0.0, 1.0], [1.0, 0.0, 3.0], [1.0, 2.0, 3.0]],
                        [[2.0, 3.0, 1.0], [3.0, -1.2, -1.0], [2.0, 2.3, -1.0]],
                    ],
                    [
                        [[0.0, 0.5, 0.0], [-1.0, 2.0, 1.0], [1.0, 3.0, 1.0]],
                        [[0.0, 1.0, 0.0], [1.0, -1.0, -2.0], [-1.0, 2.0, 2.0]],
                    ],
                ]
            ]
        )
        image = NDScalarImage(tensor=data)
        subject = torchio.Subject(one_image=image)
        transform.apply_transform(subject)

        torch.testing.assert_close(
            image.data[:, 0, :, :, :],
            torch.tensor(
                [
                    [
                        [[1.0, 0.0], [1.0, 0.0], [-1.0, 2.0]],
                        [[2.0, 3.0], [3.0, -1.2], [1.0, 2.3]],
                    ]
                ]
            ),
        )
        torch.testing.assert_close(
            image.data[:, 1, :, :, :],
            torch.tensor(
                [
                    [
                        [[1.0, 0.5], [3.0, 2.0], [1.0, 3.0]],
                        [[1.0, 1.0], [-1.0, -1.0], [-2.0, 2.0]],
                    ]
                ]
            ),
        )

    def test_is_invertible(self):
        self.assertTrue(Compress().is_invertible())

    def test_inverse(self):
        self.assertEqual(type(Compress().inverse()), Decompress)


class TestDecompress(TestCase):
    def test_apply_transform(self):
        transform = Compress()
        inverse_transform = Decompress()
        data = torch.tensor(
            [
                [
                    [
                        [[1.0, 0.0, 1.0], [1.0, 0.0, 3.0], [1.0, 2.0, 3.0]],
                        [[2.0, 3.0, 1.0], [3.0, -1.2, -1.0], [3.0, 2.3, -1.0]],
                    ],
                    [
                        [[0.0, 0.5, 0.0], [-1.0, 2.0, -1.0], [1.0, 3.0, 1.0]],
                        [[0.0, 1.0, 0.0], [1.0, -1.0, -2.0], [-1.0, 2.0, 2.0]],
                    ],
                ]
            ]
        )
        image = NDScalarImage(tensor=data)
        subject = torchio.Subject(one_image=image)
        transform.apply_transform(subject)
        inverse_transform.apply_transform(subject)
        torch.testing.assert_close(image.data, data)

    def test_is_invertible(self):
        self.assertTrue(Decompress().is_invertible())

    def test_inverse(self):
        self.assertEqual(type(Decompress().inverse()), Compress)
