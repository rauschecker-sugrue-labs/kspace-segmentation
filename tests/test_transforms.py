import torch
import torchio

from unittest import TestCase

from kseg.data.transforms import (
    KSpace,
    InverseKSpace,
    Complex2Vec,
    Vec2Complex,
)


class TestKSpace(TestCase):
    def test_apply_transform(self):
        transform = KSpace()

        xs = torch.tensor([-1.0, -0.5, 0.5, 1])
        ys = torch.tensor([-1.0, -0.5, 0.5, 1])
        x, y = torch.meshgrid(xs, ys, indexing='xy')
        image = torchio.ScalarImage(
            tensor=torch.stack([x + y, x, y]).unsqueeze(0)
        )

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
        self.assertTrue(KSpace().is_invertible())

    def test_inverse(self):
        self.assertEqual(type(KSpace().inverse()), InverseKSpace)


class TestInverseKSpace(TestCase):
    def test_apply_transform(self):
        transform = KSpace()
        inverse_transform = InverseKSpace()

        xs = torch.tensor([-1.0, -0.5, 0.5, 1])
        ys = torch.tensor([-1.0, -0.5, 0.5, 1])
        x, y = torch.meshgrid(xs, ys, indexing='xy')
        data = torch.stack([x + y, x, y]).unsqueeze(0)

        image = torchio.ScalarImage(tensor=data)
        subject = torchio.Subject(one_image=image)
        transform.apply_transform(subject)
        inverse_transform.apply_transform(subject)

        torch.testing.assert_close(subject.one_image.data, data)

    def test_is_invertible(self):
        self.assertTrue(InverseKSpace().is_invertible())

    def test_inverse(self):
        self.assertEqual(type(InverseKSpace().inverse()), KSpace)


class TestComplex2Vec(TestCase):
    def test_apply_transform(self):
        transform = Complex2Vec()
        image = torchio.ScalarImage(
            tensor=torch.tensor(
                [
                    [
                        [[0.0 + 1.0j, 1.0 - 1.0j], [0.0 + 1.0j, 1.0 - 1.0j]],
                        [[1, 1.0j], [2 + 1.0j, -1.0 + 2.0j]],
                    ]
                ]
            )
        )
        subject = torchio.Subject(one_image=image)
        transform.apply_transform(subject)
        torch.testing.assert_close(
            image.data[:, :, :, 0],
            torch.tensor([[[0.0, 1.0], [0.0, 1.0]], [[1, 0], [2, -1.0]]]),
        )
        torch.testing.assert_close(
            image.data[:, :, :, 1],
            torch.tensor([[[1.0, -1.0], [1.0, -1.0]], [[0, 1], [1, 2.0]]]),
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
                    [[0.0 + 1.0j, 1.0 - 1.0j], [0.0 + 1.0j, 1.0 - 1.0j]],
                    [[1, 1.0j], [2 + 1.0j, -1.0 + 2.0j]],
                ]
            ]
        )
        image = torchio.ScalarImage(tensor=data)
        subject = torchio.Subject(one_image=image)
        transform.apply_transform(subject)
        inverse_transform.apply_transform(subject)
        torch.testing.assert_close(image.data, data)

    def test_is_invertible(self):
        self.assertTrue(Vec2Complex().is_invertible())

    def test_inverse(self):
        self.assertEqual(type(Vec2Complex().inverse()), Complex2Vec)
