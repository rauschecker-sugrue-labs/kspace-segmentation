import torch

from unittest import TestCase

from kseg.model.modules import DiceScore


class TestDiceScore(TestCase):
    def test_forward(self):
        module = DiceScore(smooth=0.0)
        y_pred = torch.Tensor([1, 1, 1, 0, 0, 0])

        y_true = torch.Tensor([1, 1, 1, 0, 0, 0])
        self.assertEqual(module(y_pred, y_true), 1.0)

        y_true = torch.Tensor([0, 0, 0, 1, 1, 1])
        self.assertEqual(module(y_pred, y_true), 0.0)

        y_true = torch.Tensor([0, 1, 1, 1, 0, 0])
        self.assertEqual(module(y_pred, y_true), 2.0 / 3)
