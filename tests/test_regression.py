"""Small, dependency-light regression tests for the refactored utilities."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from deps.binarymask.binarymask import getTrainBinMask
from deps.common_py.dataIO import loadFiles_plus
from deps.common_py.evaluationBS import (
    evaluation_BS,
    evaluation_numpy,
    evaluation_numpy_entry,
    evaluation_numpy_entry_torch,
)
from deps.common_py.improcess import quantize
from deps.common_py.utils import setupSeed
from deps.function.arithdis import gaussianpdf, getc_X
from deps.params_input.params_input import QParams, numcheck


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.prediction = np.array([[255, 0], [0, 255]], dtype=np.uint8)
        self.truth = np.array([[255, 0], [255, 0]], dtype=np.uint8)

    def test_confusion_counts(self):
        self.assertEqual(evaluation_numpy_entry(self.prediction, self.truth), (1, 1, 1, 1))
        self.assertEqual(evaluation_BS(self.prediction, self.truth), (0.5, 0.5, 0.5, 1, 1, 1, 1))

    def test_metric_entry_points_agree(self):
        self.assertEqual(evaluation_numpy(self.prediction, self.truth), (0.5, 0.5, 0.5))
        torch_counts = evaluation_numpy_entry_torch(
            torch.from_numpy(self.prediction), torch.from_numpy(self.truth)
        )
        self.assertEqual(tuple(int(value) for value in torch_counts), (1, 1, 1, 1))


class UtilityTests(unittest.TestCase):
    def test_distribution_helpers(self):
        self.assertEqual(getc_X(-1, 1, 0.5).tolist(), [-1.0, -0.5, 0.0, 0.5, 1.0])
        values = gaussianpdf(torch.tensor([-1.0, 0.0, 1.0]), 0.0, 1.0)
        self.assertTrue(torch.allclose(values, values.flip(0)))

    def test_quantize(self):
        values = torch.tensor([-0.14, 0.0, 0.14])
        expected = torch.tensor([-0.14, 0.0, 0.14])
        self.assertTrue(torch.equal(quantize(values, delta=0.01, exp=2), expected))

    def test_seed(self):
        setupSeed(7)
        first = torch.rand(4)
        setupSeed(7)
        second = torch.rand(4)
        self.assertTrue(torch.equal(first, second))

    def test_binary_training_mask(self):
        mask = torch.tensor([[170.0, 255.0], [0.0, 170.0]])
        result = getTrainBinMask(mask, radius=1)
        expected = torch.tensor([[255.0, 255.0], [0.0, 255.0]])
        self.assertTrue(torch.equal(result, expected))


class CompatibilityTests(unittest.TestCase):
    def test_file_listing_keeps_sorted_substring_matching(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("frame_002.png", "frame_001.png", "notes.txt"):
                (root / name).touch()

            names, paths = loadFiles_plus(str(root), ".png")

        self.assertEqual(names, ["frame_001.png", "frame_002.png"])
        self.assertEqual(paths, [f"{directory}/frame_001.png", f"{directory}/frame_002.png"])

    def test_parameter_parser_keeps_legacy_option_names(self):
        QParams.m_params = {}
        params = QParams()
        argv = [
            "demo.py",
            "-pa_im",
            "input",
            "-pa_gt",
            "groundtruth",
            "-imgs_idx",
            "3",
            "4",
            "-idx_list",
            "1",
            "2",
            "-1",
            "-bk_rate",
            "0.25",
        ]

        self.assertEqual(params.setParams(len(argv), argv), 0)
        self.assertEqual(params["pa_im"], "input")
        self.assertEqual(params["pa_gt"], "groundtruth")
        self.assertEqual(params["imgs_idx"], [3, 4])
        self.assertEqual(params["idx_list"], [[1, 2]])
        self.assertEqual(params["bk_rate"], 0.25)
        self.assertTrue(numcheck("2.5"))
        self.assertFalse(numcheck("not-a-number"))


if __name__ == "__main__":
    unittest.main()
