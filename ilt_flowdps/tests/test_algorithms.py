"""CPU regressions; no pretrained weights or downloaded dataset required."""

from contextlib import redirect_stderr
from dataclasses import replace
import io
import unittest

import torch
from torch.nn import functional as F
from ilt.config import OptimizationConfig
from ilt.optimization import PixelILT

from ilt_flowdps.benchmark.cli import _arguments
from ilt_flowdps.benchmark.layout import iccad_m1_test3_target
from ilt_flowdps.physics.lithography import LithographyObjective, to_mask
from ilt_flowdps.physics.pixel_ilt import optimize_mask
from ilt_flowdps.physics.regularizers import SPATIAL_PENALTIES, haar_details
from ilt_flowdps.sampling.flowdps import FlowConfig, sample_flowdps
from ilt_flowdps.sampling.flowlps import LPSConfig, sample_flowlps
from ilt_flowdps.sampling.flowsgs import SGSConfig, sample_flowsgs


class SmallOptics:
    config = type("Config", (), {"resist_threshold": 0.5})()

    def __call__(self, mask):
        return 1 - F.avg_pool2d(
            F.pad(mask[None, None], (1, 1, 1, 1), mode="circular"), 3, stride=1)[0, 0]


class FrozenPrior:
    def __init__(self):
        self.calls = 0

    def __call__(self, state, time):
        if torch.is_grad_enabled() or state.requires_grad:
            raise AssertionError("RF forward must not build an autograd graph")
        self.calls += 1
        return 0.2 - state


class CountingObjective(LithographyObjective):
    def __init__(self, *args):
        super().__init__(*args)
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        return super().__call__(*args)

    def print_loss(self, *args):
        self.calls += 1
        return super().print_loss(*args)


class AlgorithmTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.target = torch.zeros(16, 16)
        self.target[3:12, 4:7] = 1
        self.target[9:12, 7:13] = 1
        self.cfg = replace(OptimizationConfig(), iterations=4, log_every=1,
                           mask_filter_sigma_px=1, border_margin_px=0, boundary_radius_px=1,
                           tv_weight=0.01, early_stop=False)

    def test_regularizer_derivatives_and_constants(self):
        for name, penalty in SPATIAL_PENALTIES.items():
            with self.subTest(name=name):
                variable = torch.rand(8, 8, dtype=torch.double, requires_grad=True)
                self.assertTrue(torch.autograd.gradcheck(penalty, (variable,), fast_mode=True))
                gradient, = torch.autograd.grad(penalty(variable), variable)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()), 0)
                self.assertEqual(float(penalty(torch.ones(8, 8))), 0)
        with self.assertRaises(ValueError):
            haar_details(torch.ones(10, 10), levels=3)

    def test_tv_matches_upstream_pixelilt(self):
        with redirect_stderr(io.StringIO()):
            reference = PixelILT(SmallOptics(), self.cfg).optimize(self.target)
        mask, history, timing = optimize_mask(
            SmallOptics(), self.target, self.cfg, "tv", self.cfg.tv_weight,
            label="test", on_step=None, show_progress=False)
        torch.testing.assert_close(mask, reference.optimized_mask, rtol=0, atol=0)
        self.assertEqual(timing["steps"], reference.steps_completed)
        self.assertEqual(timing["best_step"], reference.best_step + 1)
        for actual, expected in zip(history, reference.history):
            self.assertAlmostEqual(actual["loss"], expected["loss"], places=7)
            self.assertEqual(actual["iou"], expected["iou"])

    def test_early_stop_and_forced_full_iterations(self):
        class ConstantOptics(SmallOptics):
            def __call__(self, mask):
                return mask * 0 + 0.5

        cfg = replace(self.cfg, iterations=20, learning_rate=0, minimum_learning_rate=0,
                      binary_weight=0, area_weight=0, early_stop=True,
                      early_stop_min_step=0, early_stop_window=2, early_stop_patience=1,
                      hardening_fraction=0.1)
        for enabled in (True, False):
            _, history, timing = optimize_mask(
                ConstantOptics(), self.target, replace(cfg, early_stop=enabled), "tv", 0,
                label="test", on_step=None, show_progress=False)
            self.assertEqual(len(history), timing["steps"])
            if enabled:
                self.assertLess(timing["steps"], cfg.iterations)
                self.assertIn("early convergence", timing["stop_reason"])
            else:
                self.assertEqual(timing["steps"], cfg.iterations)

    def test_all_flow_samplers_are_deterministic_without_rf_backward(self):
        flow = FlowConfig(steps=3, inner_steps=2, log_every=1)
        noise = torch.randn(1, 1, 16, 16)
        for method, expected_rf, expected_optical in (("dps", 3, 6), ("lps", 3, 9), ("sgs", 6, 9)):
            outputs = []
            for _ in range(2):
                prior = FrozenPrior()
                objective = CountingObjective(SmallOptics(), self.target, self.cfg)
                completed = []
                kwargs = dict(label=method, show_progress=False, on_step=completed.append)
                generator = torch.Generator().manual_seed(19)
                if method == "dps":
                    result, history = sample_flowdps(prior, objective, noise, flow, **kwargs)
                elif method == "lps":
                    result, history = sample_flowlps(
                        prior, objective, noise, flow, LPSConfig(proximal_steps=2),
                        generator=generator, **kwargs)
                else:
                    result, history = sample_flowsgs(
                        prior, objective, noise, flow,
                        SGSConfig(steps=3, likelihood_steps=3, prior_steps=2),
                        generator=generator, **kwargs)
                self.assertEqual(prior.calls, expected_rf)
                self.assertEqual(objective.calls, expected_optical)
                self.assertEqual(completed, [1, 2, 3])
                self.assertEqual(len(history), 3)
                self.assertTrue(torch.isfinite(result).all())
                self.assertTrue(((result >= 0) & (result <= 1)).all())
                self.assertFalse(result.requires_grad)
                outputs.append(result)
            torch.testing.assert_close(*outputs, rtol=0, atol=0)

    def test_encoding_and_polarity(self):
        state = torch.tensor([[[[-2., 0., 2.]]]])
        torch.testing.assert_close(to_mask(state, "signed", "feature-high"),
                                   torch.tensor([[0., 0.5, 1.]]))
        torch.testing.assert_close(to_mask(state, "signed", "feature-low"),
                                   torch.tensor([[1., 0.5, 0.]]))

    def test_portable_layout_and_argument_validation(self):
        args = _arguments(["--methods", "proxy-tv", "--no-early-stop"])
        self.assertEqual(args.layout, "iccad2013-m1-test3")
        self.assertTrue(args.no_early_stop)
        for invalid in (["--layout", "dataset"], ["--methods", "socs-tv", "socs-ilt"],
                        ["--tv-weight", "nan"], ["--proxy-inner-steps", "0"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                _arguments(invalid)
        target, metadata = iccad_m1_test3_target(800, 1.0)
        self.assertEqual(target.shape, (800, 800))
        self.assertGreater(target.sum(), 0)
        self.assertEqual(metadata["native_grid_size_px"], 1024)


if __name__ == "__main__":
    unittest.main()
