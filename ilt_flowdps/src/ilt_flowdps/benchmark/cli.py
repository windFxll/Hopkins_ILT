import argparse
from dataclasses import asdict, replace
from datetime import datetime
import math
from pathlib import Path
from time import perf_counter

import torch
from hopkins_surrogate.api import DEFAULT_CHECKPOINT
from ilt.config import OptimizationConfig

from ..physics.lithography import LithographyObjective, load_optics
from ..physics.pixel_ilt import optimize_mask
from ..physics.regularizers import SPATIAL_PENALTIES
from ..prior.checkpoint import DEFAULT_RF_CHECKPOINT, load_prior
from ..sampling.flowdps import FlowConfig, sample_flowdps
from ..sampling.flowlps import LPSConfig, sample_flowlps
from ..sampling.flowsgs import SGSConfig, sample_flowsgs
from .layout import TARGET_LAYOUTS
from .report import (evaluate, save_binary, save_comparison, save_evaluation,
                     write_csv, write_json)


FLOW_METHODS = ("flowdps", "flowlps", "flowsgs")
METHODS = {
    f"{optics}-{algorithm}": (optics, algorithm)
    for algorithm in (*SPATIAL_PENALTIES, *FLOW_METHODS)
    for optics in ("socs", "proxy")
}
DEFAULT_METHODS = tuple(METHODS)
METHODS.update({"socs-ilt": METHODS["socs-tv"], "proxy-ilt": METHODS["proxy-tv"]})


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description="Compare SOCS and Proxy ILT and flow samplers.")
    parser.add_argument("--rf-checkpoint", type=Path, default=DEFAULT_RF_CHECKPOINT)
    parser.add_argument("--proxy-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--layout", choices=TARGET_LAYOUTS, default="iccad2013-m1-test3")
    parser.add_argument("--dataset-root", type=Path,
                        help="Dataset layout directory containing png/ and json/ (only for --layout dataset).")
    parser.add_argument("--dataset-mask", default="layout_00001")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Run exactly --max-iterations Adam steps for the selected ILT methods.")
    parser.add_argument("--tv-weight", type=float, default=0.01)
    parser.add_argument("--wavelet-weight", type=float, default=0.01)
    parser.add_argument("--wavelet-levels", type=int, default=3)
    parser.add_argument("--laplacian-weight", type=float, default=0.01)
    parser.add_argument("--binary-weight", type=float, default=OptimizationConfig.binary_weight)
    parser.add_argument("--area-weight", type=float, default=OptimizationConfig.area_weight)
    parser.add_argument("--flow-steps", type=int, default=250)
    parser.add_argument("--inner-steps", type=int, default=1,
                        help="Shared fallback for the number of optical updates per Euler step.")
    parser.add_argument("--socs-inner-steps", type=int,
                        help="Override the SOCS FlowDPS inner-step count.")
    parser.add_argument("--proxy-inner-steps", type=int,
                        help="Override the Proxy FlowDPS inner-step count.")
    parser.add_argument("--dps-scale", type=float, default=1.0)
    parser.add_argument("--max-guidance-rms", type=float, default=1.0)
    parser.add_argument("--flow-min-clean-weight", type=float, default=0.10)
    parser.add_argument("--lps-langevin-steps", type=int, default=LPSConfig.langevin_steps)
    parser.add_argument("--lps-proximal-steps", type=int, default=LPSConfig.proximal_steps)
    parser.add_argument("--lps-proximal-lr", type=float, default=LPSConfig.proximal_lr)
    parser.add_argument("--sgs-steps", type=int, default=SGSConfig.steps)
    parser.add_argument("--sgs-likelihood-steps", type=int, default=SGSConfig.likelihood_steps)
    parser.add_argument("--sgs-prior-steps", type=int, default=SGSConfig.prior_steps)
    parser.add_argument("--rf-encoding", choices=("signed", "unit"), default="signed")
    parser.add_argument("--rf-polarity", choices=("feature-high", "feature-low"), default="feature-high")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=0,
                        help="Print completed-step lines every N steps. 0 disables extra lines.")
    parser.add_argument("--allow-proxy-basis-mismatch", action="store_true")
    args = parser.parse_args(argv)
    if args.layout == "dataset" and args.dataset_root is None:
        parser.error("--layout dataset requires --dataset-root (containing png/ and json/).")
    if min(args.max_iterations, args.flow_steps, args.inner_steps, args.log_every) <= 0:
        parser.error("Iteration counts and logging interval must be positive.")
    if any(count is not None and count <= 0 for count in (args.socs_inner_steps, args.proxy_inner_steps)):
        parser.error("Per-method inner-step counts must be positive.")
    if args.lps_langevin_steps < 0 or min(args.lps_proximal_steps, args.sgs_steps,
                                         args.sgs_likelihood_steps, args.sgs_prior_steps) <= 0:
        parser.error("LPS and SGS step counts must be positive. LPS Langevin steps may be zero.")
    if not math.isfinite(args.lps_proximal_lr) or args.lps_proximal_lr <= 0:
        parser.error("--lps-proximal-lr must be finite and positive.")
    if any(not math.isfinite(w) or w < 0 for w in (
        args.tv_weight, args.wavelet_weight, args.laplacian_weight, args.binary_weight, args.area_weight
    )):
        parser.error("Mask penalty weights must be finite and non-negative.")
    if args.wavelet_levels <= 0:
        parser.error("--wavelet-levels must be positive.")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative.")
    if not math.isfinite(args.dps_scale) or args.dps_scale <= 0:
        parser.error("--dps-scale must be finite and positive.")
    if not math.isfinite(args.max_guidance_rms) or args.max_guidance_rms <= 0:
        parser.error("--max-guidance-rms must be finite and positive.")
    if not 0 < args.flow_min_clean_weight <= 1:
        parser.error("--flow-min-clean-weight must be in (0, 1].")
    if len({METHODS[method] for method in args.methods}) != len(args.methods):
        parser.error("Pass each method once.")
    return args


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _announce(message):
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def _run_method(method, optics, target, ilt_config, weights, wavelet_levels, flow_config, lps_config,
                sgs_config, prior, noise, seed, on_step):
    optics_name, algorithm = METHODS[method]
    imaging = optics[optics_name]
    device = target.device
    if algorithm in SPATIAL_PENALTIES:
        optimize_mask(imaging, target, replace(ilt_config, iterations=1), algorithm, weights[algorithm],
                      label=method, on_step=None, show_progress=False, wavelet_levels=wavelet_levels)
        _sync(device)
        start = perf_counter()
        mask, history, timing = optimize_mask(
            imaging, target, ilt_config, algorithm, weights[algorithm],
            label=method, on_step=on_step, show_progress=on_step is None,
            wavelet_levels=wavelet_levels
        )
        _sync(device)
        timing["seconds"] = perf_counter() - start
        timing["rf_evaluations"] = 0
        timing["optical_gradient_evaluations"] = timing["steps"]
        return mask, history, timing

    objective = LithographyObjective(imaging, target, ilt_config)

    def run(warmup):
        flow = replace(flow_config, steps=1) if warmup else flow_config
        display = dict(label=method, show_progress=not warmup and on_step is None,
                       on_step=None if warmup else on_step)
        if algorithm == "flowdps":
            return sample_flowdps(prior, objective, noise, flow, **display)
        generator = torch.Generator(device=device).manual_seed(seed + 1)
        if algorithm == "flowlps":
            return sample_flowlps(prior, objective, noise, flow, lps_config,
                                  generator=generator, **display)
        sgs = replace(sgs_config, steps=1, likelihood_steps=1, prior_steps=1) if warmup else sgs_config
        return sample_flowsgs(prior, objective, noise, flow, sgs, generator=generator, **display)

    run(warmup=True)
    _sync(device)
    start = perf_counter()
    mask, history = run(warmup=False)
    _sync(device)
    elapsed = perf_counter() - start
    if algorithm == "flowsgs":
        steps = sgs_config.steps
        rf_calls = steps * sgs_config.prior_steps
        optical_updates = steps * sgs_config.likelihood_steps
    else:
        steps = flow_config.steps
        rf_calls = steps
        inner = (flow_config.inner_steps if algorithm == "flowdps"
                 else lps_config.langevin_steps + lps_config.proximal_steps)
        optical_updates = steps * inner
    stop_reason = "Euler schedule completed" if algorithm == "flowdps" else "sampling schedule completed"
    timing = dict(seconds=elapsed, steps=steps, stop_reason=stop_reason, best_step=None,
                  rf_evaluations=rf_calls, optical_gradient_evaluations=optical_updates)
    return (mask >= 0.5).float(), history, timing


def main(argv=None):
    args = _arguments(argv)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    needs_prior = any(METHODS[method][1] in FLOW_METHODS for method in args.methods)
    if needs_prior and not args.rf_checkpoint.is_file():
        raise FileNotFoundError(args.rf_checkpoint)

    cfg = replace(OptimizationConfig(), iterations=args.max_iterations, early_stop=not args.no_early_stop,
                  log_every=args.log_every, mask_filter_sigma_px=10.0, border_margin_px=0,
                  boundary_radius_px=8, tv_weight=0.01)
    pixel_cfg = replace(cfg, tv_weight=args.tv_weight, binary_weight=args.binary_weight,
                        area_weight=args.area_weight)
    weights = dict(tv=args.tv_weight, wavelet=args.wavelet_weight, laplacian=args.laplacian_weight)
    lps_cfg = LPSConfig(langevin_steps=args.lps_langevin_steps, proximal_steps=args.lps_proximal_steps,
                        proximal_lr=args.lps_proximal_lr)
    sgs_cfg = SGSConfig(steps=args.sgs_steps, likelihood_steps=args.sgs_likelihood_steps,
                        prior_steps=args.sgs_prior_steps)
    flow_cfg = FlowConfig(steps=args.flow_steps, inner_steps=args.inner_steps,
                          scale=args.dps_scale, max_guidance_rms=args.max_guidance_rms,
                          min_clean_weight=args.flow_min_clean_weight,
                          hardening_fraction=cfg.hardening_fraction,
                          resist_beta_start=cfg.resist_beta_start, resist_beta_end=cfg.resist_beta_end,
                          encoding=args.rf_encoding, polarity=args.rf_polarity, log_every=args.log_every)
    inner_steps_by_method = {
        "socs-flowdps": args.socs_inner_steps or flow_cfg.inner_steps,
        "proxy-flowdps": args.proxy_inner_steps or flow_cfg.inner_steps,
    }
    dps_configs = {
        method: replace(flow_cfg, inner_steps=inner_steps_by_method[method])
        for method in args.methods if method in inner_steps_by_method
    }

    _sync(device)
    start = perf_counter()
    optics = load_optics(args.proxy_checkpoint, device,
                         allow_basis_mismatch=args.allow_proxy_basis_mismatch)
    proxy = optics["proxy"]
    if any(METHODS[method][1] == "wavelet" for method in args.methods):
        max_levels = (proxy.grid_size & -proxy.grid_size).bit_length() - 1
        if args.wavelet_levels > max_levels:
            raise ValueError(f"Haar levels must be at most {max_levels} on a {proxy.grid_size}px grid")
    layout_options = (dict(root=args.dataset_root, sample=args.dataset_mask)
                      if args.layout == "dataset" else {})
    target_np, target_source = TARGET_LAYOUTS[args.layout](proxy.grid_size, proxy.pixel_size_nm,
                                                        **layout_options)
    target = torch.tensor(target_np, device=device, dtype=torch.float32)
    target_np = target_np > 0.5
    prior, prior_metadata, noise = None, None, None
    if needs_prior:
        prior, prior_metadata = load_prior(args.rf_checkpoint, device, proxy.grid_size)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        noise = torch.randn((1, 1, *target.shape), device=device, generator=generator)
    _sync(device)
    setup_seconds = perf_counter() - start

    output = args.output or (Path("runs") / datetime.now().strftime("benchmark-%Y%m%d-%H%M%S-%f"))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    print(f"Output: {output}", flush=True)
    save_binary(output / "target.png", target_np)
    with torch.no_grad():
        baseline, _, _, _ = evaluate(proxy, target, target_np)
    write_json(output / "run_config.json", dict(
        methods=args.methods, seed=args.seed, device=str(device), progress_every=args.progress_every,
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        torch=str(torch.__version__), setup_seconds=setup_seconds,
        grid_size=proxy.grid_size, pixel_size_nm=proxy.pixel_size_nm, socs_orders=proxy.num_orders,
        boundary_mode=proxy.config.boundary_mode, proxy_threshold=proxy.threshold,
        proxy_checkpoint=str(args.proxy_checkpoint.resolve()),
        proxy_strict_load=not args.allow_proxy_basis_mismatch, rf_checkpoint=prior_metadata,
        flow_guidance="clean_endpoint_jacobian_approximation",
        layout=args.layout, target_source=target_source,
        ilt=asdict(pixel_cfg), spatial_penalty_weights=weights, wavelet_levels=args.wavelet_levels,
        flowdps_mask_penalty_weights=dict(tv=cfg.tv_weight, binary=cfg.binary_weight, area=cfg.area_weight),
        flow=asdict(flow_cfg), solver="euler",
        flow_inner_steps_by_method={method: config.inner_steps for method, config in dps_configs.items()},
        flowlps=asdict(lps_cfg), flowsgs=asdict(sgs_cfg),
        proxy_baseline_target_as_mask=baseline,
    ))

    rows, panels = [], []
    for index, method in enumerate(args.methods, start=1):
        group = f"[{index}/{len(args.methods)}] {method}"
        algorithm = METHODS[method][1]
        method_flow = dps_configs.get(method, flow_cfg)
        method_ilt_config = pixel_cfg if algorithm in SPATIAL_PENALTIES else cfg
        if algorithm in SPATIAL_PENALTIES:
            limit = pixel_cfg.iterations
            stop_mode = "with early stopping" if pixel_cfg.early_stop else "without early stopping"
            schedule = f"{limit} Adam steps {stop_mode}, {algorithm} weight={weights[algorithm]:g}"
            if algorithm == "wavelet":
                schedule += f", Haar levels={args.wavelet_levels}"
            schedule += f", binary={pixel_cfg.binary_weight:g}, area={pixel_cfg.area_weight:g}"
        elif algorithm == "flowsgs":
            limit = sgs_cfg.steps
            schedule = (f"{limit} Gibbs rounds, {sgs_cfg.likelihood_steps} optical Langevin and "
                        f"{sgs_cfg.prior_steps} prior SDE steps per round")
        elif algorithm == "flowlps":
            limit = method_flow.steps
            schedule = (f"{limit} RF steps, {lps_cfg.langevin_steps} Langevin and "
                        f"{lps_cfg.proximal_steps} proximal updates per step")
        else:
            limit = method_flow.steps
            schedule = f"Euler {limit} steps, {method_flow.inner_steps} optical update(s) per step"
        _announce(f"START {group} | {schedule}")

        def on_step(completed):
            if completed == 1 or completed % args.progress_every == 0 or completed == limit:
                _sync(device)
                _announce(f"STEP  {group} | {completed}/{limit} completed")

        progress = on_step if args.progress_every else None
        hard_mask, history, timing = _run_method(
            method, optics, target, method_ilt_config, weights, args.wavelet_levels,
            method_flow, lps_cfg, sgs_cfg, prior, noise, args.seed, progress
        )
        with torch.no_grad():
            metrics, mask, printed, aerial = evaluate(proxy, hard_mask, target_np)
        stats = dict(method=method, algorithm=algorithm, **timing,
                     flow_inner_steps=method_flow.inner_steps if algorithm == "flowdps" else None,
                     ms_per_step=1000 * timing["seconds"] / timing["steps"],
                     steps_per_second=timing["steps"] / timing["seconds"], **metrics)
        folder = output / method
        folder.mkdir()
        save_evaluation(folder, mask, printed, aerial, target_np)
        write_csv(folder / "history.csv", history)
        write_json(folder / "metrics.json", stats)
        rows.append(stats)
        panels.append((method, mask, printed, stats))
        write_csv(output / "summary.csv", rows)
        _announce(f"END   {group} | {stats['steps']}/{limit} completed | {stats['stop_reason']} | "
                  f"{stats['seconds']:.2f} s | {stats['ms_per_step']:.2f} ms/step | "
                  f"Proxy feature IoU {stats['feature_iou']:.5f}")
    save_comparison(output / "comparison.png", target_np, panels)
    _announce(f"ALL DONE | {output / 'summary.csv'} | {output / 'comparison.png'}")


if __name__ == "__main__":
    main()
