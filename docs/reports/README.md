# Reports

- [k8_cnn_default_report.html](k8_cnn_default_report.html) — **当前默认模型**
  `Hopkins_surrogate/models/k8_cnn_default.pt`：干式空气数据集，
  K=8 SOCS + 轻量 CNN，阈值 `0.3357889756677475`。含 9 阈值 / defocus /
  光源对比和 ILT mask 交叉验证。Standalone HTML（图片全部内嵌，可直接转发）。
- [k8_kernel15_cnn_report.html](k8_kernel15_cnn_report.html) — 历史模型
  K=8 SOCS + 15% kernel correction + residual CNN. Standalone HTML
  (all figures embedded as base64); sections cover data/split, architecture,
  loss, final test metrics, per-sample imaging vs. binarized aerial image,
  and the overfitting check.

旧实验版本（air baseline / V1a / V1b 等）报告为本地归档，不包含在仓库。

The historical report is a static snapshot generated from
`Hopkins_surrogate/models/kernel15_cnn_k8.pt` and the held-out test split.
The internal experiment generator and its intermediate figures are not part
of this release.

Open the HTML directly in a browser; no external assets are required.

## Kept ILT case

- [ilt_case_realistic/README.md](ilt_case_realistic/README.md) — realistic
  interconnect optimized with `PixelILT` + the final surrogate for 1000 steps.
  Feature IoU 0.7456 → 0.9761, contour CD 16.85 → 1.14 nm, optimization
  28.4 s on RTX 3050 (35.2 steps/s). Includes the target, optimized mask,
  printed/aerial images, mismatch map, history CSV and `metrics.json`.
