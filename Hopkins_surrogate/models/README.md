# Models

## 当前默认：`k8_cnn_default.pt`（2026-11-24）

| 项目 | 内容 |
|---|---|
| 结构 | K=8 SOCS + 有界残差 CNN（**无核修正**） |
| 数据集 | 干式空气 S-Litho `slitho_air.slo`，10,000 张 |
| 光学 | NA=0.93、sigma 0.6–0.9、864 点、defocus=0 |
| 损失 | band 0.2 / dice 0.05 / full 5.0 / multi 0（无多阈值项） |
| 可训练参数 | 7,425（CNN；hidden=16、downscale=4、`|ΔI| ≤ 0.15`） |
| 阈值 / scale | **0.3357889756677475** / `0.85107421875` |
| 训练 | best epoch 23；第 31 epoch 后因主机内存分配失败中断，best checkpoint 完整使用 |
| val（1000） | clear IoU **0.9942**、full-field rel MSE **≈5e-5** |
| test（1000） | clear IoU **0.9942** / Dice **0.9971** / feature IoU **0.9803** / CD **0.575 nm** |
| multiconfig 10 test | 阈值 IoU **0.9922** / CD **0.71 nm**；defocus 0.9942 / 0.47；光源 0.9853 / 1.37；full RMSE 0.0041 / 0.0041 / 0.0187 |

```python
from hopkins_surrogate import HopkinsSurrogate

model = HopkinsSurrogate.from_pretrained(device="cuda:0")  # 默认发布模型
out = model.predict(mask)      # threshold = 0.3357889756677475
```

报告：[`k8_cnn_default_report.html`](../../docs/reports/k8_cnn_default_report.html)。
旧 air / 胶内版本已归档在本地 `Hopkins_surrogate/archive/`（不包含在仓库）。
`kernel15_cnn_k8.pt`（旧纯 AI）保留为历史模型。

## 历史模型

### `kernel15_cnn_k8.pt`

| 项目 | 内容 |
|---|---|
| 结构 | K=8 SOCS + 15% 频域 B-spline 核修正（冻结）+ 残差 CNN |
| 可训练参数 | 16,722（401 核修正 + 16,321 CNN） |
| 输入 | 800×800、1 nm/px、periodic；mask 1=opaque |
| 输出 | `.dat` 原始强度单位；阈值 `0.346585` |
| test（1,000 held-out） | clear IoU **0.9799** / Dice **0.9898** / contour **1.447 nm** |
| SHA-256 | `241500024ed08db94e70390525547176e6873ceb8e1a77d0cdfab4c543058828` |

模型选择依据 val_score；该 checkpoint 对应 epoch 30，是从 epoch 30 续训到 36
后仍未出现新 best 的最终版本。机器可读的完整指标见
[`metrics.json`](metrics.json)。

## 加载历史模型

默认模型直接调用 `HopkinsSurrogate.from_pretrained()`（见上）；历史模型需显式指定：

```python
from hopkins_surrogate import HopkinsSurrogate

model = HopkinsSurrogate.from_pretrained(
    "Hopkins_surrogate/models/kernel15_cnn_k8.pt", device="cuda:0")
out = model.predict(mask)      # mask: [800,800], 1=opaque
```

历史 checkpoint 内嵌：
- SOCS `OpticsConfig`（K=8、864 点光源、193 nm、NA 1.35、periodic）；
- 15% 核修正参数（实际 mean 15.00%、max 22.54%）；
- 残差 CNN 权重（hidden=24、downscale=4、`|ΔI| ≤ 0.15`）；
- `aerial_scale = 0.869140625` 与阈值 `0.346585`。

推理不需要训练 cache、原始数据集或额外配置。

## 评估报告

完整报告（数据划分、网络/损失配置、成像对比、过拟合判断）：

[`../../docs/reports/k8_kernel15_cnn_report.html`](../../docs/reports/k8_kernel15_cnn_report.html)

## 本地归档（不包含在仓库）

以下版本只保留在本地 `Hopkins_surrogate/archive/`，不随仓库发布：

- `k8_cnn_newdata.pt` — 胶内 S-Litho resist+BARC 数据集（阈值 0.3125683）
- `k8_cnn_air.pt` — air baseline（阈值 0.3357889757）
- `k8_cnn_air_v1a.pt` / `k8_cnn_air_v1b.pt` — 全光场损失中间版本
