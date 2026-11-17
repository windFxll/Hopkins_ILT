# Final model

## `kernel15_cnn_k8.pt`

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

## 加载

```python
from hopkins_surrogate import HopkinsSurrogate

model = HopkinsSurrogate.from_pretrained(
    "Hopkins_surrogate/models/kernel15_cnn_k8.pt", device="cuda:0")
out = model.predict(mask)      # mask: [800,800], 1=opaque
```

checkpoint 内嵌：
- SOCS `OpticsConfig`（K=8、864 点光源、193 nm、NA 1.35、periodic）；
- 15% 核修正参数（实际 mean 15.00%、max 22.54%）；
- 残差 CNN 权重（hidden=24、downscale=4、`|ΔI| ≤ 0.15`）；
- `aerial_scale = 0.869140625` 与阈值 `0.346585`。

推理不需要训练 cache、原始数据集或额外配置。

## 评估报告

完整报告（数据划分、网络/损失配置、成像对比、过拟合判断）：

[`../../docs/reports/k8_kernel15_cnn_report.html`](../../docs/reports/k8_kernel15_cnn_report.html)
