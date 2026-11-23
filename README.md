# Hopkins_ILT

基于 PyTorch 的可微 Hopkins 光刻成像、像素型逆光刻（ILT）优化，以及最终训练好的
**K=8 光刻成像代理模型**。

- [`Hopkins/`](Hopkins/)：可微 Abbe / SOCS 部分相干成像，支持周期与孤立边界、离焦、胶模型。
- [`ILT/`](ILT/)：像素型 ILT 优化（mask 参数化、损失、Adam 循环）。
- [`ilt_flowdps/`](ilt_flowdps/)：SOCS/Proxy 的 TV、小波、拉普拉斯 ILT，以及实验性
  FlowDPS / FlowLPS / FlowSGS 对比；包含统一 Proxy 评价、耗时与 mask 可视化。
- [`Hopkins_surrogate/`](Hopkins_surrogate/)：最终代理模型（K=8 SOCS + 15% 频域核修正 +
  残差 CNN）及其推理封装。
- [`docs/reports/k8_kernel15_cnn_report.html`](docs/reports/k8_kernel15_cnn_report.html)：
  最终模型的完整评估报告（含逐样本成像 vs 二值化 aerial image 对比）。

## 最终模型

| 项目 | 内容 |
|---|---|
| 结构 | K=8 SOCS + 15% 频域 B-spline 核修正（冻结）+ 残差 CNN |
| 可训练参数 | 16,722（401 核修正 + 16,321 CNN） |
| 输入 | 800 × 800、1 nm/px、周期边界；mask 1=opaque |
| 输出 | `.dat` 原始强度单位；阈值 `0.346585` |
| test（1,000 held-out） | clear IoU **0.9799** / Dice **0.9898** / contour **1.447 nm** |
| checkpoint | [`Hopkins_surrogate/models/kernel15_cnn_k8.pt`](Hopkins_surrogate/models/kernel15_cnn_k8.pt)（约 95 KB，自包含） |
| 机器可读指标 | [`Hopkins_surrogate/models/metrics.json`](Hopkins_surrogate/models/metrics.json) |

checkpoint 内嵌核修正参数、CNN 权重、SOCS 配置和光度标定，推理时不需要训练 cache 或原始数据集。

## 安装

需要 Python ≥ 3.10。建议先按自己的 CUDA 版本安装 PyTorch，再以 editable 方式安装三个包：

```bash
# 1) 安装 PyTorch（示例；请按机器 CUDA 版本选择 wheel/index-url）
pip install torch

# 2) 安装本项目
pip install -e Hopkins
pip install -e Hopkins_surrogate

# 3) 可选：运行 ILT demo 时需要
pip install -e ILT
```

CPU 也可以运行（800² 单张前向约 0.1 s）；GPU（RTX 3050 实测）约 6–7 ms/图。

## 快速调用代理模型

```python
import torch
from hopkins_surrogate import HopkinsSurrogate

# 显式指定设备；CPU 环境写 device="cpu"（省略则由 API 自动选择）
model = HopkinsSurrogate.from_pretrained(device="cuda:0")

# 800 x 800 @ 1 nm；1 = opaque feature，0 = clear
mask = torch.zeros(800, 800, device=model.device)
for x in range(0, 800, 400):
    mask[:, x:x + 100] = 1.0       # 100 nm line / 400 nm pitch

# 推理（no_grad）：返回 continuous aerial 与两种二值结果
out = model.predict(mask)
aerial = out["aerial"]             # float，.dat 原始单位
clear = out["clear"]               # bool，1 = clear
printed = out["printed"]           # bool，1 = feature（与 ILT 的 target 一致）

print(aerial.shape, model.threshold)   # (800, 800) 0.346585
```

也可以直接调用 `model(mask)`，它返回 aerial，并且**对 mask 可微**：

```python
mask = mask.clone().requires_grad_(True)
loss = model(mask).mean()
loss.backward()
# mask.grad 可用；模型自身参数全部冻结
```

### 输入 / 输出约定

| 对象 | 形状 | 语义 |
|---|---|---|
| `mask` | `[800, 800]` 或 `[B, 800, 800]` | 1=opaque feature，0=clear；允许软值 `[0,1]` |
| `aerial` | 同 mask | `.dat` 原始强度单位（已乘 `aerial_scale=0.869140625`） |
| `clear` | 同 mask，bool | `aerial > 0.346585`，1=clear |
| `printed` | 同 mask，bool | `aerial < 0.346585`，1=feature |

- 固定 800×800、1 nm/px、periodic；其他网格会直接报错。
- batch 输入按 Python 循环处理，不是向量化批处理。
- 模型参数全部 `requires_grad=False`；梯度只会回传到 mask。
- `predict()` 内部使用 `torch.no_grad()`；ILT 优化请使用 `model(mask)`。

## 与 ILT 一起使用

`HopkinsSurrogate` 可以直接作为 `PixelILT` 的 imaging 模型。它暴露了
`model.config.resist_threshold`，且 `model(mask)` 可微：

```python
from dataclasses import replace
import torch

from hopkins.resist import hard_resist
from hopkins_surrogate import HopkinsSurrogate
from ilt.config import OptimizationConfig
from ilt.layouts import realistic_interconnect
from ilt.optimization import PixelILT
from ilt.utils.metrics import binary_iou

model = HopkinsSurrogate.from_pretrained(device="cuda:0")

# 代理模型要求 800x800 @ 1 nm；target 1=feature
target = torch.as_tensor(
    realistic_interconnect(model.grid_size, model.pixel_size_nm),
    dtype=torch.float32, device=model.device)

# ILT 默认 demo 是 256x256 @ 4 nm；这里按 1 nm/px 重新缩放滤波参数
opt_cfg = replace(
    OptimizationConfig(),
    iterations=200,
    log_every=10,
    mask_filter_sigma_px=5.0,
    border_margin_px=40,
    boundary_radius_px=8,
    tv_weight=2.0e-3,
)

result = PixelILT(model, opt_cfg).optimize(target)
printed = hard_resist(result.optimized_aerial, model.threshold)
print("IoU:", binary_iou(printed, target))
```

可直接运行两个示例：

```bash
python Hopkins_surrogate/examples/quickstart.py        # 推理 + mask 梯度
python Hopkins_surrogate/examples/ilt_with_surrogate.py # 100 步 ILT smoke
```

完整保留的 ILT 案例在
[`docs/reports/ilt_case_realistic/`](docs/reports/ilt_case_realistic/)：
realistic interconnect、1000 步、feature IoU **0.7456 → 0.9761**、
contour CD **16.85 → 1.14 nm**，优化耗时 **28.4 s**（35.2 steps/s，RTX 3050）。
案例包含 target、optimized mask、printed、aerial、mismatch、收敛曲线和
`metrics.json`。案例生成脚本未包含在发布包中；结果作为静态快照保留。

## 目录结构

```text
Hopkins_ILT/
├─ Hopkins/                     # 可微 Abbe/SOCS 成像包
│  ├─ hopkins/
│  ├─ tests/
│  └─ pyproject.toml
├─ ILT/                         # 像素型 ILT 优化包
│  ├─ ilt/
│  ├─ scripts/                  # demo / 版图预览
│  ├─ benchmarks/               # ICCAD-2013 M1_test3.glp
│  └─ pyproject.toml
├─ Hopkins_surrogate/           # 最终代理模型
│  ├─ hopkins_surrogate/        # 核心包 + api.py
│  ├─ models/kernel15_cnn_k8.pt # 最终 checkpoint
│  ├─ examples/                 # 推理 / ILT 示例
│  ├─ tests/
│  └─ pyproject.toml
├─ docs/
│  ├─ optical_model.md          # Hopkins 光学模型原理
│  ├─ ilt_gradient_descent.md   # ILT 方法说明
│  └─ reports/                  # 最终 HTML 评估报告 + 保留的 ILT 案例
└─ README.md
```

## 测试

```bash
# 代理模型（含最终 checkpoint 的前向/反向与 ILT 梯度检查）
pytest Hopkins_surrogate/tests -q

# Hopkins 成像包（脚本式自检）
python Hopkins/tests/smoke_test.py --device cpu --method both
python Hopkins/tests/socs_test.py --device cpu
```

测试使用合成 mask，不依赖外部数据集。

## 报告

最终模型的完整评估（数据划分、网络/损失配置、test 指标、6 组全图 + 3 组局部成像对比、
过拟合判断）见：

[docs/reports/k8_kernel15_cnn_report.html](docs/reports/k8_kernel15_cnn_report.html)

## 数据与发布范围

- 本仓库包含最终 checkpoint（约 95 KB）、推理代码、示例、测试和评估报告。
- 不包含 24 GB 训练 cache、原始数据集、训练中间产物、旧模型以及探索性对比图；
  这些内容保留在本地 `E:/hopkins_surrogate/` 归档中。
- 训练与评估模块仍保留在 `Hopkins_surrogate/hopkins_surrogate/` 内，便于检查实现；
  发布包不附带训练脚本和实验记录。
