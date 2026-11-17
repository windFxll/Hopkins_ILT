# Hopkins_surrogate

最终的光刻成像代理模型：**K=8 SOCS + 15% 频域核修正（冻结）+ 有界残差 CNN**。
推理封装在 [`hopkins_surrogate/api.py`](hopkins_surrogate/api.py)，模型文件为
[`models/kernel15_cnn_k8.pt`](models/kernel15_cnn_k8.pt)（约 95 KB，自包含）。

完整评估报告：
[`../docs/reports/k8_kernel15_cnn_report.html`](../docs/reports/k8_kernel15_cnn_report.html)。
机器可读指标：[`models/metrics.json`](models/metrics.json)。

## 模型指标

800×800 @ 1 nm、periodic、阈值 `0.346585`，1,000 个 held-out test 样本：

| 指标 | 值 |
|---|---:|
| clear IoU | **0.9799** |
| clear Dice | **0.9898** |
| contour CD | **1.447 nm** |
| 相对 K=8 SOCS baseline | IoU +6.86 pp，CD −5.48 nm |
| 相对 15% 核-only | IoU +4.93 pp，CD −3.97 nm |

## 安装

```bash
pip install torch                 # 按机器 CUDA 版本安装
pip install -e ../Hopkins
pip install -e .
```

## 最小调用

```python
import torch
from hopkins_surrogate import HopkinsSurrogate

model = HopkinsSurrogate.from_pretrained(device="cuda:0")

mask = torch.zeros(800, 800, device=model.device)
for x in range(0, 800, 400):
    mask[:, x:x + 100] = 1.0      # 1 = opaque

out = model.predict(mask)
aerial = out["aerial"]            # float, .dat units
clear = out["clear"]              # bool, 1 = clear
printed = out["printed"]          # bool, 1 = feature
```

- `model(mask)` 返回 aerial，且对 mask 可微；配合 `torch.autograd` 或 `PixelILT` 使用。
- `model.predict(mask)` 使用 `no_grad`，返回 `aerial`、`clear`、`printed`、`threshold`。
- 输入支持 `[800, 800]` 或 `[B, 800, 800]`；batch 内部按样本循环。
- 模型参数全部冻结；`model.config` 是底层 `OpticsConfig`，`model.threshold` 为 `0.346585`。
- 加载后不需要训练 cache：checkpoint 内嵌核修正、CNN、SOCS 配置与光度标定。

## 示例

```bash
python examples/quickstart.py           # 推理 + mask 梯度
python examples/ilt_with_surrogate.py   # 100 步 PixelILT smoke demo
```

完整保留的 ILT 案例（realistic interconnect，feature IoU 0.7456 → 0.9761，
CD 16.85 → 1.14 nm，优化 28.4 s）：
[`../docs/reports/ilt_case_realistic/README.md`](../docs/reports/ilt_case_realistic/README.md)。
案例生成脚本未包含在发布包中。

第二个示例需要同时安装 `ILT`：

```bash
pip install -e ../ILT
```

注意：代理模型固定 800×800 @ 1 nm；ILT 默认 demo profile 是 256×256 @ 4 nm，
使用时要按 1 nm/px 缩放滤波/边界参数（示例中已给出）。

## 包内结构

```text
Hopkins_surrogate/
├─ hopkins_surrogate/
│  ├─ api.py            # HopkinsSurrogate / load_surrogate
│  ├─ checkpoint.py     # checkpoint 构建与指纹校验
│  ├─ imaging.py        # 核修正后的 SOCS
│  ├─ correction.py     # B-spline 频域核修正
│  ├─ residual.py       # 残差 CNN 与 hybrid 组合
│  └─ ...               # 训练/评估模块（保留用于检查实现）
├─ models/kernel15_cnn_k8.pt
├─ examples/
└─ tests/
```

## 测试

```bash
pytest tests -q
```

测试使用最终 checkpoint 和合成 mask，不需要外部数据。
