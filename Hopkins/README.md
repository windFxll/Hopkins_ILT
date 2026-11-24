# Hopkins

基于 PyTorch 的可微 Hopkins 部分相干成像包，提供两套等价实现：

- **Abbe**：每个光源点做一次移位光瞳 FFT，强度非相干叠加；
- **SOCS**：构造 Hopkins TCC，对 TCC 做特征分解，按特征值截断成 K 个相干系统的和。

ILT 优化通过本包完成前向推理并反向传播梯度。完整原理见
[`../docs/optical_model.md`](../docs/optical_model.md)。

## 目录结构

```text
Hopkins/
├─ hopkins/
│  ├─ config.py        光学参数 dataclass（OpticsConfig）
│  ├─ source.py        环形光源采样（面积权重）
│  ├─ pupil.py         移位光瞳构建（含离焦相位）
│  ├─ abbe/            AbbeImaging：可微 Abbe 光源点求和
│  ├─ socs/            SocsImaging + TCC 特征分解
│  ├─ factory.py       make_imaging(method="abbe" | "socs")
│  └─ resist/          软/硬光刻胶模型、高斯滤波
├─ tests/smoke_test.py     两种方法的前向/梯度自检
├─ tests/socs_test.py      SOCS 正确性测试
├─ pyproject.toml
└─ requirements.txt
```

## 运行自检

```bash
cd Hopkins
python tests/smoke_test.py --device cuda:0 --method both
python tests/socs_test.py --device cuda:0 --full
```

## 调用示例

```python
import torch
from hopkins import AbbeImaging, OpticsConfig, SocsImaging, make_imaging

cfg = OpticsConfig()                 # 256×256 @ 4nm 演示网格；NA=0.93、864 光源点
device = torch.device("cuda:0")
mask = torch.zeros((cfg.grid_size, cfg.grid_size), device=device)

abbe = AbbeImaging(cfg, device)
socs = SocsImaging(cfg, device, num_orders=16)   # 默认低秩特征分解
auto = make_imaging(cfg, device, method="socs", num_orders=16)

aerial = socs(mask)                  # (N, N)，开域归一化为 1
```

接口约定：亮场、暗图形 mask（1 = 不透明图形），返回归一化空间像；
两种方法都支持 `periodic`（默认，周期版图）和 `isolated` 边界，以及离焦、胶模糊。

## SOCS 截断

TCC 的秩受光源点数与频率子网格限制。当前默认光学配置（NA=0.93、
σ=0.6–0.9、864 光源点）在 256×256 @4 nm 演示网格下，频率子网格为
19×19，数值秩为 277；K=8 / 16 / 32 / 64 / 128 的能量占比分别为
**89.75% / 93.36% / 95.93% / 97.77% / 99.16%**。

历史基准（旧 NA=1.35 配置，864 点 / 27×27 子网格，与 3456 点 Abbe
参考比较，RTX 3050）：

| K | 保留能量 | RMS 误差 | 相对 Abbe864 加速 |
|---|---|---|---|
| 8 | 89.7% | 7.95×10⁻³ | 72× |
| 16 | 93.3% | 5.98×10⁻³ | 41× |
| 32 | 95.8% | 3.70×10⁻³ | 24× |
| 64 | 97.5% | 2.15×10⁻³ | 13× |
| 128 | 98.7% | 1.05×10⁻³ | 7.0× |
| 256 | 99.6% | 7.36×10⁻⁴ | 3.4× |
| 576 | 100% | 7.29×10⁻⁴（光源采样地板） | 1.6× |

当前默认代理模型采用 K=8（800×800 @ 1 nm；K=8 前向约 4.7 ms/图，
RTX 3050 历史实测）；更高保真场景可临时用 128/256。基准脚本与逐版图结果
不在本发布包中。

## 与代理模型的关系

当前默认代理模型位于
[`../Hopkins_surrogate/`](../Hopkins_surrogate/)：它在本包的 SOCS 之上加入
有界残差 CNN，**不使用频域核修正**。本包仍保留可选的 B-spline 频域核修正
实现；历史模型 `kernel15_cnn_k8.pt` 使用了它。若只想调用训练好的代理模型，见
[`../README.md`](../README.md) 和
[`../Hopkins_surrogate/README.md`](../Hopkins_surrogate/README.md)。
