# ILT

基于 PyTorch 的像素型逆向光刻（ILT）优化包。光学成像默认由上级目录的
`Hopkins/` 包提供（可微 Abbe 或 SOCS 前向，运行时可切换），也可以直接换成
`Hopkins_surrogate/` 的最终代理模型；本包负责 mask 参数化、损失函数和
Adam 优化循环。

## 目录结构

```text
ILT/
├─ ilt/
│  ├─ config.py             优化参数 dataclass
│  ├─ optimization/         mask 参数化、损失、ILT 主循环
│  ├─ layouts/              可复用参数化版图
│  └─ utils/                评估指标、输出和绘图
├─ scripts/run_demo.py       demo 入口
├─ benchmarks/               ICCAD-2013 版图
└─ requirements.txt
```

## 运行

使用已有的 PyTorch CUDA 环境：

```powershell
cd ILT
python -m pip install -r requirements.txt

# 光学前向/梯度自检（在 Hopkins 目录）
cd ..\Hopkins
python tests\smoke_test.py --device cuda:0

# 回到 ILT 跑完整 demo
cd ..\ILT
python scripts\run_demo.py --device cuda:0

# SOCS 版本（K=16，低秩 TCC 特征分解）
python scripts\run_demo.py --device cuda:0 --imaging socs --socs-orders 16
```

光学算法相关参数：

```text
--imaging {abbe,socs}              成像方法（默认 abbe）
--socs-orders {8,16,32}            SOCS 截断阶数（默认 16；采用 8/16/32 三档）
--socs-kernel-batch-size N         SOCS 特征核分批大小（默认 8）
--socs-eigh-mode {low_rank,explicit,auto}
                                   特征分解路径（默认 low_rank）
```

SOCS 默认输出到 `output/demo_socs`（其他案例同理，自动加 `_socs` 后缀）。

如果目标 GPU 不是第 0 张卡，请相应修改 `--device`。

更复杂的多拐弯/密集细线版图：

```powershell
python scripts\preview_layout.py --layout complex
python scripts\run_demo.py --device cuda:0 --layout complex --iterations 1000
```

ICCAD-2013 原生 1024×1024、1 nm/px 配置（GTX 1080）：

```powershell
python scripts\run_demo.py --device cuda:0 --profile iccad-native --target benchmarks\ICCAD2013\M1_test3.glp --iterations 1000
```

`M1_test3.glp` 来自公开的
[OpenILT ICCAD-2013 benchmark](https://github.com/OpenOPC/OpenILT/tree/main/benchmark/ICCAD2013)。
可用 `scripts/prepare_glp_benchmark.py` 将 GLP 转成 NPY/PNG；这些生成文件默认不纳入 Git。

demo 最多运行 1000 步，在 beta 硬化结束后同时监控 loss、IoU 和 hard-mask
变化率并可提前收敛。demo profile 用 864 个光源点优化，再用 3456 个光源点独立验证；
原生 ICCAD profile 使用 1536 点验证。最终输出在工作目录的 `output/`（按案例分目录）。

## 使用最终代理模型（Hopkins_surrogate）

`PixelILT` 只要求 imaging 模型可调用、并且暴露 `.config.resist_threshold`。
最终的 `HopkinsSurrogate` 满足这两个条件，且 `model(mask)` 对 mask 可微：

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

# 代理模型固定 800x800 @ 1 nm；target 1=feature
target = torch.as_tensor(
    realistic_interconnect(model.grid_size, model.pixel_size_nm),
    dtype=torch.float32, device=model.device)

opt_cfg = replace(
    OptimizationConfig(),
    iterations=200,
    log_every=10,
    mask_filter_sigma_px=5.0,     # 5 nm @ 1 nm/px
    border_margin_px=40,
    boundary_radius_px=8,
    tv_weight=2.0e-3,
)
result = PixelILT(model, opt_cfg).optimize(target)

printed = hard_resist(result.optimized_aerial, model.threshold)
print(binary_iou(printed, target))
```

注意：ILT 默认 `demo` profile 是 256×256 @ 4 nm，与代理模型的 800×800 @ 1 nm
不兼容；使用代理模型时请按 1 nm/px 缩放 `mask_filter_sigma_px`、
`border_margin_px`、`boundary_radius_px` 和 `tv_weight`（上面的参数与
`iccad-native` profile 一致）。完整示例见
[`../Hopkins_surrogate/examples/ilt_with_surrogate.py`](../Hopkins_surrogate/examples/ilt_with_surrogate.py)。
一个保留的 1000 步完整案例（realistic interconnect，feature IoU 0.7456 → 0.9761，
CD 16.85 → 1.14 nm，优化 28.4 s）在
[`../docs/reports/ilt_case_realistic/README.md`](../docs/reports/ilt_case_realistic/README.md)。

## 设计要点

- mask 由可训练 logits 经高斯制造滤波、设计窗边界约束和 sigmoid 生成。
- 优化期间用可微 soft resist，验证用硬阈值 resist。
- 损失包含像素/边界加权误差、TV、二值化和面积正则。
- 光源点分批传播，控制 8 GB 显存上的峰值占用。
- 保存 hard-mask 验证 IoU 最佳的状态，而不是盲目使用末轮。

## RTX 3050 demo 实测（864 点光源，3456 点验证）

| 方法 | 早停步数 | 最佳步 | 优化时间 | hard-resist IoU | 轮廓平均距离 |
|---|---|---|---|---|---|
| Abbe（864 点） | 716 / 1000 | 595 | 120.4 s | 0.8662 → 0.9813 | 6.65 → 0.81 nm |
| SOCS K=16 | 706 / 1000 | 330 | 8.9 s | 0.8664 → 0.9811 | 6.65 → 0.82 nm |

- 峰值显存：Abbe 502.8 MB，SOCS 450.3 MB
- SOCS K=16 保留 93.26% 特征值能量；低秩初始化约 0.16 s
- 两种方法的最终 IoU 差 0.0001，轮廓距离差 0.008 nm；SOCS 训练快约 13.6 倍

