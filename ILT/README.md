# ILT

基于 PyTorch 的像素型逆向光刻（ILT）优化包。光学成像由上级目录的
`Hopkins/` 包提供（可微 Abbe/Hopkins 前向）；本包负责 mask 参数化、
损失函数和 Adam 优化循环。

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
```

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
变化率并可提前收敛。demo profile 用 36 个光源点优化，再用 144 个光源点独立验证；
原生 ICCAD profile 使用 72 点验证。最终输出在工作目录的 `output/`（按案例分目录）。

## 设计要点

- mask 由可训练 logits 经高斯制造滤波、设计窗边界约束和 sigmoid 生成。
- 优化期间用可微 soft resist，验证用硬阈值 resist。
- 损失包含像素/边界加权误差、TV、二值化和面积正则。
- 光源点分批传播，控制 8 GB 显存上的峰值占用。
- 保存 hard-mask 验证 IoU 最佳的状态，而不是盲目使用末轮。

## GTX 1080 demo 实测

- 1000 步上限，第 716 步收敛早停（保存第 375 步最佳 hard mask）
- 优化时间：21.25 s
- 144 光源点独立验证：0.15 s
- 峰值显存：651 MB
- hard-resist IoU：0.83043 → 0.97837
- 轮廓平均距离：8.35 nm → 0.96 nm
- 与 NumPy 参考前向的 RMS 差：1.1e-7

