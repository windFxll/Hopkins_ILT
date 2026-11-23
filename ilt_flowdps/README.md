# SOCS / Proxy ILT and Flow benchmarks

独立的 ILT 实验包，不依赖 CT 代码。使用同一个目标版图，对比 SOCS / Proxy 下的
TV、小波、拉普拉斯 Adam 优化，以及实验性的 FlowDPS / FlowLPS / FlowSGS。
全部方法输出的二值 mask 都交给同一个 Proxy 重新前向，生成统一指标与可视化。

## 安装与快速运行

在仓库根目录执行，使用已有的 Python >=3.10 / PyTorch 环境：

```bash
python -m pip install -e ./Hopkins -e ./Hopkins_surrogate -e ./ILT -e ./ilt_flowdps

# 不需要 RF 权重；5000 步上限，默认启用早停。
bash ilt_flowdps/run_benchmark.sh --methods socs-tv proxy-tv \
  --device cuda:0 --max-iterations 5000 --progress-every 10

# 固定运行 5000 步 Proxy-TV，不早停，保存终端日志。
bash ilt_flowdps/run_proxy_tv.sh

# Flow 方法必须提供兼容的 ILT RF 权重；省略 --methods 默认运行全部 12 组。
bash ilt_flowdps/run_benchmark.sh --rf-checkpoint /path/to/rf_latest.pt \
  --device cuda:0 --flow-steps 250 --socs-inner-steps 1 --proxy-inner-steps 1 \
  --progress-every 10
```

CLI 自动选择 `cuda:0` 或 CPU；脚本示例显式指定设备。RF 权重、数据集和运行结果
不包含在本仓库中。普通优化不加载 RF，只有选择 Flow 方法才需要其权重。
默认目标为仓库自带 `ILT/benchmarks/ICCAD2013/M1_test3.glp`：按原像素间距栅格化，
将非空几何居中到 800×800，不缩放线宽。其完整 1024 nm 视场的空白边缘不保留。

数据集输入可选，路径必须显式指定：

```bash
bash ilt_flowdps/run_benchmark.sh --methods proxy-tv --layout dataset \
  --dataset-root /path/to/dataset/layout --dataset-mask layout_00001
```

目录需包含 `png/layout_00001.png` 和 `json/layout_00001.json`。
JSON 的 `png` 对象需有 `size_px: [800, 800]`、`px_per_nm: 1`、
`foreground: "black"`、`background: "white"`。PNG 必须是匹配网格的二值图。

## 方法与默认超参数

| 方法名（分别加 `socs-` / `proxy-`） | 更新 | 默认值 |
|---|---|---|
| `tv` | Adam + 各向异性 TV | 权重 0.01 |
| `wavelet` | Adam + Haar 细节 L1 | 权重 0.01，3 层 |
| `laplacian` | Adam + 周期五点拉普拉斯平方 | 权重 0.01 |
| `flowdps` | RF Euler + clean endpoint 光学梯度近似 | 250 步，每步 1 次光学更新 |
| `flowlps` | RF + Langevin + proximal + pCN | 250 步，每步 1 次 Langevin、4 次 proximal |
| `flowsgs` | Split Gibbs：光学 Langevin / RF prior SDE | 40 轮，每轮 8 / 4 步 |

`socs-ilt` / `proxy-ilt` 是 TV 的兼容别名，同一方法不可重复选择。
无惩罚项的 Adam 基线使用 `--methods socs-tv proxy-tv --tv-weight 0 --binary-weight 0 --area-weight 0`；
此时仍使用相同的滤波参数化。

Adam 使用 `ILT` 模块的损失与参数化：滤波 logits（sigma=10 px）、边界权重 3、
边界半径 8 px、cosine LR 0.08→0.002、mask beta 1.8→10、resist beta 8→40。
前三成步数完成 beta 退火。公共惩罚为 binary=0.001、area=0.08；三种空间惩罚互相替代。
TV 使用仓库原有非周期差分；小波采用有限图像 Haar 分解；拉普拉斯使用周期差分。
早停需经过 beta 退火且至少 350 步，并满足连续 3 次检测的 loss/IoU/mask-change 平台条件；
检测窗口为最近 10 条日志，默认每 10 步记录。返回优化算子下硬二值重前向 IoU 最佳的已检测 mask。
`--no-early-stop` 可强制跑满步数。

`--tv-weight/--binary-weight/--area-weight` 等 CLI 惩罚接口只影响 Adam 方法。
FlowDPS 保留原实验中的 TV=0.01、binary=0.001、area=0.08，使用边界加权 soft-resist MSE；
LPS / SGS 使用边界加权 Bernoulli 版图损失（除以 resist beta），不添加这些手工惩罚。
Flow 方法走完整采样日程，不使用 Adam 的早停条件。

SOCS 是同一个 Proxy 实例内的 K=8 base，光学配置、基底、标定与 Proxy 相同，
只移除学习到的核修正及 CNN。它不是全阶 Abbe 参考解。
此 checkpoint 固定 800×800、1 nm/px、periodic，mask 1=opaque feature；
光刻阈值 0.346585，强度标定 0.869140625。更换光学配置通常需要匹配的 Proxy 权重。

### Flow 假设与适用范围

所有 RF 调用均在 `torch.no_grad()` 下，只对光学目标及当前 clean mask 求梯度。
FlowDPS 使用 `clean=x+(1-t)*v` 和 `noise=x-t*v`，在 clean endpoint 做近似梯度引导，
时间权重 `(1-t)/max(t,0.1)^2`、scale=1、gradient RMS clip=1。
它忽略 RF Jacobian；没有显式构建或学习流形投影矩阵，不能视为精确的 `J=(1/a) P_M`。
这保留了原有近似实现，不将其解释为精确后验采样。

FlowLPS 的固定参数：Langevin LR=0.002、proximal LR=0.2、anchor=0.25、
prior std floor=0.1。FlowSGS：likelihood LR=0.02、rho=1→max(0.12, 0.9^k)、diffusion=1。
它们是有界像素坐标、非线性光刻似然下的有限步实验适配，包含投影/离散化，
不主张继承论文的精确采样或理论保证。参考 [FlowLPS](https://arxiv.org/abs/2512.07150)
和 [FlowSGS](https://arxiv.org/abs/2609.20769)。

RF loader 仅支持本包的 `film5_bottleneck_mhsa32_v2`、单通道无条件
`objective="rectified_flow"` checkpoint；检查 `base`、`image_size`、架构及 state dict，
优先 EMA。时间方向为 noise→data（t=0→1）。默认假设训练图为 signed [-1,1]、
feature-high；需根据实际训练设置选择 `--rf-encoding` 和 `--rf-polarity`。
版图先验与可制造 OPC mask 的分布可能不同；本实现不保证 Flow 优于传统 ILT。

### Proxy fingerprint

默认严格检查 checkpoint 的 SOCS basis fingerprint。若重建基底不一致，程序会终止。
诊断时可显式使用 `--allow-proxy-basis-mismatch`（PowerShell 为 `-AllowProxyBasisMismatch`）。
它只绕过检查，不修复基底；学习的核修正依赖该基底，因此绕过后的指标需单独验证，
不能当作已经验证过的正式模型结果。严格加载状态会写入 `run_config.json`。

## PowerShell 完整参数示例

先激活自己的 Python 环境，或显式指定 `-Python`。以下列出该 PS 脚本的全部可调参数，
包含两个布尔开关；实际运行时换成真实权重路径和全新的输出目录。
`ProxyCheckpoint ''` 会选择原仓库内置权重，`DatasetRoot` 只在 dataset 模式使用。

```powershell
.\ilt_flowdps\run_benchmark.ps1 `
  -Python 'python' `
  -RfCheckpoint 'D:\models\rf_latest.pt' `
  -ProxyCheckpoint '' `
  -Device 'cuda:0' `
  -Layout 'iccad2013-m1-test3' `
  -DatasetRoot '' `
  -DatasetMask 'layout_00001' `
  -Methods @('socs-tv','proxy-tv','socs-wavelet','proxy-wavelet','socs-laplacian','proxy-laplacian','socs-flowdps','proxy-flowdps','socs-flowlps','proxy-flowlps','socs-flowsgs','proxy-flowsgs') `
  -MaxIterations 5000 `
  -NoEarlyStop:$false `
  -TvWeight 0.01 `
  -WaveletWeight 0.01 `
  -WaveletLevels 3 `
  -LaplacianWeight 0.01 `
  -BinaryWeight 0.001 `
  -AreaWeight 0.08 `
  -FlowSteps 250 `
  -SocsInnerSteps 1 `
  -ProxyInnerSteps 1 `
  -DpsScale 1.0 `
  -MaxGuidanceRms 1.0 `
  -FlowMinCleanWeight 0.10 `
  -LpsLangevinSteps 1 `
  -LpsProximalSteps 4 `
  -LpsProximalLr 0.2 `
  -SgsSteps 40 `
  -SgsLikelihoodSteps 8 `
  -SgsPriorSteps 4 `
  -ReportEvery 10 `
  -LogEvery 10 `
  -Seed 2026 `
  -OutputDir '.\ilt_flowdps\runs\comparison' `
  -RfEncoding 'signed' `
  -RfPolarity 'feature-high' `
  -AllowProxyBasisMismatch:$false
```

其他 CLI 选项见 `python -m ilt_flowdps.benchmark.cli --help`。
Linux 固定 Proxy-TV 脚本的环境变量：`PYTHON=python`、`ILT_DEVICE=cuda:0`、
`ILT_OUTPUT_DIR`（默认时间戳目录）、`ILT_TV_WEIGHT=0.01`、`ILT_BINARY_WEIGHT=0.001`、
`ILT_AREA_WEIGHT=0.08`、`ILT_REPORT_EVERY=10`、`ILT_ALLOW_BASIS_MISMATCH=0`。
它使用已激活的环境，不自动 source 任何 CT 激活脚本。

## 输出与计时

终端输出 START、STEP、END；无 `--progress-every` 时用 tqdm。
`run_config.json` 保存配置、权重路径、目标来源、随机种子、GPU 与 PyTorch 版本。
各方法先做一次短 warmup，再同步 CUDA 后计时；加载、warmup 和最终统一评价不计入方法耗时，
优化中的进度/日志及硬 mask 验证计入耗时。比较时应保持日志频率相同。
`summary.csv` 每完成一组更新一次；每组目录有 mask、Proxy 重光刻版图、aerial、
错分图（红=多出、蓝=缺失）、history 与 metrics。最后输出 `comparison.png`。
指标包含 feature/clear IoU、平均对称轮廓误差（nm）、错分像素、mask 连通域数与面积。
同时记录 RF NFE 和光学梯度次数，避免直接比较含义不同的 step。

## 目录与验证

```text
ilt_flowdps/
  src/ilt_flowdps/
    benchmark/   # CLI、目标读取、报告
    physics/     # 光学适配、目标函数、正则、Adam
    prior/       # 独立 U-Net 与 RF 权重加载
    sampling/    # FlowDPS、LPS、SGS、共用光学梯度工具
  tests/         # CPU 算法回归，不需要外部权重
  run_benchmark.ps1
  run_benchmark.sh
  run_proxy_tv.sh
```

```bash
python -m unittest discover -s ilt_flowdps/tests -v
```

测试验证正则梯度、Adam TV 与原 PixelILT 的一致性、早停/固定步数、采样次数及
不经过 RF 反传。真实 checkpoint 的短 GPU smoke 只能验证运行路径，不能代表算法收敛。
