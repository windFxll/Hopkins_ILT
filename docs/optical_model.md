# Hopkins 光学模型原理

本文面向第一次接触本项目光学部分的读者，从物理模型讲到代码实现。
读完后应当能回答：一张掩模如何变成空间像、Abbe 和 SOCS 两种算法分别
在算什么、它们为什么等价、SOCS 截断误差从哪里来。

## 1. 符号与范围

| 符号 | 含义 | 默认值 |
|---|---|---|
| `N` | 掩模网格边长（像素） | 256 |
| `dx` | 像素尺寸（nm） | 4.0 |
| `n` | FFT 网格边长（`isolated` 时 `2N`，`periodic` 时 `N`） | 256 |
| `λ` | 曝光波长（nm） | 193 |
| `NA` | 投影物镜数值孔径 | 1.35 |
| `fc = NA/λ` | 光瞳截止频率（nm⁻¹） | 0.006995 |
| `σ` | 归一化光源坐标，`σ = 光源频率 / fc` | 环形 0.6–0.9 |
| `S` | 光源采样点数 | 864 |
| `mask(x)` | 掩模像素，1 = 不透明图形 | 0/1 |
| `I(x)` | 归一化空间像，空场 = 1 | — |

本文只讨论标量、薄掩模、固定阈值光刻胶模型。矢量成像暂未启用。

## 2. 物理模型：部分相干成像与 Hopkins 方程

光刻机的光源不是理想点，而是有有限张角的面光源，因此成像是**部分相干**的。
Hopkins 方程把空间像写成掩模频谱的二次型：

$$
I(x) = \iint TCC(f', f'') \, M(f') \, M^*(f'') \,
       e^{j2\pi (f'-f'')x} \, df' df''
$$

其中

$$
TCC(f', f'') = \int S(f) \, P(f+f') \, P^*(f+f'') \, df
$$

- `M(f)`：掩模透射谱
- `P(f)`：投影物镜光瞳（低通滤波器）
- `S(f)`：光源强度分布
- `TCC`：交叉传递系数（Transmission Cross Coefficient）

直观理解：光源上每个点 `f_s` 发出一束斜入射平面波，它经过掩模后由光瞳
低通滤波，形成一个相干像；不同光源点之间不相干，强度直接相加。

## 3. 离散化约定（Abbe 与 SOCS 完全一致）

### 3.1 亮场、暗图形

代码约定 `mask = 1` 表示不透明图形（暗），`mask = 0` 表示透光背景：

$$
T(x) = 1 - mask(x)
$$

无图形处 `T = 1`，归一化后空间像 `I = 1`；图形内部 `I` 趋近 0。

### 3.2 频率网格

用 `torch.fft.fftfreq(n, d=dx)` 建自然序频率网格：DC 在索引 `[0, 0]`，
负频率在数组末尾。频率间隔 `df = 1/(n·dx)`。

### 3.3 FFT 三件套

```
M = fft2(ifftshift(T))          # 图像原点移到数组角上
E = fftshift(ifft2(M * P))      # 乘光瞳后回到以中心为原点的空间域
```

这个顺序与 `fftfreq` 的自然序严格配对，不能随意调换。

### 3.4 边界条件

| 模式 | 做法 | 适用 |
|---|---|---|
| `periodic`（默认） | 直接在 `N×N` 上做 FFT，循环卷积天然给出周期边界；跨边界图形会环绕到对面 | 周期版图（推荐） |
| `isolated` | `T` 补零到 `2N×2N`，填充值 **1.0**（清晰场），算完裁回中心 `N×N` | 孤立版图（可选） |

`periodic` 是默认模式：版图本身按周期单元处理，边缘图形有完整邻居，
不会出现单侧邻近效应造成的边缘暗带。`isolated` 用线性卷积避免相邻周期
串扰，代价是 4 倍 FFT 面积，且边缘图形只受单侧邻近效应。

## 4. 光源采样

### 4.1 环形采样

当前实现只支持环形（annular）照明。在归一化光源面 `σ ∈ [σ_in, σ_out]` 上：

$$
r_i = \sigma_{in} + (i + 0.5)\Delta\sigma, \quad
\Delta\sigma = \frac{\sigma_{out}-\sigma_{in}}{R}
$$

$$
\theta_j = \frac{2\pi j}{P}, \quad j = 0,\dots,P-1
$$

- `R = source_rings = 12`：径向环数
- `P = source_points_per_ring = 72`：每环方位点数
- 总点数 `S = R × P = 864`
- 每个点的面积权重 `w_s ∝ r_i`（面积元 `σ dσ dθ`），最后归一化到 `Σ w_s = 1`
- 方位点数取偶数，保证 `±f_s` 严格成对，成像左右对称

### 4.2 分辨率

| 量 | 数值 |
|---|---|
| 径向间隔 `Δσ` | 0.025 |
| 方位角间隔 | 5° |
| 物理频率间隔 `Δσ · fc` | 约 1.75×10⁻⁴ nm⁻¹ |
| 掩模 FFT 频率间隔 `df` | 9.77×10⁻⁴ nm⁻¹（256 网格） |

验证档使用 24 环 × 144 点 = 3456 点（`Δσ = 0.0125`，方位 2.5°），
用于独立验证优化结果。

**重要关系**：TCC 的秩不超过光源点数 `S`，也受频率子网格维度限制。
当前配置 `S=864`、子网格 27×27=729，数值秩为 576，因此 SOCS 最多 576 阶。

## 5. 光瞳与离焦

对每个光源点 `f_s` 构造移位光瞳（`pupil.py: build_shifted_pupils`）：

$$
P_s(f) =
\begin{cases}
1, & |f + f_s| \le fc \\
0, & |f + f_s| > fc
\end{cases}
$$

有离焦 `Δz` 时加相位：

$$
P_s(f) = \exp\!\left[j \frac{2\pi}{\lambda}\Delta z
\left(\sqrt{1 - (NA \cdot \nu)^2} - 1\right)\right],
\quad \nu = \frac{|f+f_s|}{fc}
$$

无离焦时光瞳严格为 0/1，代码用 `bool` 存储以节省显存。

## 6. Abbe 点源求和

### 6.1 公式

光源离散化后，Hopkins 方程严格等价于：

$$
I(x) = \sum_{s=1}^{S} w_s \left| \mathrm{IFFT}\left[M(f) \cdot P_s(f)\right](x) \right|^2
$$

每个光源点做一次相干成像（一次 iFFT），再把强度按权重相加。

### 6.2 计算流程（`abbe/imaging.py: AbbeImaging.forward`）

1. `T = 1 - mask`
2. `isolated` 时补到 `n×n`，填充 1.0
3. `M = fft2(ifftshift(T))`
4. 对每个光源点（可分批）：`E_s = fftshift(ifft2(M * P_s))`
5. `I += w_s · |E_s|²`
6. 裁回 `N×N`
7. 归一化：`I /= clear_intensity`
8. 可选高斯胶模糊

其中开域强度

$$
I_{clear} = \sum_s w_s |P_s(0)|^2 = \sum_s w_s = 1
$$

### 6.3 复杂度

每次前向 `S` 次 FFT，复杂度 `O(S · n² log n)`。默认 `S = 864`，
256² 网格（periodic），RTX 3050 上约 54 ms。

## 7. SOCS：TCC 特征分解

### 7.1 从 TCC 出发

把 TCC 在频率子网格上离散：

$$
TCC(i,j) = \sum_{s=1}^{S} w_s \, P_s(f_i) \, P_s^*(f_j)
$$

TCC 是 Hermitian 半正定矩阵。对它做特征分解：

$$
TCC = \sum_{k} \lambda_k \, \phi_k \phi_k^H
$$

得到 SOCS 前向：

$$
I(x) = \sum_{k} \lambda_k
\left| \mathrm{IFFT}\left[M \cdot \phi_k\right](x) \right|^2
$$

截断到前 `K` 个特征对，就是 **SOCS 近似**：

$$
I_K(x) = \sum_{k<K} \lambda_k
\left| \mathrm{IFFT}\left[M \cdot \phi_k\right](x) \right|^2
$$

### 7.2 频率子网格

只有可能通过光瞳的频率才需要参与 TCC：`|f_i| ≤ (1+σ_out)·fc`。

```text
R  = (1 + sigma_out) * fc
df = 1 / (n * dx)
kc = floor(R / df)                 # 默认 256@4nm periodic → kc = 13
nsub = (2*kc+1)^2 = 27 × 27 = 729
```

把 `k ∈ [-kc, kc]` 映射到完整 FFT 网格的 `k % n`。DC 在子网格中的索引
是 `kc*(2kc+1)+kc`。

### 7.3 秩上界

令 `A[s, i] = √w_s · P_s(f_i)`，则

$$
TCC = A^H A, \quad G = A A^H \in \mathbb{C}^{S \times S}
$$

`TCC` 是 `S` 个秩-1 项之和，所以 **`rank(TCC) ≤ min(S, nsub)`**。
当前 `S = 864`、`nsub = 729`，数值秩为 576；请求 `K > 576` 会被钳到 576。

### 7.4 两种特征分解路径

**低秩等价路径（默认）**

对 `S×S` 的 Gram 矩阵做特征分解：

$$
G = U \Lambda U^H, \quad
\phi_k = \frac{A^H u_k}{\sqrt{\lambda_k}}, \quad
\lambda_k = \Lambda_{kk}
$$

与直接分解 `TCC` 得到的非零特征对完全相同，但矩阵只有 864×864，
初始化约 10.2 ms（矩阵构造）+ 151.3 ms（eigh）。

**显式 TCC 路径（可选）**

真正构造 `729×729` 的 TCC，做 `torch.linalg.eigh`。
当前配置初始化约 4.6 ms + 96.1 ms；由于子网格维度小于光源点数，
显式路径在这个规模下反而略快。
两条路径的特征值相对误差为 0（浮点范围内），满秩成像 RMS 差约 7.1×10⁻⁸。

通过 `--socs-eigh-mode {low_rank,explicit,auto}` 或
`SocsImaging(..., eigh_mode=...)` 切换。

### 7.5 SOCS 前向（`socs/imaging.py: SocsImaging.forward`）

1. `T = 1 - mask`，`isolated` 时补到 `n×n`
2. `M = fft2(ifftshift(T))`
3. 取子网格 `M_sub = M[sub_y, sub_x]`
4. 对每批特征核：把 `M_sub · φ_k` 放回完整频率网格，做 iFFT
5. `I = Σ_{k<K} λ_k |E_k|²`
6. 裁回 `N×N`
7. 归一化：`I /= C_K`，其中

$$
C_K = \sum_{k<K} \lambda_k |\phi_k(0)|^2
$$

用截断后的 `C_K` 归一化，保证**任意 K 下空掩模成像恒为 1**。
满秩时 `C_K` 与 Abbe 的 `clear_intensity` 完全一致。

### 7.6 复杂度与加速

每次前向做 `K` 次 iFFT，复杂度 `O(K · n² log n)`，**与光源点数无关**。
RTX 3050、256@4nm、periodic、256² 网格上的实测：

| 方法 | 前向时间 |
|---|---|
| Abbe864 | 53.9 ms |
| Abbe3456 | 206.8 ms |
| SOCS K=8 | 约 0.75 ms |
| SOCS K=16 | 约 1.30 ms |
| SOCS K=32 | 约 2.22 ms |
| SOCS K=64 | 约 4.10 ms |
| SOCS K=128 | 约 7.70 ms |
| SOCS K=256 | 约 15.8 ms |
| SOCS K=576（满秩） | 约 33.3 ms |

## 8. Abbe 与 SOCS 的关系

### 8.1 等价性

- 物理模型相同：都是 Hopkins 部分相干成像。
- 数值上，满秩 SOCS 与 Abbe 输出一致：默认 256 配置实测 RMS 约 1.25×10⁻⁷。
- 梯度也一致：自动求导相对误差约 2×10⁻⁷。

### 8.2 误差来源

| 来源 | 说明 |
|---|---|
| 光源离散化 | 864 点 vs 3456 点参考，RMS 约 7.29×10⁻⁴ |
| SOCS 截断 | 只保留前 K 个特征对，K 越小误差越大 |

864 点 Abbe 与 3456 点参考的差异（7.29×10⁻⁴）是**光源采样误差地板**；
SOCS 的截断误差必须和它比较，而不是和零比较。

10 张版图上的实测（与 Abbe3456 参考比较）：

| K | 保留能量 | RMS 误差 | 相对 Abbe864 加速 |
|---|---|---|---|
| 8 | 89.7% | 7.95×10⁻³ | 72× |
| 16 | 93.3% | 5.98×10⁻³ | 41× |
| 32 | 95.8% | 3.70×10⁻³ | 24× |
| 64 | 97.5% | 2.15×10⁻³ | 13× |
| 128 | 98.7% | 1.05×10⁻³ | 7.0× |
| 256 | 99.6% | 7.36×10⁻⁴ | 3.4× |
| 576 | 100% | 7.29×10⁻⁴（= 地板） | 1.6× |

**采用方案**：最终代理模型采用 **K=8**（保留 89.7% 能量，前向约 4.7 ms/图）；
截断误差高于光源采样地板，但端到端 ILT 结果几乎不受影响。
若用途是离线高保真成像，可临时用 K=128/256。基准脚本与逐版图结果
不在本发布包中，历史数据保留在本地归档。

## 9. 代码结构映射

| 概念 | 文件与函数 |
|---|---|
| 光学参数 | `hopkins/config.py: OpticsConfig` |
| 光源采样 | `hopkins/source.py: annular_source_samples` |
| 移位光瞳与离焦 | `hopkins/pupil.py: build_shifted_pupils` |
| Abbe 前向 | `hopkins/abbe/imaging.py: AbbeImaging.forward` |
| TCC 子网格与特征分解 | `hopkins/socs/tcc.py: build_tcc_eigenbasis` |
| SOCS 前向 | `hopkins/socs/imaging.py: SocsImaging.forward` |
| 软/硬光刻胶 | `hopkins/resist/models.py` |
| 方法工厂 | `hopkins/factory.py: make_imaging` |
| 正确性测试 | `Hopkins/tests/socs_test.py`、`smoke_test.py` |

调用方式：

```python
import torch
from hopkins import AbbeImaging, SocsImaging, make_imaging, OpticsConfig

cfg = OpticsConfig()
device = torch.device("cuda:0")

abbe = AbbeImaging(cfg, device)
socs = SocsImaging(cfg, device, num_orders=16)   # low_rank by default
auto = make_imaging(cfg, device, method="socs", num_orders=16)

mask = torch.zeros((cfg.grid_size, cfg.grid_size), device=device)
aerial = socs(mask)          # [256, 256], float32, clear field = 1
```

ILT 通过依赖注入调用：

```python
from ilt.optimization import PixelILT
from ilt.config import OptimizationConfig

result = PixelILT(make_imaging(cfg, device, method="socs", num_orders=16),
                  OptimizationConfig()).optimize(target)
```

## 10. 参数速查与常见问题

### 10.1 关键参数（`OpticsConfig`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `grid_size` | 256 | 掩模网格边长 |
| `pixel_size_nm` | 4.0 | 像素尺寸 |
| `wavelength_nm` | 193.0 | 波长 |
| `numerical_aperture` | 1.35 | NA |
| `sigma_in/out` | 0.6/0.9 | 环形光源内外径 |
| `source_rings` | 12 | 径向环数 |
| `source_points_per_ring` | 72 | 每环点数 |
| `defocus_nm` | 0.0 | 离焦量 |
| `boundary_mode` | `periodic` | 边界条件（可选 `isolated`） |
| `resist_threshold` | 0.346585 | 胶阈值 |
| `resist_blur_sigma_nm` | 0.0 | 胶扩散 |
| `source_batch_size` | 6 | Abbe 光源分批大小 |

### 10.2 常见问题

**Q：为什么 SOCS 最多只能到 576 阶？**
A：TCC 的秩不超过光源点数和频率子网格维度。当前 864 个光源点、27×27=729
子网格，数值秩为 576。提高光源密度或增大频率子网格可以提高上限。

**Q：SOCS 的 K 与 Abbe 的光源点数是同一个东西吗？**
A：不是。Abbe 每步要做 `S` 次 iFFT；SOCS 每步做 `K` 次 iFFT。K 是 TCC
的特征模数，通常远小于 S 就能保留大部分能量。

**Q：K=16 的误差比光源采样地板大很多，正常吗？**
A：正常。截断误差和光源采样误差是两种独立误差。当前配置下 K=16 的误差
约为地板的 8.2 倍，K=32 为 5.1 倍，K=128 降到 1.43 倍，K=256 基本持平。

**Q：为什么要用子网格？**
A：完整 256×256 频率网格的 TCC 是 65536²，无法存储。只有光瞳能覆盖的
频率（27×27）才可能非零，其余频点的核恒为零。

**Q：显式 TCC 和低秩路径结果一样吗？**
A：数学上完全一样。特征值相对误差 0，满秩成像 RMS 差约 7.1×10⁻⁸。
低秩路径只是避免了构造大矩阵。

## 11. 与 ILT 的关系

ILT 优化器不关心具体成像算法，只要求对象可调用、输入输出形状一致、
并提供 `.config.resist_threshold`。因此 Abbe 与 SOCS 可以互换：

- 训练/推理：`ILT/scripts/run_demo.py --imaging {abbe,socs}`
- SOCS 截断阶数：`--socs-orders 16`
- 特征分解路径：`--socs-eigh-mode {low_rank,explicit,auto}`

ILT 的梯度推导与损失函数见根目录
[`docs/ilt_gradient_descent.md`](../../docs/ilt_gradient_descent.md)。
