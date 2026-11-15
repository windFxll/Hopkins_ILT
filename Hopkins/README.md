# Hopkins

基于 PyTorch 的可微 Hopkins/Abbe 部分相干成像模型。ILT 优化通过本包完成
前向推理并反向传播梯度。

## 目录结构

```text
Hopkins/
├─ hopkins/
│  ├─ config.py        光学参数 dataclass（OpticsConfig）
│  ├─ source.py        环形光源采样（面积权重）
│  ├─ pupil.py         光瞳构建（含离焦相位）
│  ├─ imaging.py       AbbeImaging：可微 Abbe 光源点求和
│  └─ resist/          软/硬光刻胶模型、高斯滤波
├─ tests/smoke_test.py GPU 前向/梯度自检
├─ pyproject.toml
└─ requirements.txt
```

## 运行自检

```bash
cd Hopkins
python tests/smoke_test.py --device cuda:0
```

## 调用示例

```python
import torch
from hopkins import AbbeImaging, OpticsConfig

cfg = OpticsConfig()
model = AbbeImaging(cfg, torch.device("cuda:0"))
mask = torch.zeros((cfg.grid_size, cfg.grid_size), device="cuda:0")
aerial = model(mask)          # (N, N)，开域归一化为 1
```

接口约定（与原实现一致）：亮场、暗图形 mask（1 = 不透明图形），
Abbe 光源点求和实现 Hopkins 部分相干成像，可选 isolated 零填充边界。
