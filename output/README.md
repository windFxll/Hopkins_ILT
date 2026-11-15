# 实验结果输出

每次运行 `ILT/scripts/run_demo.py` 会在本目录下按案例建子目录（如 `demo/`、
`complex/`、`<target>_<profile>/`），不会覆盖其他案例。

## demo 案例（realistic 版图，1000 步上限）

| 文件 | 内容 |
|---|---|
| `01_target.png` | 目标版图（二值，白 = 图形） |
| `02_initial_printed.png` | 优化前硬光刻胶印刷结果 |
| `03_optimized_mask.png` | Adam 优化后的掩模 |
| `04_optimized_aerial.png` | 优化后空间像（灰度，144 光源点独立验证） |
| `05_optimized_printed.png` | 优化后硬光刻胶印刷结果 |
| `optimized_mask.npy` | 优化掩模数组（float32，0/1） |
| `optimized_mask.txt` | 同上，纯文本 0/1 矩阵 |
| `history.csv` | 逐步优化记录：loss、IoU、mask/resist beta、学习率、梯度范数等 |
| `summary.png` | 汇总图：目标 / 掩模 / 空间像 / 印刷结果 + loss 与 IoU 曲线 |
| `report.txt` | 最终指标：耗时、显存、IoU、轮廓平均距离、早停信息 |

## 本次运行结果（RTX 3050 Laptop GPU）

```text
best_step=375
steps_completed=716
stop_reason=early convergence (fixed-beta plateau)
optimization_seconds=22.592
peak_gpu_memory_mb=123.5
initial_iou=0.830432
optimized_iou=0.978367
initial_contour_mean_nm=8.3487
optimized_contour_mean_nm=0.9584
```
