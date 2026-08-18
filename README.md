# compute-tool

GPU 随机负载模拟器——在 NVIDIA GPU 上持续运行 phase-based 随机矩阵计算，模拟真实业务负载的显存占用和计算强度波动。用于 GPU 资源占用、监控告警验证、调度测试等场景。

## 工作原理

程序循环执行 6 种负载 phase，按权重随机切换，每个 phase 的持续时间、矩阵大小、计算类型均随机化，使 GPU-Util 和显存占用随时间波动而非恒定：

| Phase | 权重 | 持续时间 | 显存目标 | GPU-Util 目标 | 核心操作 |
|:---|:---|:---|:---|:---|:---|
| `heavy_compute` | 25% | 300-900s | ~90% | ~100% | 3×(40500²) fp32 matmul + 转置/reduction/小batch 变体 |
| `medium_compute` | 30% | 180-480s | ~40-55% | ~50-70% | 中等矩阵 matmul + 间歇 sleep |
| `light_compute` | 15% | 60-300s | ~5% | ~10-30% | 小矩阵 + 长 sleep |
| `idle` | 10% | 15-120s | ~0% | ~0% | 100² tiny tensor + 5s sleep |
| `memory_spike` | 10% | 30-120s | 基础+突发 | variable | 基础计算 + 突发分配 38000² tensor 到 30-42GB 再释放 |
| `mixed_precision` | 10% | 120-360s | ~65% | ~100% | fp16 matmul |

phase 间有 10-90s 随机休息（20% 概率跳过休息模拟连续高负载）。连续重复同一 phase 的权重降至 1/3，鼓励负载多样性。

实测波动范围（RTX 4090 48GB）：

- GPU-Util: 0% ↔ 100%
- 显存: 482 MiB ↔ 44318 MiB（92 倍波动）
- 功耗: 22W ↔ 450W
- 温度: 29°C ↔ 69°C

## 依赖

- Python 3.12+
- PyTorch（需支持目标 GPU 的 CUDA 版本）

## 使用

### 启动

```bash
python3 gpu_matrix_loop.py
```

后台运行（推荐）：

```bash
PYTHONUNBUFFERED=1 nohup python3 gpu_matrix_loop.py > gpu_matrix_loop.log 2>&1 &
```

### 停止

```bash
pkill -f gpu_matrix_loop
```

程序捕获 SIGINT/SIGTERM，收到信号后等当前 batch 完成再优雅退出。

### 监控

```bash
# 实时看日志
tail -f gpu_matrix_loop.log

# 看 GPU 状态
nvidia-smi -l 5
```

## 参数调优

程序顶部常量按目标 GPU 显存调整：

```python
MAX_ALLOC_GB = 44.0   # 显存安全上限（留 ~3GB 余量防 OOM）

PHASES = {
    "heavy_compute":   {"weight": 25, "duration": (300, 900)},
    # ... weight 控制出现频率，duration 控制持续时间范围
}
```

`heavy_compute` 里的 `N=40500` / `NUM_PAIRS=3` 是针对 48GB 显存调的。显存更小的卡需调小 N，否则 OOM。估算公式：`(2*NUM_PAIRS + 1) * N² * 4 bytes`（fp32）。

## 已知环境问题

某些容器存在 CUDA driver mismatch（error 803）——容器内 cuda-compat 包的 libcuda 优先级高于真实宿主驱动，导致 torch 加载到错误驱动。解决用 `LD_PRELOAD` 强制加载真实驱动：

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.<真实版本> python3 gpu_matrix_loop.py
```

配合匹配驱动版本的 torch（如驱动 595.58 / CUDA 13.x → torch cu130）。

## License

MIT
