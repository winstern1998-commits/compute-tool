#!/usr/bin/env python3
"""GPU 矩阵循环计算 —— phase-based 随机负载，模拟真实业务特征。

显存占用和 GPU 计算强度随时间波动，不是恒定负载：
  heavy_compute   大矩阵 matmul,  ~90% 显存, GPU-Util ~100%
  medium_compute  中等矩阵,       ~40-55% 显存, GPU-Util ~50-70%
  light_compute   小矩阵间歇计算, ~5% 显存, GPU-Util ~10-30%
  idle            几乎不计算,     ~0% 显存, GPU-Util ~0%
  memory_spike    基础计算 + 突发大内存分配/释放
  mixed_precision float16 matmul, 高 util, ~65% 显存
"""
import gc
import torch
import time
import signal
import random

DEVICE = 'cuda'

# 安全上限：留 ~3 GiB 余量防 OOM（总 47.37 GiB）
MAX_ALLOC_GB = 44.0

# Phase 配置：权重 + 持续时间范围
PHASES = {
    "heavy_compute":   {"weight": 25, "duration": (300, 900)},
    "medium_compute":  {"weight": 30, "duration": (180, 480)},
    "light_compute":   {"weight": 15, "duration": (60, 300)},
    "idle":            {"weight": 10, "duration": (15, 120)},
    "memory_spike":    {"weight": 10, "duration": (30, 120)},
    "mixed_precision": {"weight": 10, "duration": (120, 360)},
}

running = True


def handle_signal(sig, frame):
    global running
    running = False
    print(f"\n[{time.strftime('%H:%M:%S')}] Received signal {sig}, stopping...",
          flush=True)


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


# ---- utils ----

def mem_gb():
    return torch.cuda.memory_allocated() / 1024**3


def total_mem_gb():
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def pct():
    return mem_gb() / total_mem_gb() * 100


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def select_phase(last_phase):
    """加权随机选择 phase，降低连续重复概率。"""
    names = list(PHASES.keys())
    weights = [PHASES[p]["weight"] for p in names]
    if last_phase in names:
        idx = names.index(last_phase)
        weights[idx] = max(1, weights[idx] // 3)
    return random.choices(names, weights=weights, k=1)[0]


def random_duration(phase):
    lo, hi = PHASES[phase]["duration"]
    return random.randint(lo, hi)


def check_alloc_ok(needed_gb):
    """检查分配 needed_gb 是否安全（留 0.5 GiB 余量）。"""
    return (MAX_ALLOC_GB - mem_gb()) > (needed_gb + 0.5)


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


# ---- phase implementations ----

def phase_heavy_compute(duration):
    """大矩阵 matmul，~90% 显存，GPU-Util ~100%。"""
    N = 40500
    NUM_PAIRS = 3
    pairs = []
    for i in range(NUM_PAIRS):
        A = torch.randn(N, N, device=DEVICE, dtype=torch.float32)
        B = torch.randn(N, N, device=DEVICE, dtype=torch.float32)
        pairs.append((A, B))
    C = torch.empty(N, N, device=DEVICE, dtype=torch.float32)
    torch.cuda.synchronize()
    log(f"  heavy_compute: {NUM_PAIRS}x N={N}, "
        f"mem={mem_gb():.2f}GB ({pct():.1f}%)")

    end = time.time() + duration
    start = time.time()
    batch = 0
    last_report = time.time()
    while time.time() < end and running:
        r = random.random()
        if r < 0.80:
            for A, B in pairs:
                torch.matmul(A, B, out=C)
        elif r < 0.90:
            # 转置变体（A.t() 是 view，不额外分配）
            A, B = pairs[0]
            torch.matmul(A.t(), B, out=C)
        elif r < 0.95:
            # reduction（返回标量，几乎不分配）
            A, _ = pairs[0]
            _ = A.sum()
        else:
            # 小 batch matmul（切片，仅 ~16MB）
            A, B = pairs[0]
            _ = A[:2000, :2000] @ B[:2000, :2000]
        torch.cuda.synchronize()
        batch += 1
        if time.time() - last_report > 60:
            log(f"    heavy batch={batch}, mem={mem_gb():.2f}GB, "
                f"elapsed={time.time()-start:.0f}s")
            last_report = time.time()
    log(f"  heavy done: {batch} batches in {time.time()-start:.0f}s, "
        f"mem={mem_gb():.2f}GB")
    del C
    del pairs
    cleanup()


def phase_medium_compute(duration):
    """中等矩阵，~40-55% 显存，GPU-Util ~50-70%（matmul 间插入短 sleep）。"""
    N = random.choice([25000, 28000, 30000])
    NUM_PAIRS = random.choice([3, 4])
    pairs = []
    for i in range(NUM_PAIRS):
        A = torch.randn(N, N, device=DEVICE, dtype=torch.float32)
        B = torch.randn(N, N, device=DEVICE, dtype=torch.float32)
        pairs.append((A, B))
    C = torch.empty(N, N, device=DEVICE, dtype=torch.float32)
    torch.cuda.synchronize()
    log(f"  medium_compute: {NUM_PAIRS}x N={N}, "
        f"mem={mem_gb():.2f}GB ({pct():.1f}%)")

    end = time.time() + duration
    start = time.time()
    batch = 0
    last_report = time.time()
    while time.time() < end and running:
        r = random.random()
        if r < 0.75:
            for A, B in pairs:
                torch.matmul(A, B, out=C)
        elif r < 0.85:
            # B 转置变体
            A, B = pairs[0]
            torch.matmul(A, B.t(), out=C)
        elif r < 0.92:
            # element-wise in-place on C（不额外分配）
            C.mul_(1.5).add_(0.5)
        else:
            # reduction
            A, _ = pairs[0]
            _ = A.mean()
        torch.cuda.synchronize()
        batch += 1
        time.sleep(random.uniform(0.3, 0.8))  # 降低 util 到 ~50-70%
        if time.time() - last_report > 60:
            log(f"    medium batch={batch}, mem={mem_gb():.2f}GB, "
                f"elapsed={time.time()-start:.0f}s")
            last_report = time.time()
    log(f"  medium done: {batch} batches in {time.time()-start:.0f}s, "
        f"mem={mem_gb():.2f}GB")
    del C
    del pairs
    cleanup()


def phase_light_compute(duration):
    """小矩阵间歇计算，~5% 显存，GPU-Util ~10-30%。"""
    N = random.choice([3000, 5000, 8000])
    A = torch.randn(N, N, device=DEVICE, dtype=torch.float32)
    B = torch.randn(N, N, device=DEVICE, dtype=torch.float32)
    C = torch.empty(N, N, device=DEVICE, dtype=torch.float32)
    torch.cuda.synchronize()
    log(f"  light_compute: N={N}, mem={mem_gb():.4f}GB ({pct():.2f}%)")

    end = time.time() + duration
    start = time.time()
    batch = 0
    while time.time() < end and running:
        r = random.random()
        if r < 0.6:
            torch.matmul(A, B, out=C)
        elif r < 0.8:
            C.copy_(A)
        else:
            _ = A.sum()
        torch.cuda.synchronize()
        batch += 1
        time.sleep(random.uniform(2.0, 5.0))  # 低 util
    log(f"  light done: {batch} batches in {time.time()-start:.0f}s, "
        f"mem={mem_gb():.4f}GB")
    del A, B, C
    cleanup()


def phase_idle(duration):
    """几乎不计算，释放显存，GPU-Util ~0%。"""
    tiny = torch.randn(100, 100, device=DEVICE, dtype=torch.float32)
    torch.cuda.synchronize()
    log(f"  idle: mem={mem_gb():.6f}GB ({pct():.4f}%)")

    end = time.time() + duration
    tick = 0
    while time.time() < end and running:
        time.sleep(5)
        tick += 1
        if tick % 6 == 0:  # 每 ~30s 做一次微小计算
            _ = tiny.sum()
            torch.cuda.synchronize()
            log(f"    idle tick={tick}, mem={mem_gb():.6f}GB")
    log(f"  idle done: {tick} ticks ({tick*5}s)")
    del tiny
    cleanup()


def phase_memory_spike(duration):
    """基础计算 + 突发大内存分配/释放，显存剧烈波动。"""
    N = random.choice([15000, 18000, 20000])
    A = torch.randn(N, N, device=DEVICE, dtype=torch.float32)
    B = torch.randn(N, N, device=DEVICE, dtype=torch.float32)
    C = torch.empty(N, N, device=DEVICE, dtype=torch.float32)
    torch.cuda.synchronize()
    log(f"  memory_spike: base N={N}, mem={mem_gb():.2f}GB ({pct():.1f}%)")

    end = time.time() + duration
    start = time.time()
    spike_count = 0
    temp_N = 38000  # 每个 temp tensor ~5.4 GiB
    temp_unit_gb = temp_N * temp_N * 4 / 1024**3
    while time.time() < end and running:
        # 基础 matmul
        torch.matmul(A, B, out=C)
        torch.cuda.synchronize()

        # spike：分配多个大 tensor 直至随机目标（30~42 GiB）
        target_total = random.uniform(30, MAX_ALLOC_GB - 2)
        temps = []
        while check_alloc_ok(temp_unit_gb) and mem_gb() < target_total and running:
            try:
                t = torch.randn(temp_N, temp_N, device=DEVICE, dtype=torch.float32)
                temps.append(t)
                torch.cuda.synchronize()
            except torch.OutOfMemoryError:
                break
        if temps:
            log(f"    spike {spike_count+1}: +{len(temps)} tensors "
                f"({len(temps)*temp_unit_gb:.1f}GB), "
                f"mem={mem_gb():.2f}GB ({pct():.1f}%)")
            # 保持 2-5s 让 nvidia-smi 能采样到
            hold_end = time.time() + random.uniform(2, 5)
            while time.time() < hold_end and running:
                for t in temps:
                    _ = t.sum()
                torch.cuda.synchronize()
                time.sleep(0.3)
        else:
            log(f"    spike {spike_count+1}: no alloc "
                f"(mem={mem_gb():.2f}GB, target={target_total:.1f}GB)")
        # 释放 spike tensors
        for t in temps:
            del t
        del temps
        cleanup()
        spike_count += 1
        time.sleep(random.uniform(1, 3))
    log(f"  memory_spike done: {spike_count} spikes in "
        f"{time.time()-start:.0f}s, mem={mem_gb():.2f}GB")
    del A, B, C
    cleanup()


def phase_mixed_precision(duration):
    """float16 matmul，高 util，~65% 显存。"""
    N = random.choice([45000, 50000, 55000])
    NUM_PAIRS = random.choice([2, 3])
    pairs = []
    for i in range(NUM_PAIRS):
        A = torch.randn(N, N, device=DEVICE, dtype=torch.float16)
        B = torch.randn(N, N, device=DEVICE, dtype=torch.float16)
        pairs.append((A, B))
    C = torch.empty(N, N, device=DEVICE, dtype=torch.float16)
    torch.cuda.synchronize()
    log(f"  mixed_precision: {NUM_PAIRS}x N={N} fp16, "
        f"mem={mem_gb():.2f}GB ({pct():.1f}%)")

    end = time.time() + duration
    start = time.time()
    batch = 0
    last_report = time.time()
    while time.time() < end and running:
        r = random.random()
        if r < 0.85:
            for A, B in pairs:
                torch.matmul(A, B, out=C)
        elif r < 0.93:
            A, B = pairs[0]
            torch.matmul(A.t(), B, out=C)
        else:
            A, _ = pairs[0]
            _ = A.sum()
        torch.cuda.synchronize()
        batch += 1
        if time.time() - last_report > 60:
            log(f"    mixed batch={batch}, mem={mem_gb():.2f}GB, "
                f"elapsed={time.time()-start:.0f}s")
            last_report = time.time()
    log(f"  mixed_precision done: {batch} batches in "
        f"{time.time()-start:.0f}s, mem={mem_gb():.2f}GB")
    del C
    del pairs
    cleanup()


PHASE_FUNCS = {
    "heavy_compute": phase_heavy_compute,
    "medium_compute": phase_medium_compute,
    "light_compute": phase_light_compute,
    "idle": phase_idle,
    "memory_spike": phase_memory_spike,
    "mixed_precision": phase_mixed_precision,
}


def main():
    log("=== GPU Random Workload Simulator ===")
    log(f"  Device: {torch.cuda.get_device_name(0)}")
    total = total_mem_gb()
    log(f"  Total mem: {total:.2f} GB, max alloc target: {MAX_ALLOC_GB} GB "
        f"(margin {total - MAX_ALLOC_GB:.2f} GB)")
    log(f"  Phases: {', '.join(PHASES.keys())}")

    iteration = 0
    last_phase = None
    while running:
        phase = select_phase(last_phase)
        duration = random_duration(phase)
        log(f"\n--- Loop {iteration+1}: phase={phase}, duration={duration}s ---")

        try:
            PHASE_FUNCS[phase](duration)
        except torch.OutOfMemoryError as e:
            log(f"  OOM in {phase}: {str(e)[:120]}")
            cleanup()
        except Exception as e:
            log(f"  Error in {phase}: {e}")
            cleanup()

        last_phase = phase
        iteration += 1

        if not running:
            break

        # 20% 概率跳过休息（模拟连续高负载）
        if random.random() < 0.20:
            log(f"  (skip rest, direct to next phase)")
        else:
            sleep_dur = random.randint(10, 90)
            log(f"  resting {sleep_dur}s...")
            for _ in range(sleep_dur):
                if not running:
                    break
                time.sleep(1)

    log(f"\n=== Terminated. Total loops: {iteration} ===")


if __name__ == "__main__":
    main()
