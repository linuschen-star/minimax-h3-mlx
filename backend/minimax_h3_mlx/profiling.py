"""Lightweight accounting and opt-in synchronized H3 region profiling."""

import time
from collections import defaultdict
from dataclasses import asdict, dataclass

import mlx.core as mx


@dataclass
class BackendMetrics:
    torch_to_mlx: int = 0
    mlx_to_torch: int = 0
    mlx_to_numpy: int = 0
    cpu_materializations: int = 0
    dtype_conversions: int = 0
    layout_conversions: int = 0
    explicit_sync: int = 0
    device_transitions: int = 0
    qkv_relayouts: int = 0

    def snapshot(self):
        return asdict(self)

    def delta(self, before):
        now = self.snapshot()
        return {key: now[key] - before[key] for key in now}


class RegionProfiler:
    """Accumulate broad GPU region timings for one deliberately profiled step.

    Each ``measure`` call evaluates its returned MLX arrays.  That makes region
    boundaries accurate but prevents fusion across them, so this is a deep,
    higher-overhead diagnostic mode only.  The normal model path never creates
    this object and retains its single outer-step evaluation boundary.
    """

    def __init__(self):
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self.explicit_sync = 0

    def measure(self, region, function):
        started = time.perf_counter()
        value = function()
        mx.eval(value)
        self.seconds[region] += time.perf_counter() - started
        self.calls[region] += 1
        self.explicit_sync += 1
        return value

    def report(self):
        return {
            name: {"seconds": self.seconds[name], "calls": self.calls[name]}
            for name in sorted(self.seconds)
        }
