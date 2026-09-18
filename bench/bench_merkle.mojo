"""Merkle tree at the level-1 size: L0 leaves, rows of columns x 4 bytes."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from core.params import CLIENT
from core.hash import Blake3
from core.arena import Arena, Bump
from pcs.merkle import merkle, tree_nodes

comptime p = CLIENT.grid(72, 32)
comptime LEAVES = p.L0
comptime ROW = 101 * 4
comptime REPS = 5


def main() raises:
    var ctx = DeviceContext()
    var bump = Bump()
    var code = bump.alloc(LEAVES * ROW)
    var tree = bump.alloc(tree_nodes(LEAVES) * 32)
    var arena = Arena(ctx, bump.used)
    merkle[p, Blake3](ctx, arena, code, ROW, LEAVES, tree)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(REPS):
        merkle[p, Blake3](ctx, arena, code, ROW, LEAVES, tree)
    ctx.synchronize()
    var ms = Float64(perf_counter_ns() - t0) / 1e6 / REPS
    print("merkle", LEAVES, "leaves x", ROW, "B:", ms, "ms,", Float64(LEAVES * ROW) / ms / 1e6, "GB/s")
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks (decisions.md 2026-09-16)
