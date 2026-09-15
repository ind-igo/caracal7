"""Level-1 encoder throughput: the reference profile at 64 columns, then the Keccak 2048 B shape at 142."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from caracal7.core.params import CLIENT, Params
from caracal7.core.tables import Domains, TableLayout, build_tables
from caracal7.core.arena import Arena, Bump
from caracal7.pcs.encode import EncLayout, encode, to_packed, rs_encode, idft2, pack

comptime REPS = 5


def main() raises:
    var ctx = DeviceContext()
    run[CLIENT.grid(72, 32), 64](ctx)
    run[CLIENT.grid(64, 384), 142](ctx)
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks (decisions.md 2026-09-16)


def run[p: Params, COLS: Int](ctx: DeviceContext) raises:
    var d = Domains.__init__[p]()
    var bump = Bump()
    var e = EncLayout.__init__[p](bump, COLS)
    var tab = TableLayout.__init__[p](bump.alloc(0))
    _ = bump.alloc(tab.bytes)
    var arena = Arena(ctx, bump.used)
    arena.upload(ctx, tab.base, build_tables[p](ctx, tab, d))
    var th = ctx.enqueue_create_host_buffer[DType.uint8](COLS * p.N())
    ctx.synchronize()
    for i in range(COLS * p.N()):
        th[i] = UInt8((i * 7919 + 13) % 127)
    arena.upload(ctx, e.trace, th)
    print(ctx.name(), " ", p, " columns=", COLS, " arena=", arena.bytes // (1 << 20), " MiB")

    encode[p](ctx, arena, e, tab)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(REPS):
        to_packed[p](ctx, arena, e, tab)
    ctx.synchronize()
    var grid_ms = Float64(perf_counter_ns() - t0) / 1e6 / REPS
    t0 = perf_counter_ns()
    for _ in range(REPS):
        rs_encode[p](ctx, arena, e, tab)
    ctx.synchronize()
    var rs_ms = Float64(perf_counter_ns() - t0) / 1e6 / REPS
    # RS work: pass A ceil(K/315) + (5 + 7 + 9) F4 MACs per point, 16 base MACs each
    var terms = Float64((p.N() // 4 + 314) // 315 + 21)
    var macs = Float64(p.L0) * terms * 16.0 * Float64(COLS)
    var code_bytes = Float64(p.L0 * COLS * 4)
    print("to_packed: ", grid_ms, " ms   (", grid_ms * 1000.0 / Float64(COLS), " us/column)")
    t0 = perf_counter_ns()
    for _ in range(REPS):
        idft2[p](ctx, arena, e.trace, e.ctmp, e.coeff, COLS, tab)
    ctx.synchronize()
    print("   idft2: ", Float64(perf_counter_ns() - t0) / 1e6 / REPS, " ms")
    t0 = perf_counter_ns()
    for _ in range(REPS):
        pack[p](ctx, arena, e)
    ctx.synchronize()
    print("   pack: ", Float64(perf_counter_ns() - t0) / 1e6 / REPS, " ms  (to_stored is the rest)")
    print("rs_encode: ", rs_ms, " ms   (", rs_ms * 1000.0 / Float64(COLS), " us/column, ",
          macs / (rs_ms * 1e6), " GMAC/s, ", code_bytes / (rs_ms * 1e6), " GB/s code written)")

    # per-kernel split of rs_encode
    time_mask[p, 1](ctx, arena, e, tab, "gather")
    time_mask[p, 16](ctx, arena, e, tab, "2-adic stages")
    time_mask[p, 2](ctx, arena, e, tab, "stage 5")
    time_mask[p, 4](ctx, arena, e, tab, "stages 7 x 9 fused (scatter to code)")


def time_mask[p: Params, mask: Int](ctx: DeviceContext, arena: Arena, e: EncLayout, tab: TableLayout, name: String) raises:
    var t = perf_counter_ns()
    for _ in range(REPS):
        rs_encode[p, mask](ctx, arena, e, tab)
    ctx.synchronize()
    print("  ", name, ": ", Float64(perf_counter_ns() - t) / 1e6 / REPS, " ms")
