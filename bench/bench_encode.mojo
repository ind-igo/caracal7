"""Level-1 encoder throughput, reference profile, COLS columns. Small by default."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from caracal7.params import REFERENCE
from caracal7.tables import Domains, TableLayout, build_tables
from caracal7.arena import Arena, Bump
from caracal7.encode import EncLayout, encode, to_packed, rs_encode

comptime p = REFERENCE
comptime COLS = 64
comptime REPS = 5


def main() raises:
    var ctx = DeviceContext()
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

    encode[p](ctx, arena.base(), e, tab)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(REPS):
        to_packed[p](ctx, arena.base(), e, tab)
    ctx.synchronize()
    var grid_ms = Float64(perf_counter_ns() - t0) / 1e6 / REPS
    t0 = perf_counter_ns()
    for _ in range(REPS):
        rs_encode[p](ctx, arena.base(), e, tab)
    ctx.synchronize()
    var rs_ms = Float64(perf_counter_ns() - t0) / 1e6 / REPS
    # RS work: pass A ceil(K/315) + (5 + 7 + 9) F4 MACs per point, 16 base MACs each
    var terms = Float64((p.N() // 4 + 314) // 315 + 21)
    var macs = Float64(p.L0) * terms * 16.0 * Float64(COLS)
    var code_bytes = Float64(p.L0 * COLS * 4)
    print("to_packed: ", grid_ms, " ms   (", grid_ms * 1000.0 / COLS, " us/column)")
    print("rs_encode: ", rs_ms, " ms   (", rs_ms * 1000.0 / COLS, " us/column, ",
          macs / (rs_ms * 1e6), " GMAC/s, ", code_bytes / (rs_ms * 1e6), " GB/s code written)")

    # per-kernel split of rs_encode
    time_mask[1](ctx, arena, e, tab, "pass A ")
    time_mask[2](ctx, arena, e, tab, "stage 5")
    time_mask[4](ctx, arena, e, tab, "stage 7")
    time_mask[8](ctx, arena, e, tab, "stage 9 (scatter to code)")


def time_mask[mask: Int](ctx: DeviceContext, arena: Arena, e: EncLayout, tab: TableLayout, name: String) raises:
    var t = perf_counter_ns()
    for _ in range(REPS):
        rs_encode[p, mask](ctx, arena.base(), e, tab)
    ctx.synchronize()
    print("  ", name, ": ", Float64(perf_counter_ns() - t) / 1e6 / REPS, " ms")
