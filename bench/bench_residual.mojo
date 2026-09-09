"""Residual stage throughput, reference profile: LDE, the family pass, the quotient. Synthetic
family table repeated to ENTRIES entries over COLS columns. Small by default."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from caracal7.core.params import CLIENT
from caracal7.core.tables import Domains, TableLayout, build_tables
from caracal7.core.arena import Arena, Bump
from caracal7.pcs.encode import EncLayout, to_packed
from caracal7.relations.ir import ENTRY, Families
from caracal7.workloads.synthetic import synthetic_statement
from caracal7.relations.residual import lde, residual, quotient, quotient_elems

comptime p = CLIENT.grid(72, 32)
comptime COLS = 64
comptime ENTRIES = 260          # 15 copies of the 17 synthetic entries over 8 columns each
comptime REPS = 5
comptime G = 4 * p.N()


def main() raises:
    var ctx = DeviceContext()
    var d = Domains.__init__[p]()
    var f = Families()
    var syn = synthetic_statement(with_accumulator=False).compile[p]()
    var syn_count = len(syn.families) // ENTRY
    for r in range(ENTRIES // syn_count):
        for k in range(syn_count):
            for i in range(ENTRY):
                f.bytes.append(syn.families[k * ENTRY + i])
            f.bytes[f.count * ENTRY + 16] = UInt8((Int(syn.families[k * ENTRY + 16]) + 8 * (r % (COLS // 8))) & 255)
            f.count += 1
    var bump = Bump()
    var e = EncLayout.__init__[p](bump, COLS)
    var tab = TableLayout.__init__[p](bump.alloc(0))
    _ = bump.alloc(tab.bytes)
    var families = bump.alloc(f.count * ENTRY)
    var ltmp = bump.alloc(COLS * 2 * p.h2() * 2 * p.h1() * 2)
    var lde_buf = bump.alloc(COLS * G * 2)
    var res_buf = bump.alloc(G * p.e)
    var scratch = bump.alloc(quotient_elems[p]() * p.e)
    var stored = bump.alloc(3 * p.e * p.N())
    var alpha = bump.alloc(p.e)
    var chals = bump.alloc(3 * p.e)
    var arena = Arena(ctx, bump.used)
    arena.upload(ctx, tab.base, build_tables[p](ctx, tab, d))
    var th = ctx.enqueue_create_host_buffer[DType.uint8](COLS * p.N())
    var fh = ctx.enqueue_create_host_buffer[DType.uint8](f.count * ENTRY)
    var ah = ctx.enqueue_create_host_buffer[DType.uint8](p.e)
    ctx.synchronize()
    for i in range(COLS * p.N()):
        th[i] = UInt8((i * 7919 + 13) % 127)
    for i in range(f.count * ENTRY):
        fh[i] = f.bytes[i]
    for i in range(p.e):
        ah[i] = UInt8(i + 1)
    arena.upload(ctx, e.trace, th)
    arena.upload(ctx, families, fh)
    arena.upload(ctx, alpha, ah)
    to_packed[p](ctx, arena, e, tab)
    print(ctx.name(), " ", p, " columns=", COLS, " entries=", f.count, " G=", G)

    def report(name: String, ns: Int, macs: Int):
        var us = ns // REPS // 1000
        print(name, ": ", us, " us  ", macs // (us + 1) // 1000, " GMAC/s (F2 products)")

    lde[p](ctx, arena, e.coeff, COLS, tab, ltmp, lde_buf)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(REPS):
        lde[p](ctx, arena, e.coeff, COLS, tab, ltmp, lde_buf)
    ctx.synchronize()
    report("lde", Int(perf_counter_ns() - t0), COLS * (p.N() * 2 * p.h1() + 2 * p.h1() * 2 * p.h2() * p.h2()))

    residual[p](ctx, arena, lde_buf, families, f.count, tab, alpha, chals, res_buf)
    ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(REPS):
        residual[p](ctx, arena, lde_buf, families, f.count, tab, alpha, chals, res_buf)
    ctx.synchronize()
    report("residual", Int(perf_counter_ns() - t0), f.count * G * 8)

    quotient[p](ctx, arena, res_buf, tab, scratch, stored)
    ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(REPS):
        quotient[p](ctx, arena, res_buf, tab, scratch, stored)
    ctx.synchronize()
    report("quotient", Int(perf_counter_ns() - t0), 8 * (p.h1() * 2 * p.h1() * 2 * p.h2() + p.h1() * p.h1() * 2 * p.h2()
                                                          + 2 * p.h2() * 2 * p.h2() * p.h1() + p.h1() * p.h1() * p.h2() + p.h2() * p.h2() * p.h1()))
