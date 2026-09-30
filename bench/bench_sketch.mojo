"""Sketched level-1 code (pcs/sketch.mojo) against the RS encoder, on the W tree shapes of sha256, RSA-2048 and
the passport (docs/profile.md). Both read the same packed message: the sketched code's first K rows are
`e.packed`, so to_packed is common and timed once. Per H shape (column weight, row weight, coefficients): the sketch
kernel, the RS code of the sketch, the Merkle tree over K + L_s leaves (H over F or over F4), against rs_encode and the tree over
L leaves. The two paths alternate within every repetition (M1 noise: A/B back to back)."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from core.params import CLIENT, Params
from core.tables import Domains, TableLayout, build_tables
from core.arena import Arena, Bump
from core.hash import Blake3
from pcs.encode import EncLayout, to_packed, rs_encode
from pcs.merkle import merkle, tree_nodes
from pcs.sketch import SketchMatrix, SketchLayout, upload_sketch, sketch_code, sketch

comptime REPS = 5
comptime SHAPES = [(16, 128, 1), (8, 64, 1), (16, 64, 1), (24, 192, 1), (16, 128, 4), (24, 192, 4)]   # (c, d, coefficient width)


def main() raises:
    var ctx = DeviceContext()
    run[CLIENT.grid(32, 2688), 43](ctx, "sha256")
    run[CLIENT.grid(144, 2016), 172](ctx, "rsa-2048")
    run[CLIENT.grid(144, 4032), 285](ctx, "passport")
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks


def _ms(t0: Int) -> Float64:
    return Float64(Int(perf_counter_ns()) - t0) / 1e6


def run[p: Params, COLS: Int](ctx: DeviceContext, name: String) raises:
    if p.n_cw() != 1:
        raise Error("split codewords: the sketched code takes one codeword per column")
    var d = Domains.__init__[p]()
    var bump = Bump()
    var e = EncLayout.__init__[p](bump, COLS)
    var tree = bump.alloc(tree_nodes(p.L()) * Blake3.DIGEST)
    var hs = List[SketchMatrix]()
    var ss = List[SketchLayout]()
    var trees = List[Int]()
    for sh in materialize[SHAPES]():
        hs.append(SketchMatrix(p.K(), sh[0], sh[1], 1, sh[2]))
        ss.append(SketchLayout(bump, hs[len(hs) - 1], COLS))
        trees.append(bump.alloc(tree_nodes(ss[len(ss) - 1].leaves()) * Blake3.DIGEST))
    var tab = TableLayout.__init__[p](bump.alloc(0))
    _ = bump.alloc(tab.bytes)
    var arena = Arena(ctx, bump.used)
    arena.upload(ctx, tab.base, build_tables[p](ctx, tab, d))
    for i in range(len(ss)):
        upload_sketch(ctx, arena, ss[i], hs[i])
    var th = ctx.enqueue_create_host_buffer[DType.uint8](COLS * p.N())
    ctx.synchronize()
    for i in range(COLS * p.N()):
        th[i] = UInt8((i * 7919 + 13) % 127)
    arena.upload(ctx, e.trace, th)
    print(ctx.name(), " ", name, " ", p, " columns=", COLS, " K=", p.K(), " L=", p.L(),
          " arena=", arena.bytes // (1 << 20), " MiB")

    var packed = e.packed
    to_packed[p](ctx, arena, e, tab)                 # warm
    ctx.synchronize()
    var t0 = Int(perf_counter_ns())
    for _ in range(REPS):
        to_packed[p](ctx, arena, e, tab)
    ctx.synchronize()
    print("  to_packed (both): ", _ms(t0) / REPS, " ms")

    var rs = 0.0
    var mk = 0.0
    var sk = List[Float64](length=len(ss), fill=0.0)
    var sr = List[Float64](length=len(ss), fill=0.0)
    var sm = List[Float64](length=len(ss), fill=0.0)
    for rep in range(REPS + 1):                      # rep 0 warms
        var w = 0.0 if rep == 0 else 1.0
        e.packed = packed
        ctx.synchronize()
        t0 = Int(perf_counter_ns())
        rs_encode[p](ctx, arena, e, tab)
        ctx.synchronize()
        rs += w * _ms(t0)
        t0 = Int(perf_counter_ns())
        merkle[p, Blake3](ctx, arena, e.code, COLS * 4, p.L(), tree)
        ctx.synchronize()
        mk += w * _ms(t0)
        for i in range(len(ss)):
            var s = ss[i]
            e.packed = s.code                        # the message rows of the sketched code
            to_packed[p](ctx, arena, e, tab)
            ctx.synchronize()
            t0 = Int(perf_counter_ns())
            sketch(ctx, arena, s)
            ctx.synchronize()
            sk[i] += w * _ms(t0)
            t0 = Int(perf_counter_ns())
            sketch_code(ctx, arena, s)
            ctx.synchronize()
            sr[i] += w * (_ms(t0))
            t0 = Int(perf_counter_ns())
            merkle[p, Blake3](ctx, arena, s.code, s.row_bytes(), s.leaves(), trees[i])
            ctx.synchronize()
            sm[i] += w * _ms(t0)
    print("  rs_encode ", rs / REPS, " ms   merkle over ", p.L(), " leaves ", mk / REPS, " ms   sum ", (rs + mk) / REPS)
    var i = 0
    for sh in materialize[SHAPES]():
        var s = ss[i]
        print("  H ", sh[0], "/", sh[1], " over F" if sh[2] == 1 else " over F4", " rows=", s.rows, " L_s=", s.m * s.L0, " (", s.m, " cosets)",
              "   sketch ", sk[i] / REPS, " ms   sketch + RS ", sr[i] / REPS, " ms   merkle over ", s.leaves(),
              " leaves ", sm[i] / REPS, " ms   sum ", (sr[i] + sm[i]) / REPS)
        i += 1
