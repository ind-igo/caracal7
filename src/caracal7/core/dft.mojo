"""The grid DFTs as three radix stages per axis, one elementwise kernel each (docs/decisions.md).
A dense DFT of length n costs n multiply-adds per output; on the chain axis (2 h2 = 1792 for ECDSA)
that made the LDE quadratic in the chain count. With n = n1 n2 n3 (n1 the odd part, n2 <= n3 the
two halves of the power of two) the cost is n1 + n2 + n3 per output, three passes over memory.
Index split: input k = k1 + n1 k2 + n1 n2 k3, output j = jj + n2 n3 j1 with jj = j3 + n3 j2.
    stage 3   Y3[j3][k1 + n1 k2] = sum_k3 T3[j3, k3] x[k]                      T3 = s root^(n1 n2 j3 k3)
    stage 2   Y2[jj][k1]         = sum_k2 T2[j3][j2, k2] Y3[j3][k1 + n1 k2]    T2 = root^(n1 jj k2)
    stage 1   X[j]               = sum_k1 T1[jj][j1, k1] Y2[jj][k1]            T1 = root^(j k1)
The twists of Cooley-Tukey are folded into the per-prefix tables T2 and T1. A stage is one thread
per (line, prefix, inner) reading its kin inputs along the digit and writing r outputs; radices are
at most 32, so the tiled GEMM skeleton has nothing to amortize here.
Axis 2 transforms the rows of a column (W F2 per row); axis 1 transforms each row (W = 1, the
lines are the rows of every column). Every n-row buffer has stride n W 2 per line."""
from std.math import ceildiv
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.gpu import global_idx
from max.gpu.host import DeviceContext

from caracal7.core.params import Params
from caracal7.core.field import F2, f_reduce_signed
from caracal7.core.backend import BACKEND
from caracal7.core.bytes import Base
from caracal7.core.arena import Arena


@fieldwise_init
struct DftPlan(TrivialRegisterPassable):
    """n = n1 n2 n3 and the nonzero inputs k3 of stage 3 (k_in = n1 n2 k3)."""
    var n: Int
    var n1: Int
    var n2: Int
    var n3: Int
    var k3: Int

    def __init__(out self, n: Int, k_in: Int):
        var odd = n
        while odd % 2 == 0:
            odd //= 2
        var a = 0
        var t = n // odd
        while t > 1:
            t //= 2
            a += 1
        self.n = n
        self.n1 = odd
        self.n2 = 1 << (a // 2)
        self.n3 = 1 << (a - a // 2)
        self.k3 = k_in // (self.n1 * self.n2)

    def t3(self) -> Int:
        return 0

    def t2(self) -> Int:
        return self.n3 * self.k3 * 2

    def t1(self) -> Int:
        return self.t2() + self.n3 * self.n2 * self.n2 * 2

    def bytes(self) -> Int:
        return self.t1() + self.n2 * self.n3 * self.n1 * self.n1 * 2


@fieldwise_init
struct Radix(TrivialRegisterPassable, DevicePassable):
    """One stage: thread (line, prefix, inner) with prefix = rest % od. Byte offsets and strides;
    the table of a prefix is tab + (prefix % tabmod) r kin 2."""
    var src: Int
    var so_line: Int
    var so_pre: Int
    var sk: Int
    var si: Int
    var dst: Int
    var to_line: Int
    var to_pre: Int
    var tj: Int
    var ti: Int
    var tab: Int
    var od: Int
    var tabmod: Int
    var inner: Int
    var total: Int

    comptime device_type: AnyType = Self

    def _to_device_type(self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "Radix"


def k_radix[r: Int, kin: Int, bytes_in: Bool, V: Int = 1](base: Base, o: Radix):
    """y[j] = sum_k T[j, k] x[k] for one thread's V consecutive inner positions (V = 1 when the
    inner stride is not 2 bytes); x is F (one byte) when bytes_in. The 2 V byte loads and stores
    are what the stage is bound by at V = 1."""
    comptime assert kin * 2 * 126 * 126 < 127 * (1 << 15), "a stage's accumulator stays below WIDE_BIAS"
    comptime assert V == 1 or not bytes_in
    var t = global_idx.x
    if t >= Int(o.total):
        return
    var inner = (t % Int(o.inner)) * V
    var rest = t // Int(o.inner)
    var pre = rest % Int(o.od)
    var line = rest // Int(o.od)
    var src = Int(o.src) + line * Int(o.so_line) + pre * Int(o.so_pre) + inner * Int(o.si)
    var dst = Int(o.dst) + line * Int(o.to_line) + pre * Int(o.to_pre) + inner * Int(o.ti)
    var tab = Int(o.tab) + (pre % Int(o.tabmod)) * r * kin * 2
    var x0 = InlineArray[SIMD[DType.int32, V], kin](fill=0)
    var x1 = InlineArray[SIMD[DType.int32, V], kin](fill=0)
    comptime for k in range(kin):
        comptime if bytes_in:
            x0[k] = Int32(base[unsafe_offset=src + k * Int(o.sk)])
        else:
            var v = base.unsafe_load[width=2 * V](src + k * Int(o.sk)).deinterleave()
            x0[k] = rebind[SIMD[DType.int32, V]](v[0].cast[DType.int32]())
            x1[k] = rebind[SIMD[DType.int32, V]](v[1].cast[DType.int32]())
    comptime for j in range(r):
        var re = SIMD[DType.int32, V](0)
        var im = SIMD[DType.int32, V](0)
        comptime for k in range(kin):
            var w = base.unsafe_load[width=2](tab + (j * kin + k) * 2)
            var w0 = Int32(w[0])
            var w1 = Int32(w[1])
            re += w0 * x0[k] - w1 * x1[k]
            im += w0 * x1[k] + w1 * x0[k]
        base.unsafe_store[width=2 * V](dst + j * Int(o.tj), f_reduce_signed(re).interleave(f_reduce_signed(im)))


def _stage[r: Int, kin: Int, bytes_in: Bool, V: Int = 1](ctx: DeviceContext, arena: Arena, o: Radix) raises:
    comptime kernel = k_radix[r, kin, bytes_in, V]
    ctx.enqueue_function[kernel](arena.buf, o, grid_dim=ceildiv(Int(o.total), BACKEND.block), block_dim=BACKEND.block)


def dft_axis[p: Params, forward: Bool, axis: Int, bytes_in: Bool = False](
    ctx: DeviceContext, arena: Arena, src: Int, dst: Int, scratch: Int, W: Int, lines: Int, tab: Int
) raises:
    """`lines` lines of h rows (W F2 per row, or W = 1 F bytes when bytes_in) -> n rows: n = 2 h
    forward (the points of G) or h inverse (coefficients), h the axis length. Stages go src -> dst
    -> scratch -> dst; `scratch` holds n rows per line and may alias `src` when src is dead after
    the call. `tab` is the DftPlan table (tables.mojo)."""
    comptime h = p.h1() if axis == 1 else p.h2()
    comptime plan = DftPlan(2 * h, h) if forward else DftPlan(h, h)
    comptime n = plan.n
    comptime n1 = plan.n1
    comptime n2 = plan.n2
    comptime n3 = plan.n3
    comptime k3 = plan.k3
    comptime assert n1 * n2 * k3 == h, "the input length must split as n1 n2 k3"
    var R = W * 2
    var Ri = W if bytes_in else R
    comptime V = 4 if axis == 2 else 1          # axis 2 rows are W = h1 or 2 h1 contiguous F2 values: 8-byte accesses
    comptime V3 = 1 if bytes_in else V          # byte input is one byte per position
    comptime assert p.h1() % V == 0             # holds for every profile (h1 = 2^a1 m1, a1 >= 2)
    _stage[n3, k3, bytes_in, V3](ctx, arena, Radix(
        src=src, so_line=h * Ri, so_pre=0, sk=n1 * n2 * Ri, si=(1 if bytes_in else 2),
        dst=dst, to_line=n * R, to_pre=0, tj=n1 * n2 * R, ti=2,
        tab=tab + plan.t3(), od=1, tabmod=1, inner=n1 * n2 * W // V3, total=lines * n1 * n2 * W // V3))
    _stage[n2, n2, False, V](ctx, arena, Radix(
        src=dst, so_line=n * R, so_pre=n1 * n2 * R, sk=n1 * R, si=2,
        dst=scratch, to_line=n * R, to_pre=n1 * R, tj=n3 * n1 * R, ti=2,
        tab=tab + plan.t2(), od=n3, tabmod=n3, inner=n1 * W // V, total=lines * n3 * n1 * W // V))
    # ponytail: at n1 = 1 this stage is a copy; skip it when a grid with a power-of-two axis matters
    _stage[n1, n1, False, V](ctx, arena, Radix(
        src=scratch, so_line=n * R, so_pre=n1 * R, sk=R, si=2,
        dst=dst, to_line=n * R, to_pre=R, tj=n2 * n3 * R, ti=2,
        tab=tab + plan.t1(), od=n2 * n3, tabmod=n2 * n3, inner=W // V, total=lines * n2 * n3 * W // V))
