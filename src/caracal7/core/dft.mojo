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
An odd part past 9 splits once more, n1 = na nb with k1 = ka + na kb and j1 = jb + nb ja (63 = 7 x 9:
16 instead of 63 multiply-adds per output, and no 126-accumulator thread): stage 1 becomes
    stage b   Yb[jj][jb][ka]     = sum_kb Tb[jj][jb, kb] Y2[jj][ka + na kb]     Tb = root^(na (jj + n2 n3 jb) kb)
    stage a   X[j]               = sum_ka Ta[jb + nb jj][ja, ka] Yb[jj][jb][ka]  Ta = root^(j ka)
with stage b in place on the scratch (a thread reads and writes the same kb = jb slots), so the
src -> dst -> scratch -> dst buffer walk of three stages still holds.
Axis 2 transforms the rows of a column (W F2 per row); axis 1 transforms each row (W = 1, the
lines are the rows of every column). Every n-row buffer has stride n W 2 per line."""
from std.math import ceildiv
from std.sys import is_defined
from std.time import perf_counter_ns
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.gpu import global_idx, thread_idx, block_idx, lane_id
from max.gpu.host import DeviceContext

from caracal7.core.field import F2, f_reduce_signed, fp_canonical
from caracal7.core.backend import BACKEND, APPLE8, frag8, mma8
from caracal7.core.bytes import Base
from caracal7.core.arena import Arena


@fieldwise_init
struct DftPlan(TrivialRegisterPassable):
    """n = n1 n2 n3 and the nonzero inputs k3 of stage 3 (k_in = n1 n2 k3); n1 = na nb, na = 1 when
    the odd part is at most 9."""
    var n: Int
    var k_in: Int
    var n1: Int
    var n2: Int
    var n3: Int
    var k3: Int
    var na: Int
    var nb: Int

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
        self.k_in = k_in
        self.n1 = odd
        self.n2 = 1 << (a // 2)
        self.n3 = 1 << (a - a // 2)
        self.k3 = k_in // (self.n1 * self.n2)
        var na = 1
        if odd > 9:
            var d = 2
            while d * d <= odd:
                if odd % d == 0:
                    na = d
                d += 1
        self.na = na
        self.nb = odd // na

    def t3(self) -> Int:
        return 0

    def t2(self) -> Int:
        return self.n3 * self.k3 * 2

    def t1(self) -> Int:
        """Stage 1, or stage b when split: n2 n3 tables of nb x nb."""
        return self.t2() + self.n3 * self.n2 * self.n2 * 2

    def ta(self) -> Int:
        """Stage a when split: n2 n3 nb tables of na x na."""
        return self.t1() + self.n2 * self.n3 * self.nb * self.nb * 2

    def bytes(self) -> Int:
        return self.ta() + (self.n2 * self.n3 * self.nb * self.na * self.na * 2 if self.na > 1 else 0)


@fieldwise_init
struct Radix(TrivialRegisterPassable, DevicePassable):
    """One stage: thread (line, prefix, inner) with prefix = rest % od, split as hi = prefix // od_lo
    and lo = prefix % od_lo when the two prefix digits have different strides (od_lo = 1 otherwise).
    Byte offsets and strides; the table of a prefix is tab + (prefix % tabmod) r kin 2."""
    var src: Int
    var so_line: Int
    var so_pre: Int
    var so_pre_lo: Int
    var sk: Int
    var si: Int
    var dst: Int
    var to_line: Int
    var to_pre: Int
    var to_pre_lo: Int
    var tj: Int
    var ti: Int
    var tab: Int
    var od: Int
    var od_lo: Int
    var tabmod: Int
    var inner: Int
    var total: Int

    comptime device_type: AnyType = Self

    def _to_device_type(self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "Radix"


def k_radix[r: Int, kin: Int, bytes_in: Bool, V: Int = 1, LB: Int = 1](base: Base, o: Radix):
    """y[j] = sum_k T[j, k] x[k] for one thread's V consecutive inner positions (V = 1 when the
    inner stride is not 2 bytes) or, at V = 1, LB consecutive lines: the lanes share the table loads,
    which outnumber the data loads r-fold. x is F (one byte) when bytes_in."""
    comptime assert kin * 2 * 126 * 126 < 127 * (1 << 15), "a stage's accumulator stays below WIDE_BIAS"
    comptime assert V == 1 or (not bytes_in and LB == 1)
    comptime VL = V * LB
    var t = global_idx.x
    if t >= Int(o.total):
        return
    var inner = (t % Int(o.inner)) * V
    var rest = t // Int(o.inner)
    var pre = rest % Int(o.od)
    var line = (rest // Int(o.od)) * LB
    var hi = pre // Int(o.od_lo)
    var lo = pre % Int(o.od_lo)
    var src = Int(o.src) + line * Int(o.so_line) + hi * Int(o.so_pre) + lo * Int(o.so_pre_lo) + inner * Int(o.si)
    var dst = Int(o.dst) + line * Int(o.to_line) + hi * Int(o.to_pre) + lo * Int(o.to_pre_lo) + inner * Int(o.ti)
    var tab = Int(o.tab) + (pre % Int(o.tabmod)) * r * kin * 2
    var x0 = InlineArray[SIMD[DType.int32, VL], kin](fill=0)
    var x1 = InlineArray[SIMD[DType.int32, VL], kin](fill=0)
    comptime for k in range(kin):
        comptime if LB > 1:
            comptime for l in range(LB):
                comptime if bytes_in:
                    x0[k][l] = Int32(base[unsafe_offset=src + l * Int(o.so_line) + k * Int(o.sk)])
                else:
                    var v = base.unsafe_load[width=2, alignment=2](src + l * Int(o.so_line) + k * Int(o.sk))
                    x0[k][l] = Int32(v[0])
                    x1[k][l] = Int32(v[1])
        elif bytes_in:
            x0[k] = Int32(base[unsafe_offset=src + k * Int(o.sk)])
        else:
            var v = base.unsafe_load[width=2 * V, alignment=2 * V](src + k * Int(o.sk)).deinterleave()
            x0[k] = rebind[SIMD[DType.int32, VL]](v[0].cast[DType.int32]())
            x1[k] = rebind[SIMD[DType.int32, VL]](v[1].cast[DType.int32]())
    comptime for j in range(r):
        var re = SIMD[DType.int32, VL](0)
        var im = SIMD[DType.int32, VL](0)
        comptime for k in range(kin):
            var w = base.unsafe_load[width=2, alignment=2](tab + (j * kin + k) * 2)
            var w0 = Int32(w[0])
            var w1 = Int32(w[1])
            re += w0 * x0[k] - w1 * x1[k]
            im += w0 * x1[k] + w1 * x0[k]
        var rr = f_reduce_signed(re)
        var ii = f_reduce_signed(im)
        comptime if LB > 1:
            comptime for l in range(LB):
                base.unsafe_store[width=2, alignment=2](dst + l * Int(o.to_line) + j * Int(o.tj), SIMD[DType.uint8, 2](rr[l], ii[l]))
        else:
            base.unsafe_store[width=2 * VL, alignment=2 * VL](dst + j * Int(o.tj), rr.interleave(ii))


comptime RADIX8_TILES = 2        # 8-position tiles per simdgroup of k_radix8
comptime RADIX8_LPB = 4          # lines per block: the table fragments load once per block
comptime RADIX8_THREADS = 128
comptime RADIX8_SG = RADIX8_THREADS // 32   # simdgroups per block: a block covers 64 positions of each of its lines
comptime RADIX8_MIN_INNER = 8    # stages with fewer contiguous positions per line keep the lane kernel


def k_radix8[r: Int, kin: Int, bytes_in: Bool, V: Int = 1, LB: Int = 1](base: Base, o: Radix):
    """k_radix on the 8x8 fp16 simdgroup op (backend.mma8) for r, kin in {8, 16}. A stage is Y = T X per
    prefix: T the r x kin table in F2, X the kin inputs of every position (line, inner) of the prefix.
    Over the reals that is A = [[W0, -W1], [W1, W0]] (2r x 2kin, rows (plane, j), columns (plane, k);
    kin columns of [W0; W1] when the input is F bytes) times B[(plane, k), position] = the input byte.
    With r and kin multiples of 8 a lane's fragment rows pair up: row tile kt holds input k = (kt mod
    kin / 8) 8 + fr of plane kt / (kin / 8), so one 4-byte load of input k at the lane's two positions
    (re, im, re, im) feeds both planes, and the lane's output rows j and r + j of tile pairs are the
    (re, im) of one output, stored as 4 bytes. No threadgroup memory, no barrier. Grid (tile chunks,
    line groups, prefixes): a block holds one prefix's A fragments in registers and walks RADIX8_LPB
    lines, each simdgroup RADIX8_TILES tiles of 8 positions; the per-tile coordinates need no division
    (a 64-bit division is a few hundred instructions on Metal; the prefix decode is once per thread). Bytes below 127 are exact in fp16, their
    products in the fp32 accumulator; 2 kin <= 32 terms stay below 2^19, so every sum is exact before
    fp_canonical. V and LB only scale the descriptor's counts: `inner` counts V positions, `total`
    counts LB lines. Radices 7 and 9 stay on the lane kernel: measured slower here (2026-09-16)."""
    comptime assert r % 8 == 0 and kin % 8 == 0 and r <= 16 and kin <= 16, "k_radix8 pairs fragment rows"
    comptime KQ = kin // 8                          # input row tiles per plane
    comptime KT = KQ if bytes_in else 2 * KQ         # real K tiles (plane, k tile)
    comptime RQ = r // 8                            # output row tiles per plane
    comptime SI = 1 if bytes_in else 2
    var sg = Int(thread_idx.x) // 32
    var rc = frag8(Int(lane_id()))
    var fr = rc[0]
    var fc = rc[1]
    var inner = Int(o.inner) * V
    var lines = (Int(o.total) // (Int(o.od) * Int(o.inner))) * LB
    var pre = Int(block_idx.z)
    var hi = pre // Int(o.od_lo)
    var lo = pre % Int(o.od_lo)
    var src_pre = Int(o.src) + hi * Int(o.so_pre) + lo * Int(o.so_pre_lo)
    var dst_pre = Int(o.dst) + hi * Int(o.to_pre) + lo * Int(o.to_pre_lo)
    var tab = Int(o.tab) + (pre % Int(o.tabmod)) * r * kin * 2
    var af = InlineArray[SIMD[DType.float16, 2], 2 * RQ * KT](fill=SIMD[DType.float16, 2](0))
    comptime for jt in range(2 * RQ):
        comptime po = jt // RQ
        var j = (jt % RQ) * 8 + fr
        comptime for kt in range(KT):
            comptime pi = kt // KQ
            var v = SIMD[DType.float16, 2](0)
            comptime for el in range(2):
                var k = (kt % KQ) * 8 + fc + el
                var w = base.unsafe_load[width=2](tab + (j * kin + k) * 2)
                var x = Float32(w[0]) if po == pi else Float32(w[1])
                comptime if po == 0 and pi == 1:
                    x = -x
                v[el] = x.cast[DType.float16]()
            af[jt * KT + kt] = v
    var tile0 = (Int(block_idx.x) * RADIX8_SG + sg) * RADIX8_TILES
    for l in range(RADIX8_LPB):
        var line = Int(block_idx.y) * RADIX8_LPB + l
        if line >= lines:                            # uniform over the block
            return
        var src_line = src_pre + line * Int(o.so_line)
        var dst_line = dst_pre + line * Int(o.to_line)
        for t in range(RADIX8_TILES):
            var p0 = (tile0 + t) * 8
            if p0 >= inner:                          # uniform over the simdgroup
                break
            var p = p0 + fc                          # the lane's positions p, p + 1; past the line end they are masked
            var ok0 = p < inner
            var ok1 = p + 1 < inner
            var acc = InlineArray[SIMD[DType.float32, 2], 2 * RQ](fill=SIMD[DType.float32, 2](0))
            comptime for kq in range(KQ):
                var off = src_line + (kq * 8 + fr) * Int(o.sk) + p * SI
                comptime if bytes_in:
                    var vb = SIMD[DType.uint8, 2](0)
                    if ok1:
                        vb = base.unsafe_load[width=2](off)
                    elif ok0:
                        vb[0] = base[unsafe_offset=off]
                    var bf = vb.cast[DType.float16]()
                    comptime for jt in range(2 * RQ):
                        acc[jt] = mma8(af[jt * KT + kq], bf, acc[jt])
                else:
                    var vb = SIMD[DType.uint8, 4](0)
                    if ok1:
                        vb = base.unsafe_load[width=4](off)
                    elif ok0:
                        vb[0] = base[unsafe_offset=off]
                        vb[1] = base[unsafe_offset=off + 1]
                    var fb = vb.cast[DType.float16]()
                    var b0 = SIMD[DType.float16, 2](fb[0], fb[2])      # plane 0 at p, p + 1
                    var b1 = SIMD[DType.float16, 2](fb[1], fb[3])      # plane 1
                    comptime for jt in range(2 * RQ):
                        acc[jt] = mma8(af[jt * KT + kq], b0, acc[jt])
                        acc[jt] = mma8(af[jt * KT + KQ + kq], b1, acc[jt])
            comptime for jq in range(RQ):
                var re = fp_canonical(acc[jq])
                var im = fp_canonical(acc[RQ + jq])
                var off = dst_line + (jq * 8 + fr) * Int(o.tj) + p * Int(o.ti)
                if ok1:
                    base.unsafe_store[width=4](off, SIMD[DType.uint8, 4](re[0], im[0], re[1], im[1]))
                elif ok0:
                    base[unsafe_offset=off] = re[0]
                    base[unsafe_offset=off + 1] = im[0]


def _stage[r: Int, kin: Int, bytes_in: Bool, V: Int = 1, LB: Int = 1](ctx: DeviceContext, arena: Arena, o: Radix) raises:
    comptime kernel = k_radix[r, kin, bytes_in, V, LB]
    comptime PROF = is_defined["CARACAL_DFT_PROFILE"]()      # -D CARACAL_DFT_PROFILE: synchronize and print every stage
    var t0 = perf_counter_ns()
    comptime if PROF:
        ctx.synchronize()
        t0 = perf_counter_ns()
    var launched = False
    comptime if APPLE8 and r % 8 == 0 and kin % 8 == 0 and r <= 16 and kin <= 16:
        if Int(o.inner) * V >= RADIX8_MIN_INNER:
            comptime kernel8 = k_radix8[r, kin, bytes_in, V, LB]
            var lines = (Int(o.total) // (Int(o.od) * Int(o.inner))) * LB
            var tiles_per_line = ceildiv(Int(o.inner) * V, 8)
            ctx.enqueue_function[kernel8](arena.buf, o, grid_dim=(ceildiv(tiles_per_line, RADIX8_SG * RADIX8_TILES), ceildiv(lines, RADIX8_LPB), Int(o.od)),
                                          block_dim=RADIX8_THREADS)
            launched = True
    if not launched:
        ctx.enqueue_function[kernel](arena.buf, o, grid_dim=ceildiv(Int(o.total), BACKEND.block), block_dim=BACKEND.block)
    comptime if PROF:
        ctx.synchronize()
        print("    radix", r, "kin", kin, "V", V, "LB", LB, "threads", o.total, "us", (perf_counter_ns() - t0) // 1000)


# TODO(perf): the axis-1 stages (W = 1) run 1 to 5 ms per stage against a memory bound near 1 ms, and are
# the largest part of the LDE (60 ms) and the encodes (96 ms) at the ECDSA grid. Next step: fuse the three
# stages of a line in threadgroup memory, one read of the input and one write of the output.
def _vcap(r: Int, V: Int) -> Int:
    """Positions per thread for a radix-r stage: at most 2 from radix 16 (r V input lanes twice, plus r V
    accumulators, must stay in registers)."""
    return 2 if r >= 16 and V > 2 else V


def dft_axis[plan: DftPlan, V: Int = 1, bytes_in: Bool = False](
    ctx: DeviceContext, arena: Arena, src: Int, dst: Int, scratch: Int, W: Int, lines: Int, tab: Int,
    dst_line: Int = 0, dst_j: Int = 0
) raises:
    """`lines` lines of h = n1 n2 k3 rows (W F2 per row, or W = 1 F bytes when bytes_in) -> plan.n rows.
    Stages go src -> dst -> scratch -> dst; `scratch` holds n rows per line and may alias `src` when src
    is dead after the call. `tab` is the plan's table (tables.mojo). The last stage writes output row j
    of a line at dst + line dst_line + j dst_j (default: lines of n contiguous rows). V consecutive
    inner positions per thread (W a multiple of V; 4 when rows are wide)."""
    comptime n = plan.n
    comptime n1 = plan.n1
    comptime n2 = plan.n2
    comptime n3 = plan.n3
    comptime k3 = plan.k3
    comptime na = plan.na
    comptime nb = plan.nb
    comptime h = n1 * n2 * k3
    comptime assert h == plan.k_in, "the input length must split as n1 n2 k3"
    comptime V3 = 1 if bytes_in else V          # byte input is one byte per position
    comptime LB = 4 if V == 1 else 1            # single-lane rows: four lines per thread instead
    comptime LB3 = 4 if V3 == 1 else 1
    if W % V != 0 or lines % LB != 0 or lines % LB3 != 0:
        raise Error("dft_axis: W is not a multiple of V, or lines of LB")
    comptime V3c = _vcap(n3, V3)                # radix 16 at four positions holds 128 input lanes: spills
    comptime V2c = _vcap(n2, V)
    comptime LB3c = 4 if V3c == 1 else 1
    comptime LB2c = 4 if V2c == 1 else 1
    var R = W * 2
    var Ri = W if bytes_in else R
    var dl = dst_line if dst_line > 0 else n * R
    var dj = dst_j if dst_j > 0 else R
    comptime if n1 == 1:
        if scratch != src:
            # the odd part is 1, so the last stage would be a copy into the output layout: stage 3 into
            # scratch, stage 2 straight into the output layout (row j n3 + hi of a line)
            _stage[n3, k3, bytes_in, V3c, LB3c](ctx, arena, Radix(
                src=src, so_line=h * Ri, so_pre=0, so_pre_lo=0, sk=n2 * Ri, si=(1 if bytes_in else 2),
                dst=scratch, to_line=n * R, to_pre=0, to_pre_lo=0, tj=n2 * R, ti=2,
                tab=tab + plan.t3(), od=1, od_lo=1, tabmod=1, inner=n2 * W // V3c, total=lines // LB3c * n2 * W // V3c))
            _stage[n2, n2, False, V2c, LB2c](ctx, arena, Radix(
                src=scratch, so_line=n * R, so_pre=n2 * R, so_pre_lo=0, sk=R, si=2,
                dst=dst, to_line=dl, to_pre=dj, to_pre_lo=0, tj=n3 * dj, ti=2,
                tab=tab + plan.t2(), od=n3, od_lo=1, tabmod=n3, inner=W // V2c, total=lines // LB2c * n3 * W // V2c))
            return
    _stage[n3, k3, bytes_in, V3c, LB3c](ctx, arena, Radix(
        src=src, so_line=h * Ri, so_pre=0, so_pre_lo=0, sk=n1 * n2 * Ri, si=(1 if bytes_in else 2),
        dst=dst, to_line=n * R, to_pre=0, to_pre_lo=0, tj=n1 * n2 * R, ti=2,
        tab=tab + plan.t3(), od=1, od_lo=1, tabmod=1, inner=n1 * n2 * W // V3c, total=lines // LB3c * n1 * n2 * W // V3c))
    _stage[n2, n2, False, V2c, LB2c](ctx, arena, Radix(
        src=dst, so_line=n * R, so_pre=n1 * n2 * R, so_pre_lo=0, sk=n1 * R, si=2,
        dst=scratch, to_line=n * R, to_pre=n1 * R, to_pre_lo=0, tj=n3 * n1 * R, ti=2,
        tab=tab + plan.t2(), od=n3, od_lo=1, tabmod=n3, inner=n1 * W // V2c, total=lines // LB2c * n3 * n1 * W // V2c))
    comptime if na == 1:
        # at n1 = 1 with scratch aliasing src this stage is a copy into the output layout
        _stage[n1, n1, False, V, LB](ctx, arena, Radix(
            src=scratch, so_line=n * R, so_pre=n1 * R, so_pre_lo=0, sk=R, si=2,
            dst=dst, to_line=dl, to_pre=dj, to_pre_lo=0, tj=n2 * n3 * dj, ti=2,
            tab=tab + plan.t1(), od=n2 * n3, od_lo=1, tabmod=n2 * n3, inner=W // V, total=lines // LB * n2 * n3 * W // V))
    else:
        # stage b in place: prefix jj, digit kb -> jb in the same slots; then stage a with prefix (jj, jb)
        _stage[nb, nb, False, V, LB](ctx, arena, Radix(
            src=scratch, so_line=n * R, so_pre=n1 * R, so_pre_lo=0, sk=na * R, si=2,
            dst=scratch, to_line=n * R, to_pre=n1 * R, to_pre_lo=0, tj=na * R, ti=2,
            tab=tab + plan.t1(), od=n2 * n3, od_lo=1, tabmod=n2 * n3, inner=na * W // V, total=lines // LB * n2 * n3 * na * W // V))
        _stage[na, na, False, V, LB](ctx, arena, Radix(
            src=scratch, so_line=n * R, so_pre=n1 * R, so_pre_lo=na * R, sk=R, si=2,
            dst=dst, to_line=dl, to_pre=dj, to_pre_lo=n2 * n3 * dj, tj=n2 * n3 * nb * dj, ti=2,
            tab=tab + plan.ta(), od=n2 * n3 * nb, od_lo=nb, tabmod=n2 * n3 * nb, inner=W // V, total=lines // LB * n2 * n3 * nb * W // V))
