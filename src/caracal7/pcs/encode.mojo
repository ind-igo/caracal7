"""Level-1 encoder: trace -> coeff -> stored -> packed -> code (docs/design.md sections 3, 4; spec 9.1).

Every kernel is rung 1 of the ladder: one thread per output, correct against the scalar
references in tests/test_encode.mojo, measured in bench/bench_encode.mojo. Kernels take the arena
base pointer plus width-typed regions of it (rule 2, bytes.mojo) and are parameterized on Params (rule 3).

Buffers (bytes; shapes slowest ... fastest):
    trace   (column, x2, x1)          F
    ctmp    (column, x2, k1, 2)       F2, after the axis-1 inverse DFT; then (column, k2, x1 + 2^a1 r1, 2),
                                      the coefficients after the m1-point DFT on the y1 digit
    coeff   (column, k2, k1, 2)       F2 monomial coefficients
    stored  (column, slot)            F, Frobenius-real slots (t, x1', x2, r)
    packed  (i, column, 4)            F4, coordinate basis (1, i, j, ij)
    etmp    (lin, t1, column, 4)      F4, RS intermediate; lin = d5*63 + d7*9 + d9
    code    (s, column, 4)            F4, leaf-major, leaf s is the point g^s

From `packed` on the column index is fastest: a SIMD group of threads handles one (i, t1) for 32
consecutive columns, so every load and store of the RS passes is contiguous (ladder step 2), and
the twiddle loads are group-wide broadcasts. Full coalescing needs 32+ columns per launch.

RS encode of K message symbols (i, columns, 4) on an RS domain of m cosets of the order-L0 subgroup,
L0 = 2^b * M with M | 315 (tables.RsTables); level 1 has K = N/4, the tail levels K = rows with the
8 E-valued columns seen as 32 F4 columns. Per coset k: twist x_i by gamma4^(k i), then
    Good-Thomas across 2^b x M (no twiddles between), Cooley-Tukey inside 2^b (twiddles folded into
    each stage's coefficients), then the radices of M (5, 7, 9 or 3) inside M.
    input index  i  <-> (i mod 2^b, i mod M), i mod M <-> (i mod 5, i mod 7, i mod 9)
    output index s  =  (M t1 + 2^b t2) mod L0,  t2 = sum_r (M / r) t_r mod M
    the 2^b axis  2^b = B2 B1, B1 = 2^min(b, 6); n = n1 + B1 n2, t1 = k2 + B2 k1. Line lin has the
                 Q = ceil(K / M) inputs i = crt(lin) + M q, at most ceil(Q / B1) per residue n1.
    gather       etmp[lin, k2 + B2 n1] = sum_{i mod B1 = n1} x_i gA^(i k2): the B2-point DFTs, done as
                 the sparse sums they are (one term per output at level 1), one thread per (lin, n1).
    stage 2^lr   decimation in frequency on the B1 digits: r-point DFT over the top digit of each
                 size-S block with W_S folded in, in place; the block ends digit-reversed (`_t1_true`).
    stage r      r-point DFT along digit d_r of lin, in place, twiddles w_r^(t k)
    the last stage also scatters to the leaf-major `code`, coset k at s + k L0.
"""

from std.math import ceildiv
from std.gpu import thread_idx, block_idx, block_dim, global_idx
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from layout import row_major, stack_allocation

from caracal7.core.field import F2, F4, V2, V4, f_add, f_mul
from caracal7.core.field import fp_reduce, fp_center, fp_canonical, fp_mul2, fp_mul4, fp_const_mul
from caracal7.core.params import Params
from caracal7.core.tables import TableLayout, RsTables, two_adic, rs_factors
from caracal7.core.arena import Bump
from caracal7.core.bytes import Base, Buf, u16
from caracal7.core.arena import Arena
from caracal7.core.backend import BACKEND, Strided, Bytes, launch_gemm_f2, strided
from caracal7.core.dft import dft_axis, DftPlan, Radix, _stage as radix_stage

comptime CW = 32                    # columns per SIMD group in the RS passes (block x)
comptime RW = 8                     # (t1, line) rows per block (block y)


struct EncLayout(TrivialRegisterPassable):
    """Arena offsets of the encoder buffers for `columns` columns."""
    var columns: Int
    var trace: Int
    var ctmp: Int
    var coeff: Int
    var stored: Int
    var packed: Int
    var etmp: Int
    var code: Int

    def __init__[p: Params](out self, mut bump: Bump, columns: Int):
        self.columns = columns
        self.trace = bump.alloc(columns * p.N())
        self.ctmp = bump.alloc(columns * p.N() * 2)
        self.coeff = bump.alloc(columns * p.N() * 2)
        self.stored = bump.alloc(columns * p.N())
        self.packed = bump.alloc(columns * p.N())
        self.etmp = bump.alloc(columns * p.L0 * 4)
        self.code = bump.alloc(p.L() * columns * 4)


# ---- idft2: inverse 2D DFT over F2, three radix stages per axis (dft.mojo) ----

def idft2[p: Params](ctx: DeviceContext, arena: Arena, trace: Int, ctmp: Int, coeff: Int,
                     columns: Int, tab: TableLayout) raises:
    """trace (column, x2, x1) F -> coeff (column, k2, k1, 2): inverse DFT per axis, three radix stages
    each (dft.mojo). Axis 1 reads the bytes into ctmp with coeff as scratch; axis 2 reads ctmp, its own
    scratch, into coeff."""
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    dft_axis[DftPlan(h1, h1), bytes_in=True](ctx, arena, trace, ctmp, coeff, 1, columns * h2, tab.base + tab.inv1)
    dft_axis[DftPlan(h2, h2), 4](ctx, arena, ctmp, coeff, ctmp, h1, columns, tab.base + tab.inv2)


# ---- to_stored: mixed basis on the odd digit, Frobenius-real slots (spec 9.1) ----

@always_inline
def slot_target[p: Params](slot: Int) -> Tuple[Int, Int, Int, Int]:
    """slot -> (x1, x2, r, coord): the binary index whose c_x(r) this slot holds, and which
    F2 coordinate of it (0 = u, 1 = v). Fixed classes hold real values, coord 0."""
    comptime H1 = 1 << (p.a1 - 1)
    comptime H2 = 1 << (p.a2 - 1)
    var t = slot & 1
    var rest = slot >> 1
    var x1p = rest % H1
    rest //= H1
    var x2 = rest % (1 << p.a2)
    var r = rest // (1 << p.a2)
    if x1p >= 1:
        return (x1p, x2, r, t)
    if x2 == 0 or x2 == H2:
        return (t * H1, x2, r, 0)
    if x2 < H2:
        return (0, x2, r, t)
    return (H1, x2 - H2, r, t)


def k_to_stored[p: Params](base: Base, tmp: Buf[2], stored: Buf[1], columns: Int32):
    """stored[c, slot] = coord of c_x(r) = sum_y coeff[c, x2 + 2^a2 y2, x1 + 2^a1 y1] rho1^(y1 r1) rho2^(y2 r2),
    both sums already taken by the radix passes of `to_packed` into tmp[c, x2 + 2^a2 r2, x1 + 2^a1 r1]:
    a gather. One thread per slot pair (t = 0, 1): the pair holds the two coordinates of one value, or
    in the fixed classes coordinate 0 of two values (x1 = 0 and x1 = H1)."""
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    comptime N = p.N()
    comptime H1 = 1 << (p.a1 - 1)
    var gid = Int(global_idx.x)
    if gid >= Int(columns) * (N // 2):
        return
    var c = gid // (N // 2)
    var slot = 2 * (gid % (N // 2))
    var x1: Int
    var x2: Int
    var r: Int
    var coord: Int
    x1, x2, r, coord = slot_target[p](slot)
    var r1 = r % p.m1
    var r2 = r // p.m1
    var k1 = x1 + (1 << p.a1) * r1
    var row = (c * h2 + x2 + (1 << p.a2) * r2) * h1
    var acc = tmp.load(base, row + k1)
    if x1 == 0 and coord == 0 and slot_target[p](slot + 1)[0] == H1:     # a fixed class: the second value, x1 = H1
        acc[1] = tmp.load(base, row + k1 + H1)[0]
    base.unsafe_store[width=2](stored.at(c * N + slot), acc)


# ---- pack: four slots on the packing digit -> one F4 symbol ----

@always_inline
def pack_slot[p: Params](i: Int, j: Int) -> Int:
    """slot(i, j): i over (t, x1', x2 >> 2, r), j the two low bits of x2."""
    comptime H1 = 1 << (p.a1 - 1)
    var t = i & 1
    var rest = i >> 1
    var x1p = rest % H1
    rest //= H1
    var x2h = rest % (1 << (p.a2 - 2))
    var r = rest // (1 << (p.a2 - 2))
    return t + 2 * (x1p + H1 * (j + 4 * x2h + (1 << p.a2) * r))


@always_inline
def pack_index[p: Params](slot: Int) -> Tuple[Int, Int]:
    """The inverse of pack_slot: slot -> (i, j)."""
    comptime H1 = 1 << (p.a1 - 1)
    var t = slot & 1
    var rest = slot >> 1
    var x1p = rest % H1
    rest //= H1
    var x2 = rest % (1 << p.a2)
    var r = rest // (1 << p.a2)
    return (t + 2 * (x1p + H1 * ((x2 >> 2) + (1 << (p.a2 - 2)) * r)), x2 & 3)


def k_pack[p: Params](base: Base, stored: Buf[1], packed: Buf[4], columns: Int32):
    comptime N = p.N()
    var gid = Int(global_idx.x)
    if gid >= Int(columns) * (N // 4):
        return
    var c = gid % Int(columns)
    var i = gid // Int(columns)
    var v = F4(0)
    comptime for j in range(4):
        v[j] = stored.load(base, c * N + pack_slot[p](i, j))
    packed.store(base, gid, v)


# ---- rs_encode ----

@always_inline
def _rs_thread[V: Int = 1](columns: Int32) -> Tuple[Int, Int, Bool]:
    """2D launch: block (CW, RW); x walks groups of V columns, y walks (t1, line) rows. Returns (c, row, valid)."""
    var c = (Int(block_idx.x) * CW + Int(thread_idx.x)) * V
    var row = Int(block_idx.y) * RW + Int(thread_idx.y)
    return (c, row, c < Int(columns))


@always_inline
def _adic_split(b: Int) -> Tuple[Int, Int]:
    """(log2 B1, log2 B2): 2^b = B2 B1 with B1 = 2^min(b, 6). The gather sums the B2 axis, radix stages do B1."""
    var b1 = min(b, 6)
    return (b1, b - b1)


@always_inline
def _t1_true(pos: Int, b: Int) -> Int:
    """Position within a lin block -> the 2-adic output index t1. The radix stages (8, 8, ..., then the
    remainder) leave the B1 digits reversed: k = e + r k' sits at T e + pos(k')."""
    var b1: Int
    var b2: Int
    b1, b2 = _adic_split(b)
    var p = pos >> b2
    var k = 0
    var shift = 0
    var S = b1
    while S > 0:
        var lr = min(3, S)
        var t = S - lr
        k += (p >> t) << shift
        shift += lr
        p &= (1 << t) - 1
        S = t
    return (k << b2) | (pos & ((1 << b2) - 1))


def k_rs_gather[B2: Int](base: Base, src: Buf[4], etmp: Buf[4], ga: Buf[4], crt: Buf[1], twist: Buf[4],
                         columns: Int32, K: Int32, b: Int32, M: Int32, Q: Int32, minv: Int32, twisted: Int32):
    """etmp[lin, k2 + B2 n1] = sum over inputs i = crt(lin) + M q with i mod B1 = n1 of x_i gA^(i k2),
    x twisted by gamma4^(k i) on coset k. q runs over one residue class mod B1: q0 = (n1 - crt) M^-1.
    One thread per (lin, n1) computes the B2 outputs k2 < B2: each input is loaded and twisted once."""
    var c: Int
    var row: Int
    var ok: Bool
    c, row, ok = _rs_thread(columns)
    var bb = Int(b)
    var Mi = Int(M)
    var b1: Int
    var b2: Int
    b1, b2 = _adic_split(bb)
    if not ok or row >= Mi << b1:
        return
    var lin = row >> b1
    var n1 = row & ((1 << b1) - 1)
    var c0 = u16(base, crt.at(lin * 2))
    var mask1 = (1 << b1) - 1
    var q = (((n1 - c0) & mask1) * Int(minv)) & mask1
    # fp32 lanes: x (twisted and reduced, |x| <= 190) times a canonical gA power is below 192 K per
    # coordinate; 64 terms stay below 12.3 M, exact
    var acc = InlineArray[V4, B2](fill=V4(0))
    var terms = 0
    while q < Int(Q):
        var i = c0 + Mi * q
        if i < Int(K):
            var x = src.load(base, i * Int(columns) + c).cast[DType.float32]()
            if twisted != 0:
                x = fp_reduce(fp_mul4(x, twist.load(base, i).cast[DType.float32]()))
            comptime for k2 in range(B2):
                acc[k2] += fp_mul4(ga.load(base, (i * k2) & ((1 << bb) - 1)).cast[DType.float32](), x)
            terms += 1
            if terms == 64:
                comptime for k2 in range(B2):
                    acc[k2] = fp_reduce(acc[k2])
                terms = 0
        q += 1 << b1
    var out = (((lin << b1) + n1) << b2) * Int(columns) + c
    comptime for k2 in range(B2):
        etmp.store(base, out + k2 * Int(columns), fp_canonical(acc[k2]))


@always_inline
def _dif8(xs: InlineArray[V4, 8], wr: InlineArray[V2, 8]) -> InlineArray[V4, 8]:
    """Decimation in frequency: y[2m] = DFT4(x_n + x_(n+4)), y[2m+1] = DFT4((x_n - x_(n+4)) W_8^n),
    DFT4 with W_4 = W_8^2 and W_8^4 = -1: 5 products, 24 sums (a dense 8 x 8 is 64 products).
    Inputs |x| <= 190 and |W| <= 63 keep every sum below 2^24; the outputs are reduced (|y| <= 190)."""
    var ys = InlineArray[V4, 8](fill=V4(0))
    var a = InlineArray[V4, 4](fill=V4(0))
    var d = InlineArray[V4, 4](fill=V4(0))
    comptime for n in range(4):
        a[n] = xs[n] + xs[n + 4]
        comptime if n == 0:
            d[n] = xs[n] - xs[n + 4]
        else:
            d[n] = fp_mul2(wr[n], xs[n] - xs[n + 4])
    comptime for half in range(2):
        var v0 = a[0] if half == 0 else d[0]
        var v1 = a[1] if half == 0 else d[1]
        var v2 = a[2] if half == 0 else d[2]
        var v3 = a[3] if half == 0 else d[3]
        var c3 = fp_mul2(wr[2], v1 - v3)
        ys[half] = fp_reduce(v0 + v2 + v1 + v3)
        ys[half + 2] = fp_reduce(v0 - v2 + c3)
        ys[half + 4] = fp_reduce(v0 + v2 - v1 - v3)
        ys[half + 6] = fp_reduce(v0 - v2 - c3)
    return ys^


def k_rs_stage64(base: Base, etmp: Buf[4], ga: Buf[4], columns: Int32, b: Int32, M: Int32):
    """The two radix-8 steps of one 64-point block (B1 = 64) in one launch: block (CW, 8) owns the
    block of one (lin, k2) for CW columns. Thread n_lo does the S = 64 step over positions n_lo + 8 k
    with the W_64 twiddle, the eight threads exchange canonical bytes through 8 KB of threadgroup
    memory, then thread j does the S = 8 step over positions 8 j + k in place. Same digit order as
    the two-launch path (`k_rs_stage2` at S = 6 then S = 3). One sweep of etmp instead of two."""
    var c = Int(block_idx.x) * CW + Int(thread_idx.x)
    var n_lo = Int(thread_idx.y)
    var bb = Int(b)
    var b2 = bb - 6
    var blk = Int(block_idx.y)
    var cols = Int(columns)
    var ok = c < cols and blk < (Int(M) << b2)
    var k2 = blk & ((1 << b2) - 1)
    var block_base = ((((blk >> b2) << 6) << b2) + k2) * cols + c
    var pos_step = (1 << b2) * cols
    var sh = stack_allocation[DType.uint8, address_space=AddressSpace.SHARED](row_major[64 * CW * 4]())
    var wr = InlineArray[V2, 8](fill=V2(0))       # W_8^m = gA^(m 2^b / 8)
    comptime for m in range(8):
        wr[m] = fp_center(base.unsafe_load[width=2](ga.at(m << (bb - 3))))
    var xs = InlineArray[V4, 8](fill=V4(0))
    if ok:
        comptime for k in range(8):
            xs[k] = etmp.load(base, block_base + (n_lo + 8 * k) * pos_step).cast[DType.float32]()
    var ys = _dif8(xs, wr)
    comptime for e in range(8):
        var y = ys[e]
        comptime if e > 0:
            var ws = fp_center(base.unsafe_load[width=2](ga.at(((e * n_lo) << (bb - 6)) & ((1 << bb) - 1))))
            y = fp_mul2(ws, y)
        sh.ptr.unsafe_store(((n_lo + 8 * e) * CW + Int(thread_idx.x)) * 4, fp_canonical(y))
    barrier()
    comptime for k in range(8):
        xs[k] = sh.ptr.unsafe_load[width=4](((8 * n_lo + k) * CW + Int(thread_idx.x)) * 4).cast[DType.float32]()
    ys = _dif8(xs, wr)
    if ok:
        comptime for e in range(8):
            etmp.store(base, block_base + (8 * n_lo + e) * pos_step, fp_canonical(ys[e]))


def k_rs_stage2[r: Int](base: Base, etmp: Buf[4], ga: Buf[4], columns: Int32, b: Int32, M: Int32, s: Int32):
    """One decimation-in-frequency step of size S = 2^s on the B1 digits: for each block of S positions
    (stride B2) and each n_lo < T = S / r, y[e] = W_S^(n_lo e) sum_k x[n_lo + T k] W_r^(k e), stored at
    n_lo + T e. Both roots are gA powers of order at most 64, so in F2. The r powers of W_r sit in
    registers, indexed at compile time; the table is read r + r times, not r^2 (that was half the stage)."""
    var c: Int
    var row: Int
    var ok: Bool
    c, row, ok = _rs_thread(columns)
    var bb = Int(b)
    var Mi = Int(M)
    var ss = Int(s)
    comptime lr = 1 if r == 2 else (2 if r == 4 else 3)
    if not ok or row >= (Mi << bb) >> lr:
        return
    var b2 = bb - min(bb, 6)
    var t = ss - lr
    var k2 = row & ((1 << b2) - 1)
    var rest = row >> b2
    var n_lo = rest & ((1 << t) - 1)
    rest >>= t
    var first = ((((rest << ss) + n_lo) << b2) + k2) * Int(columns) + c
    var step = (1 << (t + b2)) * Int(columns)
    var shift = bb - ss
    var mask = (1 << bb) - 1
    var wr = InlineArray[V2, r](fill=V2(0))       # W_r^m = gA^(m 2^b / r)
    comptime for m in range(r):
        wr[m] = fp_center(base.unsafe_load[width=2](ga.at(m << (bb - lr))))
    var xs = InlineArray[V4, r](fill=V4(0))
    comptime for k in range(r):
        xs[k] = etmp.load(base, first + k * step).cast[DType.float32]()
    var ys = InlineArray[V4, r](fill=V4(0))
    comptime if r == 8:
        var y8 = _dif8(rebind[InlineArray[V4, 8]](xs), rebind[InlineArray[V2, 8]](wr))
        comptime for e in range(8):
            ys[e] = y8[e]
    else:
        comptime for e in range(r):
            var acc = V4(0)
            comptime for k in range(r):
                acc += fp_mul2(wr[(e * k) % r], xs[k])
            ys[e] = fp_reduce(acc)
    comptime for e in range(r):
        var y = ys[e]
        comptime if e > 0:
            var ws = fp_center(base.unsafe_load[width=2](ga.at(((e * n_lo) << shift) & mask)))
            y = fp_mul2(ws, y)
        etmp.store(base, first + e * step, fp_canonical(y))


@always_inline
def _ldw[V: Int](base: Base, buf: Buf[4], i: Int, n: Int) -> SIMD[DType.float32, 4 * V]:
    """V consecutive F4 values from element i as float lanes; only the first n are valid (n >= V: one
    wide load, the common case; the last block of columns loads them one by one). V a power of 2;
    halves are joined because Metal has no vector insert."""
    comptime W = 4 * V
    if n >= V:
        return base.unsafe_load[width=W](buf.at(i)).cast[DType.float32]()
    comptime if V == 1:
        return rebind[SIMD[DType.float32, W]](SIMD[DType.float32, 4](0))
    else:
        comptime H = V // 2
        return rebind[SIMD[DType.float32, W]](_ldw[H](base, buf, i, n).join(_ldw[H](base, buf, i + H, n - H)))


@always_inline
def _stw[V: Int](base: Base, buf: Buf[4], i: Int, n: Int, y: SIMD[DType.uint8, 4 * V]):
    """Store V consecutive F4 values at element i; only the first n are valid."""
    comptime W = 4 * V
    if n >= V:
        base.unsafe_store[width=W](buf.at(i), y)
        return
    comptime if V > 1:
        comptime H = V // 2
        _stw[H](base, buf, i, n, rebind[SIMD[DType.uint8, 4 * H]](y.slice[W // 2]()))
        _stw[H](base, buf, i + H, n - H, rebind[SIMD[DType.uint8, 4 * H]](y.slice[W // 2, offset=W // 2]()))


@always_inline
def _mul2w[W: SIMDLength](w: V2, y: SIMD[DType.float32, W]) -> SIMD[DType.float32, W]:
    """F2 times each F4 in the W / 4 lane groups of y."""
    comptime if W == 4:
        return rebind[SIMD[DType.float32, W]](fp_mul2(w, rebind[V4](y)))
    else:
        return rebind[SIMD[DType.float32, W]](_mul2w(w, y.slice[W // 2]()).join(_mul2w(w, y.slice[W // 2, offset=W // 2]())))


@always_inline
def _jw[W: SIMDLength](y: SIMD[DType.float32, W]) -> SIMD[DType.float32, W]:
    """j times each F4 in the W / 4 lane groups of y."""
    comptime if W == 4:
        return rebind[SIMD[DType.float32, W]](fp_const_mul[3](rebind[V4](y)))
    else:
        return rebind[SIMD[DType.float32, W]](_jw(y.slice[W // 2]()).join(_jw(y.slice[W // 2, offset=W // 2]())))


@always_inline
def _dft_odd[r: Int, W: SIMDLength](base: Base, wr: Buf[4], xs: InlineArray[SIMD[DType.float32, W], r]) -> InlineArray[SIMD[DType.float32, W], r]:
    """r-point DFT of xs (W / 4 F4 values per lane group, |x| <= 126) with the twiddles of row 1 of
    the (r, r) table `wr`. The twiddles of radix 3, 7, 9 lie in F (their orders divide 126); radix 5
    needs F4. The r - 1 powers sit in registers, indexed at compile time. Symmetric form: with
    c_m = (w^m + w^-m) / 2 and s_m = (w^m - w^-m) / 2, y_t = x_0 + sum_k c_(tk) (x_k + x_(r-k)) + s_(tk)
    (x_k - x_(r-k)) and y_(r-t) flips the s sum: h = (r - 1) / 2 products per pair instead of r^2.
    For radix 5 the Frobenius of F4 / F2 maps w to w^-1, so c_m lies in F2 and s_m in F2 j: the F4
    products become F2 x F4 ones. fp32 lanes: |c|, |s| <= 64, |x_k +- x_(r-k)| <= 252, every sum below
    2^17 (radix 5: j (x_k - x_(r-k)) has lanes <= 756, sums below 2^19); the outputs are unreduced."""
    comptime VF = SIMD[DType.float32, W]
    comptime h = (r - 1) // 2
    var pw = InlineArray[V4, r](fill=V4(0))         # w_r^m, row 1 of the (r, r) table
    comptime for m in range(1, r):
        pw[m] = fp_center(wr.load(base, r + m))
    var cw = InlineArray[V4, h + 1](fill=V4(0))     # (w^m + w^-m) / 2, in F2 for every radix
    var sw = InlineArray[V4, h + 1](fill=V4(0))     # (w^m - w^-m) / 2, in F2 j for radix 5 (lanes 2, 3)
    cw[0] = V4(1, 0, 0, 0)                          # t k = 0 mod r happens for radix 9
    comptime for m in range(1, h + 1):
        cw[m] = fp_reduce((pw[m] + pw[r - m]) * 64.0)
        sw[m] = fp_reduce((pw[m] - pw[r - m]) * 64.0)
    var ys = InlineArray[VF, r](fill=VF(0))
    var sa = InlineArray[VF, h + 1](fill=VF(0))     # x_k + x_(r-k), x_k - x_(r-k)
    var sb = InlineArray[VF, h + 1](fill=VF(0))
    ys[0] = xs[0]
    comptime for k in range(1, h + 1):
        sa[k] = xs[k] + xs[r - k]
        sb[k] = xs[k] - xs[r - k]
        ys[0] += sa[k]
    comptime for t in range(1, h + 1):
        var even = xs[0]
        var odd = VF(0)
        comptime for k in range(1, h + 1):
            comptime m = (t * k) % r
            comptime idx = m if m <= h else r - m
            comptime sign = Float32(1.0) if m <= h else Float32(-1.0)
            comptime if r == 5:
                even += _mul2w(cw[idx].slice[2](), sa[k])
                odd += _mul2w(sw[idx].slice[2, offset=2](), _jw(sb[k])) * sign
            else:
                even = sa[k].fma(VF(cw[idx][0]), even)
                odd = sb[k].fma(VF(sw[idx][0] * sign), odd)
        ys[t] = even + odd
        ys[r - t] = even - odd
    return ys^


@always_inline
def _scatter(base: Base, ruri: Buf[1], lin: Int, s_t1: Int, L0: Int, bb: Int) -> Int:
    """Leaf index of line lin's output t1: s = M t1 + 2^b ruri(lin) mod L0."""
    var s = s_t1 + (u16(base, ruri.at(lin * 2)) << bb)    # < 2 L0
    return s - L0 if s >= L0 else s


def k_rs_stage[r: Int, stride: Int](base: Base, etmp: Buf[4], wr: Buf[4], code: Buf[4], ruri: Buf[1],
                                    columns: Int32, b: Int32, M: Int32):
    """In-place r-point DFT along one digit of lin; the last stage (stride 1) scatters into `code`.
    The stride is comptime: a runtime division here costs a quarter of the encoder. Loads and stores
    bound these stages (a pass is four in-place sweeps of etmp, one per coset, near the bandwidth of
    the strided access pattern). Two columns per thread (8-byte accesses) gains only on radix 5;
    radix 7 and 9 spill registers at 2 (measured slower at 2 and 4)."""
    comptime final = stride == 1
    comptime V = 2 if r == 5 else 1
    comptime VF = SIMD[DType.float32, 4 * V]
    var c: Int
    var rest: Int
    var ok: Bool
    c, rest, ok = _rs_thread[V](columns)
    var n = Int(columns) - c
    var bb = Int(b)
    var Mi = Int(M)
    comptime st = stride
    if not ok or rest >= (Mi // r) << bb:
        return
    var t1 = rest & ((1 << bb) - 1)
    var l = rest >> bb
    var first_lin = (l // st) * r * st + l % st
    var col_off = t1 * Int(columns) + c        # + lin * 2^b * columns
    var xs = InlineArray[VF, r](fill=VF(0))
    comptime for k in range(r):
        xs[k] = _ldw[V](base, etmp, ((first_lin + k * st) << bb) * Int(columns) + col_off, n)
    var ys = _dft_odd[r](base, wr, xs)
    var s_t1 = Mi * _t1_true(t1, bb) if final else 0
    comptime for t in range(r):
        var y = fp_canonical(ys[t])
        var lin = first_lin + t * st
        comptime if final:
            _stw[V](base, code, _scatter(base, ruri, lin, s_t1, Mi << bb, bb) * Int(columns) + c, n, y)
        else:
            _stw[V](base, etmp, (lin << bb) * Int(columns) + col_off, n, y)


def k_rs_stage_pair[ra: Int, rb: Int](base: Base, etmp: Buf[4], wa: Buf[4], wb: Buf[4], code: Buf[4], ruri: Buf[1],
                                      columns: Int32, b: Int32, M: Int32):
    """The last two odd stages in one launch: radix ra along the digit of stride rb, then radix rb
    along the last digit with the scatter into `code`. Block (CW, max(ra, rb)) owns the ra rb lines
    of one (t1, l) for CW columns; the exchange goes through ra rb CW F4 bytes of threadgroup
    memory (8 KB at 63 lines). One sweep of etmp instead of two: the stages are memory-bound."""
    var c = Int(block_idx.x) * CW + Int(thread_idx.x)
    var y = Int(thread_idx.y)
    var row = Int(block_idx.y)
    var bb = Int(b)
    var Mi = Int(M)
    var cols = Int(columns)
    var ok = c < cols and row < (Mi // (ra * rb)) << bb
    var t1 = row & ((1 << bb) - 1)
    var base_lin = (row >> bb) * ra * rb
    var col_off = t1 * cols + c
    var sh = stack_allocation[DType.uint8, address_space=AddressSpace.SHARED](row_major[ra * rb * CW * 4]())
    if ok and y < rb:                                   # radix ra over lines base + k rb + y
        var xs = InlineArray[V4, ra](fill=V4(0))
        comptime for k in range(ra):
            xs[k] = etmp.load(base, ((base_lin + k * rb + y) << bb) * cols + col_off).cast[DType.float32]()
        var ys = _dft_odd[ra](base, wa, xs)
        comptime for t in range(ra):
            sh.ptr.unsafe_store(((t * rb + y) * CW + Int(thread_idx.x)) * 4, fp_canonical(ys[t]))
    barrier()
    if ok and y < ra:                                   # radix rb over lines base + y rb + k
        var xs = InlineArray[V4, rb](fill=V4(0))
        comptime for k in range(rb):
            xs[k] = sh.ptr.unsafe_load[width=4](((y * rb + k) * CW + Int(thread_idx.x)) * 4).cast[DType.float32]()
        var ys = _dft_odd[rb](base, wb, xs)
        var s_t1 = Mi * _t1_true(t1, bb)
        comptime for t in range(rb):
            var s = _scatter(base, ruri, base_lin + y * rb + t, s_t1, Mi << bb, bb)
            code.store(base, s * cols + c, fp_canonical(ys[t]))


# ---- host orchestration: enqueue the whole encoder on one stream ----

@always_inline
def grid(n: Int) -> Int:
    return ceildiv(n, BACKEND.block)


def encode[p: Params](ctx: DeviceContext, arena: Arena, e: EncLayout, tab: TableLayout) raises:
    to_packed[p](ctx, arena, e, tab)
    rs_encode[p](ctx, arena, e, tab)


def to_packed[p: Params](ctx: DeviceContext, arena: Arena, e: EncLayout, tab: TableLayout) raises:
    """trace -> coeff -> stored -> packed."""
    var cols = Int32(e.columns)
    var n_grid = e.columns * p.N()
    idft2[p](ctx, arena, e.trace, e.ctmp, e.coeff, e.columns, tab)
    # the y1 digit: an m1-point DFT with rho1 twiddles per (column, k2, x1), coeff -> ctmp
    comptime B1 = 1 << p.a1
    comptime m1 = p.m1
    var src: Int
    comptime if m1 > 1:
        radix_stage[m1, m1, False](ctx, arena, Radix(
            src=e.coeff, so_line=p.h1() * 2, so_pre=0, so_pre_lo=0, sk=B1 * 2, si=2,
            dst=e.ctmp, to_line=p.h1() * 2, to_pre=0, to_pre_lo=0, tj=B1 * 2, ti=2,
            tab=tab.base + tab.rho1t, od=1, od_lo=1, tabmod=1, inner=B1, total=e.columns * p.h2() * B1))
        src = e.ctmp
    else:
        src = e.coeff
    # the y2 digit: an m2-point DFT with rho2 twiddles per (column, x2, k1), rows x2 + B2 y2 -> x2 + B2 r2,
    # a row of h1 F2 as the inner axis with x2; split as na x nb past 9 like dft_axis (y2 = ka + na kb,
    # r2 = jb + nb ja). coeff must survive for the LDE, so the passes go src -> code -> ctmp (code is
    # free until rs_encode). Before 2026-09-11 k_to_stored summed the m2 terms per slot: 72 of the
    # 85 ms encode at h2 = 8064
    comptime B2 = 1 << p.a2
    comptime m2 = p.m2
    comptime plan2 = DftPlan(m2, m2)
    comptime na = plan2.na
    comptime nb = plan2.nb
    comptime W = p.h1()
    comptime R = W * 2
    comptime LR = p.h2() * R
    var src2: Int
    comptime if m2 > 1:
        comptime if na == 1:
            radix_stage[m2, m2, False, 4](ctx, arena, Radix(
                src=src, so_line=LR, so_pre=0, so_pre_lo=0, sk=B2 * R, si=2,
                dst=e.code, to_line=LR, to_pre=0, to_pre_lo=0, tj=B2 * R, ti=2,
                tab=tab.base + tab.rho2t + plan2.t1(), od=1, od_lo=1, tabmod=1, inner=B2 * W // 4, total=e.columns * B2 * W // 4))
            src2 = e.code
        else:
            radix_stage[nb, nb, False, 4](ctx, arena, Radix(
                src=src, so_line=LR, so_pre=B2 * R, so_pre_lo=0, sk=B2 * na * R, si=2,
                dst=e.code, to_line=LR, to_pre=B2 * R, to_pre_lo=0, tj=B2 * na * R, ti=2,
                tab=tab.base + tab.rho2t + plan2.t1(), od=na, od_lo=1, tabmod=1, inner=B2 * W // 4, total=e.columns * na * B2 * W // 4))
            radix_stage[na, na, False, 4](ctx, arena, Radix(
                src=e.code, so_line=LR, so_pre=B2 * na * R, so_pre_lo=0, sk=B2 * R, si=2,
                dst=e.ctmp, to_line=LR, to_pre=B2 * R, to_pre_lo=0, tj=B2 * nb * R, ti=2,
                tab=tab.base + tab.rho2t + plan2.ta(), od=nb, od_lo=1, tabmod=nb, inner=B2 * W // 4, total=e.columns * nb * B2 * W // 4))
            src2 = e.ctmp
    else:
        src2 = src
    comptime k3 = k_to_stored[p]
    ctx.enqueue_function[k3](arena.buf, Buf[2](src2), Buf[1](e.stored), cols,
                             grid_dim=grid(n_grid // 2), block_dim=BACKEND.block)
    pack[p](ctx, arena, e)


def pack[p: Params](ctx: DeviceContext, arena: Arena, e: EncLayout) raises:
    """stored -> packed. Also the entry point of the quotient tree, which writes `stored` directly."""
    comptime k4 = k_pack[p]
    ctx.enqueue_function[k4](arena.buf, Buf[1](e.stored), Buf[4](e.packed), Int32(e.columns),
                             grid_dim=grid(e.columns * p.N() // 4), block_dim=BACKEND.block)


def rs_encode[p: Params, mask: Int = 31](ctx: DeviceContext, arena: Arena, e: EncLayout, tab: TableLayout) raises:
    """Level 1: packed -> code on the profile's domain. `mask` selects passes for the bench only:
    1 gather, 16 the 2-adic stages, 2 a first odd stage alone, 4 the last odd stages (fused pair or single)."""
    rs_encode_on[mask](ctx, arena, e.packed, e.etmp, e.code, e.columns, p.N() // 4, p.L0, p.m_cosets, tab.rs)


def rs_encode_on[mask: Int = 31](ctx: DeviceContext, arena: Arena,
                                 src: Int, etmp: Int, code: Int, columns: Int, K: Int, L0: Int, m: Int, rs: RsTables) raises:
    """src (K, columns, 4) -> code (m L0, columns, 4) on the domain of `rs`: pass A, then the radix
    stages of M, per coset. etmp holds (L0, columns, 4)."""
    var b = two_adic(L0)
    var M = L0 >> b
    var F5: Int
    var F7: Int
    var F9: Int
    F5, F7, F9 = rs_factors(M)
    if F5 * F7 * F9 != M or M == 1:
        raise Error("odd part of L0 must divide 315 and be > 1")
    var cols = Int32(columns)
    var gx = ceildiv(columns, CW)
    var ruri = rs.base + rs.ruri
    var b1 = _adic_split(b)[0]
    var Q = (K + M - 1) // M
    var minv = 0                                  # M^-1 mod B1
    for x in range(1 << b1):
        if (M * x) & ((1 << b1) - 1) == 1:
            minv = x
    for k in range(m):
        var twist = rs.base + rs.twist + k * K * 4 if m > 1 else 0
        var code_k = code + k * L0 * columns * 4
        comptime if mask & 1:
            if b - b1 > 3:
                raise Error("unsupported gather radix")
            comptime for B2 in [1, 2, 4, 8]:
                if (1 << (b - b1)) == B2:
                    ctx.enqueue_function[k_rs_gather[B2]](arena.buf, Buf[4](src), Buf[4](etmp), Buf[4](rs.base + rs.ga), Buf[1](rs.base + rs.crt),
                                                          Buf[4](twist), cols, Int32(K), Int32(b), Int32(M), Int32(Q), Int32(minv),
                                                          Int32(1 if m > 1 else 0),
                                                          grid_dim=(gx, ceildiv(M << b1, RW)), block_dim=(CW, RW))
        comptime if mask & 16:
            if b1 == 6:
                ctx.enqueue_function[k_rs_stage64](arena.buf, Buf[4](etmp), Buf[4](rs.base + rs.ga), cols, Int32(b), Int32(M),
                                                   grid_dim=(gx, M << (b - 6)), block_dim=(CW, 8))
            else:
                var S = b1
                while S > 0:
                    var lr = min(3, S)
                    _stage2(ctx, arena, etmp, rs.base + rs.ga, cols, b, M, S, lr, gx)
                    S -= lr
        var n_odd = (1 if F5 > 1 else 0) + (1 if F7 > 1 else 0) + (1 if F9 > 1 else 0)
        comptime if mask & 2:                          # a first odd stage alone when there are three
            if n_odd == 3:
                _stage[5](ctx, arena, etmp, rs.base + rs.w5, code_k, ruri, cols, b, M, F7 * F9)
        comptime if mask & 4:                          # the last two odd stages fused, or the only one
            if n_odd == 1:
                if F5 > 1:
                    _stage[5](ctx, arena, etmp, rs.base + rs.w5, code_k, ruri, cols, b, M, 1)
                elif F7 > 1:
                    _stage[7](ctx, arena, etmp, rs.base + rs.w7, code_k, ruri, cols, b, M, 1)
                elif F9 == 9:
                    _stage[9](ctx, arena, etmp, rs.base + rs.w9, code_k, ruri, cols, b, M, 1)
                else:
                    _stage[3](ctx, arena, etmp, rs.base + rs.w9, code_k, ruri, cols, b, M, 1)
            elif F9 == 1:
                _pair[5, 7](ctx, arena, etmp, rs.base + rs.w5, rs.base + rs.w7, code_k, ruri, cols, b, M)
            else:
                var wa = rs.base + (rs.w7 if F7 > 1 else rs.w5)
                comptime for ra in [5, 7]:
                    comptime for rb in [3, 9]:
                        if (F7 > 1) == (ra == 7) and F9 == rb:
                            _pair[ra, rb](ctx, arena, etmp, wa, rs.base + rs.w9, code_k, ruri, cols, b, M)


def _stage2(ctx: DeviceContext, arena: Arena, etmp: Int, ga: Int, cols: Int32, b: Int, M: Int, S: Int, lr: Int, gx: Int) raises:
    """Dispatch the radix 2^lr of one 2-adic step to a comptime one."""
    comptime for r in [2, 4, 8]:
        if (1 << lr) == r:
            ctx.enqueue_function[k_rs_stage2[r]](arena.buf, Buf[4](etmp), Buf[4](ga), cols, Int32(b), Int32(M), Int32(S),
                                                 grid_dim=(gx, ceildiv((M << b) // r, RW)), block_dim=(CW, RW))
            return
    raise Error("unsupported 2-adic radix")


def _pair[ra: Int, rb: Int](ctx: DeviceContext, arena: Arena, etmp: Int, wa: Int, wb: Int, code_k: Int, ruri: Int,
                           cols: Int32, b: Int, M: Int) raises:
    ctx.enqueue_function[k_rs_stage_pair[ra, rb]](arena.buf, Buf[4](etmp), Buf[4](wa), Buf[4](wb), Buf[4](code_k), Buf[1](ruri),
                                                  cols, Int32(b), Int32(M),
                                                  grid_dim=(ceildiv(Int(cols), CW), (1 << b) * (M // (ra * rb))), block_dim=(CW, max(ra, rb)))


def _stage[r: Int](ctx: DeviceContext, arena: Arena, etmp: Int, wr: Int, code_k: Int, ruri: Int,
                   cols: Int32, b: Int, M: Int, stride: Int) raises:
    """Dispatch the runtime stride (a product of the later radices) to a comptime one."""
    comptime for st in [1, 3, 7, 9, 21, 63]:
        if stride == st:
            ctx.enqueue_function[k_rs_stage[r, st]](arena.buf, Buf[4](etmp), Buf[4](wr), Buf[4](code_k), Buf[1](ruri), cols, Int32(b), Int32(M),
                                                    grid_dim=(ceildiv(Int(cols), CW * (2 if r == 5 else 1)), ceildiv((1 << b) * (M // r), RW)), block_dim=(CW, RW))
            return
    raise Error("unsupported radix stride")
