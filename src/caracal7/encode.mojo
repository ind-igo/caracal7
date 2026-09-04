"""Level-1 encoder: trace -> coeff -> stored -> packed -> code (docs/design.md sections 3, 4; spec 9.1).

Every kernel is rung 1 of the ladder: one thread per output, correct against the scalar
references in tests/test_encode.mojo, measured in bench/bench_encode.mojo. Kernels take the arena
base pointer plus Int64 byte offsets (rule 6) and are parameterized on Params (rule 3).

Buffers (bytes; shapes slowest ... fastest):
    trace   (column, x2, x1)          F
    ctmp    (column, x2, k1, 2)       F2, after the axis-1 inverse DFT
    coeff   (column, k2, k1, 2)       F2 monomial coefficients
    stored  (column, slot)            F, Frobenius-real slots (t, x1', x2, r)
    packed  (column, i, 4)            F4, coordinate basis (1, i, j, ij)
    etmp    (column, t1, lin, 4)      F4, RS intermediate; lin = d5*63 + d7*9 + d9
    code    (s, column, 4)            F4, leaf-major, leaf s is the point g^s

RS encode of the K = N/4 message symbols on the order-L0 subgroup, L0 = 2^b * 315:
    Good-Thomas across 2^b x 315, then 5 x 7 x 9 inside 315, no twiddles between stages.
    input index  i  <-> (i mod 2^b, i mod 315), i mod 315 <-> (i mod 5, i mod 7, i mod 9)
    output index s  =  (315 t1 + 2^b t2) mod L0,  t2 = (63 t5 + 45 t7 + 35 t9) mod 315
    pass A       Y[t1, lin] = sum_{i = i2 + 315 q < K} x_i gA^(t1 (i mod 2^b)),   i2 = crt(lin)
    stage r      r-point DFT along digit d_r of lin, in place, twiddles w_r^(t k)
    stage 9 also scatters to the leaf-major `code`.
"""

from std.math import ceildiv
from std.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext

from caracal7.field import F2, F4, f_add, f_mul, ext_mul
from caracal7.params import Params
from caracal7.tables import TableLayout, two_adic, modinv
from caracal7.arena import Bump

comptime BLOCK = 256
comptime M_ODD = 315                # ponytail: L0 = 2^b * 315 only; other divisors of 315 when a profile needs them
comptime E5 = modinv(63, 5)         # CRT reconstruction coefficients inside 315
comptime E7 = modinv(45, 7)
comptime E9 = modinv(35, 9)


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
        comptime assert p.L0 >> two_adic(p.L0) == M_ODD, "L0 must be 2^b * 315"
        comptime assert p.m_cosets == 1, "coset twist not implemented"   # ponytail: add the g_k^i twist with m_cosets > 1
        self.columns = columns
        self.trace = bump.alloc(columns * p.N())
        self.ctmp = bump.alloc(columns * p.N() * 2)
        self.coeff = bump.alloc(columns * p.N() * 2)
        self.stored = bump.alloc(columns * p.N())
        self.packed = bump.alloc(columns * p.N())
        self.etmp = bump.alloc(columns * p.L0 * 4)
        self.code = bump.alloc(p.L0 * columns * 4)


@always_inline
def _gid() -> Int:
    return Int(block_idx.x * block_dim.x + thread_idx.x)


@always_inline
def _ld[w: SIMDLength](base: Pointer[UInt8, MutAnyOrigin], off: Int) -> SIMD[DType.uint8, w]:
    return base.unsafe_load[width=w](off)


@always_inline
def _st[w: SIMDLength](base: Pointer[UInt8, MutAnyOrigin], off: Int, v: SIMD[DType.uint8, w]):
    base.unsafe_store[width=w](off, v)


# ---- idft2: inverse 2D DFT over F2, one dense pass per axis ----
# ponytail: dense O(h^2) per axis; mixed-radix stages (spec 10.2) when h_l grows past a few hundred.

def k_idft_axis1[p: Params](base: Pointer[UInt8, MutAnyOrigin], trace: Int64, ctmp: Int64, winv1: Int64, columns: Int32):
    """ctmp[c, x2, k1] = sum_t1 Winv1[k1, t1] * trace[c, x2, t1]   (F2 x F)."""
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    var gid = _gid()
    if gid >= Int(columns) * h2 * h1:
        return
    var k1 = gid % h1
    var line = gid // h1                    # (c, x2)
    var acc = F2(0)
    for t1 in range(h1):
        var w = _ld[2](base, Int(winv1) + (k1 * h1 + t1) * 2)
        var v = base[Int(trace) + line * h1 + t1]
        acc = f_add(acc, f_mul(w, F2(v)))
    _st(base, Int(ctmp) + (line * h1 + k1) * 2, acc)


def k_idft_axis2[p: Params](base: Pointer[UInt8, MutAnyOrigin], ctmp: Int64, coeff: Int64, winv2: Int64, columns: Int32):
    """coeff[c, k2, k1] = sum_t2 Winv2[k2, t2] * ctmp[c, t2, k1]   (F2 x F2)."""
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    var gid = _gid()
    if gid >= Int(columns) * h2 * h1:
        return
    var k1 = gid % h1
    var k2 = (gid // h1) % h2
    var c = gid // (h1 * h2)
    var acc = F2(0)
    for t2 in range(h2):
        var w = _ld[2](base, Int(winv2) + (k2 * h2 + t2) * 2)
        var v = _ld[2](base, Int(ctmp) + ((c * h2 + t2) * h1 + k1) * 2)
        acc = f_add(acc, ext_mul[1](w, v))
    _st(base, Int(coeff) + ((c * h2 + k2) * h1 + k1) * 2, acc)


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


def k_to_stored[p: Params](base: Pointer[UInt8, MutAnyOrigin], coeff: Int64, stored: Int64, rho1: Int64, rho2: Int64, columns: Int32):
    """stored[c, slot] = coord of c_x(r) = sum_y coeff[c, x2 + 2^a2 y2, x1 + 2^a1 y1] rho1^(y1 r1) rho2^(y2 r2)."""
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    comptime N = p.N()
    var gid = _gid()
    if gid >= Int(columns) * N:
        return
    var c = gid // N
    var slot = gid % N
    var x1: Int
    var x2: Int
    var r: Int
    var coord: Int
    x1, x2, r, coord = slot_target[p](slot)
    var r1 = r % p.m1
    var r2 = r // p.m1
    var acc = F2(0)
    for y2 in range(p.m2):
        var s2 = base[Int(rho2) + (y2 * r2) % p.m2]
        for y1 in range(p.m1):
            var s1 = base[Int(rho1) + (y1 * r1) % p.m1]
            var k1 = x1 + (1 << p.a1) * y1
            var k2 = x2 + (1 << p.a2) * y2
            var v = _ld[2](base, Int(coeff) + ((c * h2 + k2) * h1 + k1) * 2)
            acc = f_add(acc, f_mul(v, F2(f_mul(SIMD[DType.uint8, 1](s1), SIMD[DType.uint8, 1](s2))[0])))
    base[Int(stored) + gid] = acc[coord]


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


def k_pack[p: Params](base: Pointer[UInt8, MutAnyOrigin], stored: Int64, packed: Int64, columns: Int32):
    comptime N = p.N()
    var gid = _gid()
    if gid >= Int(columns) * (N // 4):
        return
    var c = gid // (N // 4)
    var i = gid % (N // 4)
    var v = F4(0)
    comptime for j in range(4):
        v[j] = base[Int(stored) + c * N + pack_slot[p](i, j)]
    _st(base, Int(packed) + gid * 4, v)


# ---- rs_encode ----

@always_inline
def _crt315(lin: Int) -> Int:
    var d5 = lin // 63
    var d7 = (lin // 9) % 7
    var d9 = lin % 9
    return (d5 * 63 * E5 + d7 * 45 * E7 + d9 * 35 * E9) % M_ODD


def k_rs_pass_a[p: Params](base: Pointer[UInt8, MutAnyOrigin], packed: Int64, etmp: Int64, ga: Int64, columns: Int32):
    """etmp[c, t1, lin] = sum over message symbols i = crt(lin) + 315 q of x_i * gA^(t1 * (i mod 2^b))."""
    comptime b = two_adic(p.L0)
    comptime K = p.N() // 4
    var gid = _gid()
    if gid >= Int(columns) * p.L0:
        return
    var c = gid // p.L0
    var t1 = (gid % p.L0) // M_ODD
    var lin = gid % M_ODD
    var i = _crt315(lin)
    var acc = F4(0)
    while i < K:
        var x = _ld[4](base, Int(packed) + (c * K + i) * 4)
        var w = _ld[4](base, Int(ga) + ((t1 * (i & ((1 << b) - 1))) & ((1 << b) - 1)) * 4)
        acc = f_add(acc, ext_mul[2](x, w))
        i += M_ODD
    _st(base, Int(etmp) + gid * 4, acc)


def k_rs_stage[p: Params, r: Int, stride: Int, final: Bool](
    base: Pointer[UInt8, MutAnyOrigin], etmp: Int64, wr: Int64, code: Int64, columns: Int32
):
    """In-place r-point DFT along one digit of lin; the last stage scatters into `code`."""
    comptime b = two_adic(p.L0)
    comptime LINES = M_ODD // r
    var gid = _gid()
    if gid >= Int(columns) * (1 << b) * LINES:
        return
    var l = gid % LINES
    var row = gid // LINES                       # (c, t1)
    var c = row >> b
    var t1 = row & ((1 << b) - 1)
    var first = row * M_ODD + (l // stride) * r * stride + l % stride
    var xs = InlineArray[F4, r](fill=F4(0))
    comptime for k in range(r):
        xs[k] = _ld[4](base, Int(etmp) + (first + k * stride) * 4)
    comptime for t in range(r):
        var acc = F4(0)
        comptime for k in range(r):
            acc = f_add(acc, ext_mul[2](_ld[4](base, Int(wr) + (t * r + k) * 4), xs[k]))
        comptime if final:
            var lin = (l // stride) * r * stride + l % stride + t * stride
            var t2 = (63 * (lin // 63) + 45 * ((lin // 9) % 7) + 35 * (lin % 9)) % M_ODD
            var s = (M_ODD * t1 + (1 << b) * t2) % p.L0
            _st(base, Int(code) + (s * Int(columns) + c) * 4, acc)
        else:
            _st(base, Int(etmp) + (first + t * stride) * 4, acc)


# ---- host orchestration: enqueue the whole encoder on one stream ----

@always_inline
def grid(n: Int) -> Int:
    return ceildiv(n, BLOCK)


def encode[p: Params](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin], e: EncLayout, tab: TableLayout) raises:
    to_packed[p](ctx, base, e, tab)
    rs_encode[p](ctx, base, e, tab)


def to_packed[p: Params](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin], e: EncLayout, tab: TableLayout) raises:
    """trace -> coeff -> stored -> packed."""
    var cols = Int32(e.columns)
    var n_grid = e.columns * p.N()
    comptime k1 = k_idft_axis1[p]
    ctx.enqueue_function[k1](base, Int64(e.trace), Int64(e.ctmp), Int64(tab.base + tab.winv1), cols,
                             grid_dim=grid(n_grid), block_dim=BLOCK)
    comptime k2 = k_idft_axis2[p]
    ctx.enqueue_function[k2](base, Int64(e.ctmp), Int64(e.coeff), Int64(tab.base + tab.winv2), cols,
                             grid_dim=grid(n_grid), block_dim=BLOCK)
    comptime k3 = k_to_stored[p]
    ctx.enqueue_function[k3](base, Int64(e.coeff), Int64(e.stored), Int64(tab.base + tab.rho1), Int64(tab.base + tab.rho2), cols,
                             grid_dim=grid(n_grid), block_dim=BLOCK)
    comptime k4 = k_pack[p]
    ctx.enqueue_function[k4](base, Int64(e.stored), Int64(e.packed), cols,
                             grid_dim=grid(n_grid // 4), block_dim=BLOCK)


def rs_encode[p: Params, mask: Int = 15](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin], e: EncLayout, tab: TableLayout) raises:
    """packed -> code: pass A, then the 5, 7, 9 stages. `mask` selects passes for the bench only."""
    comptime b = two_adic(p.L0)
    var cols = Int32(e.columns)
    comptime if mask & 1:
        comptime k5 = k_rs_pass_a[p]
        ctx.enqueue_function[k5](base, Int64(e.packed), Int64(e.etmp), Int64(tab.base + tab.ga), cols,
                                 grid_dim=grid(e.columns * p.L0), block_dim=BLOCK)
    comptime if mask & 2:
        comptime s5 = k_rs_stage[p, 5, 63, False]
        ctx.enqueue_function[s5](base, Int64(e.etmp), Int64(tab.base + tab.w5), Int64(e.code), cols,
                                 grid_dim=grid(e.columns * (1 << b) * (M_ODD // 5)), block_dim=BLOCK)
    comptime if mask & 4:
        comptime s7 = k_rs_stage[p, 7, 9, False]
        ctx.enqueue_function[s7](base, Int64(e.etmp), Int64(tab.base + tab.w7), Int64(e.code), cols,
                                 grid_dim=grid(e.columns * (1 << b) * (M_ODD // 7)), block_dim=BLOCK)
    comptime if mask & 8:
        comptime s9 = k_rs_stage[p, 9, 1, True]
        ctx.enqueue_function[s9](base, Int64(e.etmp), Int64(tab.base + tab.w9), Int64(e.code), cols,
                                 grid_dim=grid(e.columns * (1 << b) * (M_ODD // 9)), block_dim=BLOCK)
