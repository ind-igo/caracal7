"""Level-1 encoder: trace -> coeff -> stored -> packed -> code (docs/design.md sections 3, 4; spec 9.1).

Every kernel is rung 1 of the ladder: one thread per output, correct against the scalar
references in tests/test_encode.mojo, measured in bench/bench_encode.mojo. Kernels take the arena
base pointer plus width-typed regions of it (rule 2, bytes.mojo) and are parameterized on Params (rule 3).

Buffers (bytes; shapes slowest ... fastest):
    trace   (column, x2, x1)          F
    ctmp    (column, x2, k1, 2)       F2, after the axis-1 inverse DFT
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
    Good-Thomas across 2^b x M, then the radices of M (5, 7, 9 or 3) inside M, no twiddles between stages.
    input index  i  <-> (i mod 2^b, i mod M), i mod M <-> (i mod 5, i mod 7, i mod 9)
    output index s  =  (M t1 + 2^b t2) mod L0,  t2 = sum_r (M / r) t_r mod M
    pass A       Y[t1, lin] = sum_{i = i2 + M q < K} x_i gA^(t1 (i mod 2^b)),   i2 = crt(lin)
                 an F4 GEMM per lin on the skeleton: A = the wa table (2^b, Q), B = the message rows (Q, columns)
    stage r      r-point DFT along digit d_r of lin, in place, twiddles w_r^(t k)
    the last stage also scatters to the leaf-major `code`, coset k at s + k L0.
"""

from std.math import ceildiv
from std.gpu import thread_idx, block_idx, block_dim, global_idx
from max.gpu.host import DeviceContext

from caracal7.core.field import F2, F4, f_add, f_mul, ext_mul, f4_mac_wide, f_reduce_signed, F4_MAC_MAX
from caracal7.core.params import Params
from caracal7.core.tables import TableLayout, RsTables, two_adic, rs_factors
from caracal7.core.arena import Bump
from caracal7.core.bytes import Base, Buf, u16
from caracal7.core.arena import Arena
from caracal7.core.backend import BACKEND, F4_TILE, Operands, Loader4, Strided, Bytes, launch_gemm_f2, launch_gemm_f4, strided

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


# ---- idft2: inverse 2D DFT over F2, one dense pass per axis ----
# ponytail: dense O(h^2) per axis on the F2 skeleton; mixed-radix stages (spec 10.2) when h_l grows past a few hundred.

def idft2[p: Params](ctx: DeviceContext, arena: Arena, trace: Int, ctmp: Int, coeff: Int,
                     columns: Int, tab: TableLayout) raises:
    """trace (column, x2, x1) F -> coeff (column, k2, k1, 2): inverse DFT per axis, two skeleton launches.
    Axis 1: C[k1, line] = sum_t1 Winv1[k1, t1] trace[line, t1]; axis 2 per column: C[k2, k1] = sum_t2 Winv2[k2, t2] ctmp[t2, k1]."""
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    var o1 = strided(a=tab.base + tab.winv1, sa_m=h1 * 2, sa_k=2, b=trace, sb_k=1, sb_hi=h1, sb_lo=0,
                     c=ctmp, sc_m=2, sc_hi=h1 * 2, sc_lo=0)
    launch_gemm_f2[BACKEND, BACKEND.tile, Bytes, 1](ctx, arena, o1, h1, columns * h2, h1)
    var o2 = strided(a=tab.base + tab.winv2, sa_m=h2 * 2, sa_k=2, b=ctmp, sb_k=h1 * 2, sb_hi=2, sb_lo=0,
                     c=coeff, sc_m=h1 * 2, sc_hi=2, sc_lo=0, sb_z=h2 * h1 * 2, sc_z=h2 * h1 * 2)
    launch_gemm_f2[BACKEND, BACKEND.tile, Strided, 1](ctx, arena, o2, h2, h1, h2, batch=columns)


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


def k_to_stored[p: Params](base: Base, coeff: Buf[2], stored: Buf[1], rho1: Buf[1], rho2: Buf[1], columns: Int32):
    """stored[c, slot] = coord of c_x(r) = sum_y coeff[c, x2 + 2^a2 y2, x1 + 2^a1 y1] rho1^(y1 r1) rho2^(y2 r2)."""
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    comptime N = p.N()
    var gid = Int(global_idx.x)
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
        var s2 = base[unsafe_offset=rho2.at((y2 * r2) % p.m2)]
        for y1 in range(p.m1):
            var s1 = base[unsafe_offset=rho1.at((y1 * r1) % p.m1)]
            var k1 = x1 + (1 << p.a1) * y1
            var k2 = x2 + (1 << p.a2) * y2
            var v = coeff.load(base, (c * h2 + k2) * h1 + k1)
            acc = f_add(acc, f_mul(v, F2(f_mul(SIMD[DType.uint8, 1](s1), SIMD[DType.uint8, 1](s2))[0])))
    base[unsafe_offset=stored.at(gid)] = acc[coord]


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
        v[j] = base[unsafe_offset=stored.at(c * N + pack_slot[p](i, j))]
    packed.store(base, gid, v)


# ---- rs_encode ----

@always_inline
def _rs_thread(columns: Int32) -> Tuple[Int, Int, Bool]:
    """2D launch: block (CW, RW); x walks columns, y walks (t1, line) rows. Returns (c, row, valid)."""
    var c = Int(block_idx.x) * CW + Int(thread_idx.x)
    var row = Int(block_idx.y) * RW + Int(thread_idx.y)
    return (c, row, c < Int(columns))


struct RsRows(Loader4):
    """Pass A's B operand: row q of line z is message symbol i = crt(z) + M q (zero past K), twisted
    by gamma4^(k i) on coset k. Operand fields: b = src (i, column, 4), sb_z = the row stride, sb_hi = 4,
    sb_k = M (a count, not a stride), aux0 = crt, aux1 = twist or -1, aux2 = K."""
    @staticmethod
    def load(base: Base, o: Operands, k: Int, n_hi: Int, n_lo: Int, z: Int) -> F4:
        var i = u16(base, Int(o.aux0) + z * 2) + Int(o.sb_k) * k
        if i >= Int(o.aux2):
            return F4(0)
        var x = base.unsafe_load[width=4](Int(o.b) + i * Int(o.sb_z) + n_hi * Int(o.sb_hi))
        if o.aux1 >= 0:
            x = ext_mul[2](x, base.unsafe_load[width=4](Int(o.aux1) + i * 4))
        return x


def rs_pass_a(ctx: DeviceContext, arena: Arena, src: Int, etmp: Int, rs: RsTables,
              twist: Int, columns: Int, K: Int, b: Int, M: Int) raises:
    """etmp[lin, t1, c] = sum_q wa[lin, t1, q] x_{crt(lin) + M q} twist: one F4 GEMM per line,
    M = 2^b, N = columns, K = Q, batch = the odd part."""
    var o = strided(a=rs.base + rs.wa, sa_m=rs.Q * 4, sa_k=4, sa_z=(1 << b) * rs.Q * 4,
                    b=src, sb_k=M, sb_hi=4, sb_lo=0, sb_z=columns * 4,
                    c=etmp, sc_m=columns * 4, sc_hi=4, sc_lo=0, sc_z=(1 << b) * columns * 4)
    o.aux0 = Int64(rs.base + rs.crt)
    o.aux1 = Int64(twist)
    o.aux2 = Int64(K)
    launch_gemm_f4[BACKEND, F4_TILE, RsRows, 1](ctx, arena, o, 1 << b, columns, rs.Q, batch=M)


def k_rs_stage[r: Int, stride: Int](base: Base, etmp: Buf[4], wr: Buf[4], code: Buf[4], ruri: Buf[1],
                                    columns: Int32, b: Int32, M: Int32):
    """In-place r-point DFT along one digit of lin; the last stage (stride 1) scatters into `code`.
    The stride is comptime: a runtime division here costs a quarter of the encoder."""
    comptime final = stride == 1
    var c: Int
    var rest: Int
    var ok: Bool
    c, rest, ok = _rs_thread(columns)
    var bb = Int(b)
    var Mi = Int(M)
    comptime st = stride
    if not ok or rest >= (Mi // r) << bb:
        return
    var t1 = rest & ((1 << bb) - 1)
    var l = rest >> bb
    var first_lin = (l // st) * r * st + l % st
    var col_off = t1 * Int(columns) + c        # + lin * 2^b * columns
    var xs = InlineArray[F4, r](fill=F4(0))
    comptime for k in range(r):
        xs[k] = etmp.load(base, ((first_lin + k * st) << bb) * Int(columns) + col_off)
    comptime assert r <= F4_MAC_MAX
    comptime for t in range(r):
        var wide = SIMD[DType.int32, 4](0)
        comptime for k in range(r):
            f4_mac_wide(wide, wr.load(base, t * r + k), xs[k])
        var acc = f_reduce_signed(wide)
        var lin = first_lin + t * st
        comptime if final:
            var s = Mi * t1 + (u16(base, ruri.at(lin * 2)) << bb)    # < 2 L0
            if s >= (Mi << bb):
                s -= (Mi << bb)
            code.store(base, s * Int(columns) + c, acc)
        else:
            etmp.store(base, (lin << bb) * Int(columns) + col_off, acc)


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
    comptime k3 = k_to_stored[p]
    ctx.enqueue_function[k3](arena.buf, Buf[2](e.coeff), Buf[1](e.stored), Buf[1](tab.base + tab.rho1), Buf[1](tab.base + tab.rho2), cols,
                             grid_dim=grid(n_grid), block_dim=BACKEND.block)
    pack[p](ctx, arena, e)


def pack[p: Params](ctx: DeviceContext, arena: Arena, e: EncLayout) raises:
    """stored -> packed. Also the entry point of the quotient tree, which writes `stored` directly."""
    comptime k4 = k_pack[p]
    ctx.enqueue_function[k4](arena.buf, Buf[1](e.stored), Buf[4](e.packed), Int32(e.columns),
                             grid_dim=grid(e.columns * p.N() // 4), block_dim=BACKEND.block)


def rs_encode[p: Params, mask: Int = 15](ctx: DeviceContext, arena: Arena, e: EncLayout, tab: TableLayout) raises:
    """Level 1: packed -> code on the profile's domain. `mask` selects passes for the bench only."""
    rs_encode_on[mask](ctx, arena, e.packed, e.etmp, e.code, e.columns, p.N() // 4, p.L0, p.m_cosets, tab.rs)


def rs_encode_on[mask: Int = 15](ctx: DeviceContext, arena: Arena,
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
    for k in range(m):
        var twist = rs.base + rs.twist + k * K * 4 if m > 1 else -1
        var code_k = code + k * L0 * columns * 4
        comptime if mask & 1:
            rs_pass_a(ctx, arena, src, etmp, rs, twist, columns, K, b, M)
        comptime if mask & 2:
            if F5 > 1:
                _stage[5](ctx, arena, etmp, rs.base + rs.w5, code_k, ruri, cols, b, M, F7 * F9, gx)
        comptime if mask & 4:
            if F7 > 1:
                _stage[7](ctx, arena, etmp, rs.base + rs.w7, code_k, ruri, cols, b, M, F9, gx)
        comptime if mask & 8:
            if F9 == 9:
                _stage[9](ctx, arena, etmp, rs.base + rs.w9, code_k, ruri, cols, b, M, 1, gx)
            elif F9 == 3:
                _stage[3](ctx, arena, etmp, rs.base + rs.w9, code_k, ruri, cols, b, M, 1, gx)


def _stage[r: Int](ctx: DeviceContext, arena: Arena, etmp: Int, wr: Int, code_k: Int, ruri: Int,
                   cols: Int32, b: Int, M: Int, stride: Int, gx: Int) raises:
    """Dispatch the runtime stride (a product of the later radices) to a comptime one."""
    comptime for st in [1, 3, 7, 9, 21, 63]:
        if stride == st:
            ctx.enqueue_function[k_rs_stage[r, st]](arena.buf, Buf[4](etmp), Buf[4](wr), Buf[4](code_k), Buf[1](ruri), cols, Int32(b), Int32(M),
                                                    grid_dim=(gx, ceildiv((1 << b) * (M // r), RW)), block_dim=(CW, RW))
            return
    raise Error("unsupported radix stride")
