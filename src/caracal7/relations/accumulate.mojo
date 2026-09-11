"""The Z stage (spec 10 steps 4-6, 6.2, 7.1-7.3): factors, batched inversion, the running product
per chain, and Z2 across chains, for permutation accumulators on witness columns.

Accumulator descriptor (ACC bytes, u16 little-endian): z_col [0, 2) the global index of coordinate
column 0 of Z; w_num [2, 4), w_den [4, 6); num columns [6, 22); den columns [22, 38); kind u8 [38]
(KIND_PERM, KIND_LOOKUP, KIND_HORNER); table id u8 [39] (lookup: the index into Shape.tables); family u16 [40, 42).
A KIND_HORNER descriptor (polynomial-mulmod 5, the spec's {start, ingest, scale, end}) reuses [2, 8): first
ingest entry u16 [2, 4), ingest count [4, 6), start u8 [6], scale u8 [7] (0 none, else element + 1). Its
ingest terms are the family entries themselves (ir.Families.horner), so the kernel and the residual read
one definition: R(1, x2) = start, R(omega1 x1, x2) = scale R - sum_entries coef chal c(omega1^k1 x1, x2)
(the entries carry the transition's sign). No Z2, N, or D: chain ends meet in chain-end families
(smallgrid.mojo). ponytail: one thread per chain for the scan (h1 steps); the segmented affine scan of
10.1 when a client grid measures it.
fp(c) = sum_j c_j b_j is the fingerprint of 6.1 (b_j the unit vector j of E: no multiplication, the
column bytes are the coordinates). The factors per kind:
    KIND_PERM (6.2)     N = gamma + fp(num),  D = gamma + fp(den)
    KIND_LOOKUP (6.3)   num are the record columns f, den the sorted copy s (sort.mojo), both width w_num:
                        N = (1 + beta) (delta + fp(f)),  D = (1 + beta) delta + fp(s)(row) + beta fp(s)(row + 1)
                        with row + 1 taken cyclically. In row-major order row + 1 is the next row both
                        inside a chain and across the chain end, so d_end already holds the chain-end
                        factor of 6.3 and the small grid is unchanged. D at the last row is the wrap to
                        row 0: no check reads it (the transition is gated off at x1 = e1, the small grid
                        at x2 = e2, and the boundary Z2(e2) Z(e1, e2) N(e1, e2) = C_T has no D), but
                        the line d_end must be the same degree < h2 polynomial the verifier evaluates
                        from the openings at (e1, z2) and (1, omega2 z2).
    TODO(memory): KIND_MEMORY (6.4) adds the (addr, ts, value) columns and its own factor pair here;
                  it waits for a profile with memory (configuration.md 3).
The stage-1 challenges arrive as the element list of ir.mojo: beta, delta, gamma, then the derivation table's rows;
the factor kernels read 1 + beta and (1 + beta) delta at 3 and 4 (Shape checks the table starts with standard_chals).

Buffers (bytes; slowest ... fastest), per accumulator:
    num, den     (row, e)     N and D per row, row = x2 h1 + x1
    zscratch     (row, e)     per segment: prefix products of D, then N / D; then the segment total at
                              its last row and the exclusive prefix of the segments at its first row
    zval         (row, e)     Z(x) with Z(1, x2) = 1 and Z(next) = Z N / D along the chain (7.2)
    chain_prod   (x2, e)      Z(e1, x2) N(e1, x2) / D(e1, x2), the whole chain's product
    n_end, d_end (x2, e)      N(e1, x2), D(e1, x2): the small grid's lines (smallgrid.mojo)
    z2           (x2 + 1, e)  Z2(1) = 1, Z2(omega2 x2) = Z2(x2) chain_prod(x2) (7.3, (W) for (P)); entry h2 is 1

Wiring (polynomial-mulmod "Wiring"): a copy constraint on the chain-end values of slot columns, PLONK-style.
Slot s (a Z block) on chain j has the value w = R(e1, omega2^j) and the id kappa_s omega2^j in F2 (kappa_s a
coset representative, so ids are distinct); sigma (Shape.sigma, F2 per slot and chain) is the public
permutation. A wiring product (WIRE record: two slots, family) is the Z2 line of the per-chain factors
    N = prod_s (w_s + beta_w id_s + gamma_w),  D = prod_s (w_s + beta_w sigma_s + gamma_w)
with beta_w, gamma_w the two wiring elements (squeezed after the derivation table). k_wire_factors writes
the four factor lines (n0, n1, d0, d1: (4, x2, e)) for the small grid and chain_prod = N / D for k_z2; a NONE
slot contributes 1. Products multiply across groups: the verifier closes them jointly on the public factors.

The scan is the two-level one of 10.1: segments of S = seg_len(h1) rows in parallel (N / S threads),
one thread per chain over the segment totals, a fix-up per row. ponytail: one thread for Z2 (h2 steps). A zero D (probability ~ 1 / |E|)
gives 0 from ext_inv0 for its segment, a zero chain product, and a proof the verifier rejects; no abort path.
"""

from std.math import ceildiv
from std.gpu import global_idx
from max.gpu.host import DeviceContext

from caracal7.core.field import F2, E, f_add, f_sub, f_mul, ext_mul, ext_pow, ext_embed, ext_inv0, ext_one, fp_ext_mul, fp_reduce, fp_canonical
from caracal7.core.params import Params
from caracal7.core.backend import BACKEND
from caracal7.core.bytes import Base, Buf, u16
from caracal7.relations.ir import ACC, ENTRY, WIRE, NONE, KIND_LOOKUP, CHAL, CHAL_ADD, CHAL_ONE, SAMPLED
from caracal7.core.arena import Arena, Bump



struct AccLayout(TrivialRegisterPassable):
    """Arena offsets of the Z stage. N, D and the scan scratch serve every accumulator in turn; Z is
    per accumulator; the grand-product lines are per product (accumulators first, then wiring products,
    `pi` counts both); the factor lines are per wiring product."""
    var num: Int            # (row, e)
    var den: Int
    var scratch: Int
    var zval: Int           # (accumulator, row, e)     Z values, the Z tree's trace before the coordinate split
    var chain_prod: Int     # (x2, e)
    var z2: Int             # (product, x2, e) + e      Z2 in the clear; one trailing 1 (k_z2)
    var n_end: Int          # (product, x2, e)          N(e1, x2), D(e1, x2)
    var d_end: Int
    var wlines: Int         # (wiring product, 4, x2, e)  the factor lines n0, n1, d0, d1 (k_wire_factors)
    var rows: Int           # N e bytes, one accumulator's Z
    var line: Int           # h2 e bytes, one line

    def __init__[p: Params](out self, mut bump: Bump, accumulators: Int, products: Int, wiring: Int):
        self.rows = p.N() * p.e
        self.line = p.h2() * p.e
        self.num = bump.alloc(self.rows)
        self.den = bump.alloc(self.rows)
        self.scratch = bump.alloc(self.rows)
        self.zval = bump.alloc(accumulators * self.rows)
        self.chain_prod = bump.alloc(self.line)
        self.z2 = bump.alloc(products * self.line + p.e)
        self.n_end = bump.alloc(accumulators * self.line)
        self.d_end = bump.alloc(accumulators * self.line)
        self.wlines = bump.alloc(wiring * 4 * self.line)

    def zval_at(self, k: Int) -> Int:
        return self.zval + k * self.rows

    def z2_at(self, pi: Int) -> Int:
        return self.z2 + pi * self.line

    def n_end_at(self, pi: Int) -> Int:
        return self.n_end + pi * self.line

    def d_end_at(self, pi: Int) -> Int:
        return self.d_end + pi * self.line

    def wline_at(self, g: Int, j: Int) -> Int:
        return self.wlines + (4 * g + j) * self.line


def k_derive_chals(base: Base, chals: Buf[16], table: Buf[1], rows: Int32):
    """One thread: element SAMPLED + i = op(a, b) per table row, in order (ir.derived_chals is the host twin)."""
    if global_idx.x != 0:
        return
    for i in range(Int(rows)):
        var ia = Int(table.load(base, i * CHAL + 1))
        var ib = Int(table.load(base, i * CHAL + 2))
        var a = ext_one[4]() if ia == CHAL_ONE else chals.load(base, ia)
        var b = ext_one[4]() if ib == CHAL_ONE else chals.load(base, ib)
        chals.store(base, SAMPLED + i, f_add(a, b) if Int(table.load(base, i * CHAL)) == CHAL_ADD else ext_mul[4](a, b))


def k_factors[p: Params](base: Base, trace: Buf[1], acc: Buf[1], chals: Buf[16], num: Buf[16], den: Buf[16]):
    """One thread per row: the factor pair of the descriptor's kind (module docstring)."""
    comptime N = p.N()
    var row = global_idx.x
    if row >= N:
        return
    var w_num = u16(base, acc.at(2))
    var w_den = u16(base, acc.at(4))
    var n: E
    var d: E
    if Int(acc.load(base, 38)) == KIND_LOOKUP:
        var nxt = row + 1 if row + 1 < N else 0
        var ff = E(0)
        var s0 = E(0)
        var s1 = E(0)
        for j in range(w_num):
            ff[j] = trace.load(base, u16(base, acc.at(6 + 2 * j)) * N + row)
            s0[j] = trace.load(base, u16(base, acc.at(22 + 2 * j)) * N + row)
            s1[j] = trace.load(base, u16(base, acc.at(22 + 2 * j)) * N + nxt)
        n = ext_mul[4](chals.load(base, 3), f_add(chals.load(base, 1), ff))
        d = f_add(f_add(chals.load(base, 4), s0), ext_mul[4](chals.load(base, 0), s1))
    else:
        var g = chals.load(base, 2)
        n = g
        d = g
        for j in range(w_num):
            n[j] = f_add(n[j], trace.load(base, u16(base, acc.at(6 + 2 * j)) * N + row))
        for j in range(w_den):
            d[j] = f_add(d[j], trace.load(base, u16(base, acc.at(22 + 2 * j)) * N + row))
    num.store(base, row, n)
    den.store(base, row, d)


def seg_len(h1: Int) -> Int:
    """Rows per scan segment: the largest of 16, 8, 4 dividing h1 (check() gives a1 >= 2)."""
    return 16 if h1 % 16 == 0 else (8 if h1 % 8 == 0 else 4)


def k_seg_scan[p: Params](base: Base, num: Buf[16], den: Buf[16], scratch: Buf[16], zval: Buf[16]):
    """One thread per segment of S rows inside a chain: batched inversion of D over the segment
    (prefix products, one inverse, backward pass), q = N / D, then the segment's exclusive prefix of
    q into zval and its total into scratch at the segment's last row."""
    comptime S = seg_len(p.h1())
    var seg = global_idx.x
    if seg >= p.N() // S:
        return
    var row0 = seg * S
    var acc = ext_one[4]()
    for i in range(S):
        acc = ext_mul[4](acc, den.load(base, row0 + i))
        scratch.store(base, row0 + i, acc)
    var inv = ext_inv0[4](acc)
    for i in range(S - 1, -1, -1):
        var pref = scratch.load(base, row0 + i - 1) if i > 0 else ext_one[4]()
        var inv_d = ext_mul[4](inv, pref)
        inv = ext_mul[4](inv, den.load(base, row0 + i))
        scratch.store(base, row0 + i, ext_mul[4](num.load(base, row0 + i), inv_d))
    var z = ext_one[4]()
    for i in range(S):
        zval.store(base, row0 + i, z)
        z = ext_mul[4](z, scratch.load(base, row0 + i))
    scratch.store(base, row0 + S - 1, z)


def k_chain_scan[p: Params](base: Base, num: Buf[16], den: Buf[16], scratch: Buf[16], chain_prod: Buf[16],
                            n_end: Buf[16], d_end: Buf[16]):
    """One thread per chain over its h1 / S segment totals (scratch, last row of each segment): the
    exclusive prefix goes to the segment's first row, the whole product to chain_prod."""
    comptime h1 = p.h1()
    comptime S = seg_len(h1)
    var x2 = global_idx.x
    if x2 >= p.h2():
        return
    var row0 = x2 * h1
    var z = ext_one[4]()
    for g in range(h1 // S):
        var total = scratch.load(base, row0 + g * S + S - 1)
        scratch.store(base, row0 + g * S, z)
        z = ext_mul[4](z, total)
    chain_prod.store(base, x2, z)
    n_end.store(base, x2, num.load(base, row0 + h1 - 1))
    d_end.store(base, x2, den.load(base, row0 + h1 - 1))


def k_seg_fixup[p: Params](base: Base, scratch: Buf[16], zval: Buf[16]):
    """One thread per row: Z = local prefix x the prefix of the segments before it."""
    comptime S = seg_len(p.h1())
    var row = global_idx.x
    if row >= p.N():
        return
    zval.store(base, row, ext_mul[4](zval.load(base, row), scratch.load(base, (row // S) * S)))


def k_z2[p: Params](base: Base, chain_prod: Buf[16], z2: Buf[16]):
    """One thread: Z2 across the chains, then Z2(omega2^h2) = Z2(1) = 1 at index h2 so the shifted
    line Z2(omega2 X2) of the small grid is the same buffer one element on."""
    if global_idx.x != 0:
        return
    var z = ext_one[4]().cast[DType.float32]()
    for x2 in range(p.h2()):
        z2.store(base, x2, fp_canonical(z))
        z = fp_reduce(fp_ext_mul[4](z, chain_prod.load(base, x2).cast[DType.float32]()))
    z2.store(base, p.h2(), ext_one[4]())


def k_wire_factors[p: Params](base: Base, zval: Buf[16], wire: Buf[1], sigma: Buf[1], slot0: Int32, columns_w: Int32, wchal: Buf[16],
                              ka: UInt8, kb: UInt8, kc: UInt8, kd: UInt8, wa: UInt8, wb: UInt8, chain_prod: Buf[16], lines: Buf[16]):
    """One thread per chain: the wiring factor lines and N / D (module docstring). (ka, kb), (kc, kd) are the
    two slots' coset representatives, (wa, wb) omega2."""
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    var j = global_idx.x
    if j >= h2:
        return
    var x = ext_pow[1](F2(wa, wb), j)
    var beta = wchal.load(base, 0)
    var gamma = wchal.load(base, 1)
    var n = ext_one[4]()
    var d = ext_one[4]()
    for s in range(2):
        var ns = ext_one[4]()
        var ds = ext_one[4]()
        var col = u16(base, wire.at(2 * s))
        if col != NONE:
            var w = zval.load(base, ((col - Int(columns_w)) // p.e) * p.N() + j * h1 + h1 - 1)
            var kappa = F2(ka, kb) if s == 0 else F2(kc, kd)
            var sg = F2(sigma.load(base, ((Int(slot0) + s) * h2 + j) * 2), sigma.load(base, ((Int(slot0) + s) * h2 + j) * 2 + 1))
            var wg = f_add(w, gamma)
            ns = f_add(wg, ext_mul[4](beta, ext_embed[4](ext_mul[1](kappa, x))))
            ds = f_add(wg, ext_mul[4](beta, ext_embed[4](sg)))
        lines.store(base, s * h2 + j, ns)
        lines.store(base, (2 + s) * h2 + j, ds)
        n = ext_mul[4](n, ns)
        d = ext_mul[4](d, ds)
    chain_prod.store(base, j, ext_mul[4](n, ext_inv0[4](d)))


def k_ingest[p: Params](base: Base, trace: Buf[1], families: Buf[1], acc: Buf[1], chals: Buf[16], num: Buf[16]):
    """One thread per row: sum over the descriptor's ingest entries of coef chal c_a(omega1^k1 x1, x2), the
    read cyclic inside the chain (fields of ir.mojo; kappa at [0, 16) is not read, it is folded later)."""
    comptime N = p.N()
    comptime h1 = p.h1()
    var row = global_idx.x
    if row >= N:
        return
    var x2 = row // h1
    var x1 = row % h1
    var first = u16(base, acc.at(2))
    var count = u16(base, acc.at(4))
    var s = SIMD[DType.float32, 16](0)          # a scalar times an E value is a lane product: coef c_a
    for i in range(first, first + count):       # below 2^14, times a canonical challenge below 2^21, reduced
        var o = i * ENTRY
        var col = u16(base, families.at(o + 16))
        var k1 = u16(base, families.at(o + 18)) // 2
        var t = Float32(Int(trace.load(base, col * N + x2 * h1 + (x1 + k1) % h1))) * Float32(Int(families.load(base, o + 29)))
        var chal = Int(families.load(base, o + 32))
        if chal != 0:
            s += fp_reduce(chals.load(base, chal - 1).cast[DType.float32]() * t)
        else:
            s[0] += fp_reduce(t)
    num.store(base, row, fp_canonical(s))


def k_horner_scan[p: Params](base: Base, acc: Buf[1], chals: Buf[16], num: Buf[16], zval: Buf[16]):
    """One thread per chain: R(1) = start, R(next) = scale R - ingest(row)."""
    comptime h1 = p.h1()
    var x2 = global_idx.x
    if x2 >= p.h2():
        return
    var r = SIMD[DType.float32, 16](0)
    r[0] = Float32(acc.load(base, 6))
    var scale = ext_one[4]()
    if acc.load(base, 7) != 0:
        scale = chals.load(base, Int(acc.load(base, 7)) - 1)
    var scale_f = scale.cast[DType.float32]()
    for x1 in range(h1):                            # |r| <= 190, scale canonical: the product is below 3.1 M
        var row = x2 * h1 + x1
        zval.store(base, row, fp_canonical(r))
        r = fp_reduce(fp_ext_mul[4](scale_f, r) - num.load(base, row).cast[DType.float32]())


def horner[p: Params](ctx: DeviceContext, arena: Arena, trace: Int, families: Int, acc: Int, chals: Int, A: AccLayout, k: Int) raises:
    """KIND_HORNER accumulator k: R into its Z block (row, e)."""
    comptime B = BACKEND.block
    ctx.enqueue_function[k_ingest[p]](arena.buf, Buf[1](trace), Buf[1](families), Buf[1](acc), Buf[16](chals), Buf[16](A.num),
                                      grid_dim=ceildiv(p.N(), B), block_dim=B)
    ctx.enqueue_function[k_horner_scan[p]](arena.buf, Buf[1](acc), Buf[16](chals), Buf[16](A.num), Buf[16](A.zval_at(k)),
                                           grid_dim=ceildiv(p.h2(), B), block_dim=B)


def wiring[p: Params](ctx: DeviceContext, arena: Arena, A: AccLayout, g: Int, pi: Int, wire: Int, sigma: Int, columns_w: Int, wchal: Int,
                      kappa_a: F2, kappa_b: F2, omega2: F2) raises:
    """Wiring product g (product index pi): its factor lines into A.wlines (4, h2, e) and its Z2 line (h2 + 1, e).
    Its slots are 2 g and 2 g + 1; kappa_a, kappa_b are kappa to those powers."""
    comptime B = BACKEND.block
    ctx.enqueue_function[k_wire_factors[p]](arena.buf, Buf[16](A.zval), Buf[1](wire), Buf[1](sigma), Int32(2 * g), Int32(columns_w), Buf[16](wchal),
                                            kappa_a[0], kappa_a[1], kappa_b[0], kappa_b[1], omega2[0], omega2[1], Buf[16](A.chain_prod), Buf[16](A.wline_at(g, 0)),
                                            grid_dim=ceildiv(p.h2(), B), block_dim=B)
    ctx.enqueue_function[k_z2[p]](arena.buf, Buf[16](A.chain_prod), Buf[16](A.z2_at(pi)), grid_dim=1, block_dim=1)


def derive_chals(ctx: DeviceContext, arena: Arena, chals: Int, table: Int, rows: Int) raises:
    """Complete the stage-1 elements at `chals` from the sampled ones by the `rows` table rows at `table`."""
    ctx.enqueue_function[k_derive_chals](arena.buf, Buf[16](chals), Buf[1](table), Int32(rows), grid_dim=1, block_dim=1)


def accumulate[p: Params](ctx: DeviceContext, arena: Arena, trace: Int, acc: Int, chals: Int, A: AccLayout, k: Int, pi: Int) raises:
    """Accumulator k (product index pi): its descriptor at `acc`, the stage-1 elements at `chals`, Z into its
    Z block (row, e), Z2 into its Z2 line (h2, e), the chain-end factors into its n_end, d_end lines (h2, e)."""
    comptime N = p.N()
    comptime B = BACKEND.block
    var num = A.num
    var den = A.den
    var scratch = A.scratch
    var zval = A.zval_at(k)
    var chain_prod = A.chain_prod
    var z2 = A.z2_at(pi)
    var n_end = A.n_end_at(pi)
    var d_end = A.d_end_at(pi)
    ctx.enqueue_function[k_factors[p]](arena.buf, Buf[1](trace), Buf[1](acc), Buf[16](chals), Buf[16](num), Buf[16](den),
                                       grid_dim=ceildiv(N, B), block_dim=B)
    comptime S = seg_len(p.h1())
    comptime assert p.h1() % S == 0, "scan segments must tile the chain"
    ctx.enqueue_function[k_seg_scan[p]](arena.buf, Buf[16](num), Buf[16](den), Buf[16](scratch), Buf[16](zval),
                                        grid_dim=ceildiv(N // S, B), block_dim=B)
    ctx.enqueue_function[k_chain_scan[p]](arena.buf, Buf[16](num), Buf[16](den), Buf[16](scratch), Buf[16](chain_prod),
                                          Buf[16](n_end), Buf[16](d_end), grid_dim=ceildiv(p.h2(), B), block_dim=B)
    ctx.enqueue_function[k_seg_fixup[p]](arena.buf, Buf[16](scratch), Buf[16](zval), grid_dim=ceildiv(N, B), block_dim=B)
    ctx.enqueue_function[k_z2[p]](arena.buf, Buf[16](chain_prod), Buf[16](z2), grid_dim=1, block_dim=1)
