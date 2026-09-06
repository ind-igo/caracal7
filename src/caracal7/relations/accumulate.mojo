"""The Z stage (spec 10 steps 4-6, 6.2, 7.1-7.3): factors, batched inversion, the running product
per chain, and Z2 across chains, for permutation accumulators on witness columns.

Accumulator descriptor (ACC bytes, u16 little-endian): z_col [0, 2) the global index of coordinate
column 0 of Z; w_num [2, 4), w_den [4, 6); num columns [6, 22); den columns [22, 38); kind u8 [38]
(KIND_PERM, KIND_LOOKUP); table id u8 [39] (lookup: the index into Shape.tables).
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

The scan is the two-level one of 10.1: segments of S = seg_len(h1) rows in parallel (N / S threads),
one thread per chain over the segment totals, a fix-up per row. ponytail: one thread for Z2 (h2 steps). A zero D (probability ~ 1 / |E|)
gives 0 from ext_inv0 for its segment, a zero chain product, and a proof the verifier rejects; no abort path.
"""

from std.math import ceildiv
from std.gpu import global_idx
from max.gpu.host import DeviceContext

from caracal7.core.field import E, f_add, ext_mul, ext_inv0, ext_one
from caracal7.core.params import Params
from caracal7.core.backend import BACKEND
from caracal7.core.bytes import Base, Buf, u16
from caracal7.relations.ir import ACC, KIND_LOOKUP, CHAL, CHAL_ADD, CHAL_ONE, SAMPLED
from caracal7.core.arena import Arena



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
    var z = ext_one[4]()
    for x2 in range(p.h2()):
        z2.store(base, x2, z)
        z = ext_mul[4](z, chain_prod.load(base, x2))
    z2.store(base, p.h2(), ext_one[4]())


def derive_chals(ctx: DeviceContext, arena: Arena, chals: Int, table: Int, rows: Int) raises:
    """Complete the stage-1 elements at `chals` from the sampled ones by the `rows` table rows at `table`."""
    ctx.enqueue_function[k_derive_chals](arena.buf, Buf[16](chals), Buf[1](table), Int32(rows), grid_dim=1, block_dim=1)


def accumulate[p: Params](ctx: DeviceContext, arena: Arena, trace: Int, acc: Int, chals: Int,
                          num: Int, den: Int, scratch: Int, zval: Int, chain_prod: Int, z2: Int, n_end: Int, d_end: Int) raises:
    """One accumulator: its descriptor at `acc`, the stage-1 elements at `chals`, Z into zval (row, e), Z2 into z2 (h2, e),
    the chain-end factors into n_end, d_end (h2, e)."""
    comptime N = p.N()
    comptime B = BACKEND.block
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
