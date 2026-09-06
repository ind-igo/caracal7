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
The stage-1 challenges arrive as the CHALS elements of ir.mojo: beta, delta, gamma, 1 + beta, (1 + beta) delta.

Buffers (bytes; slowest ... fastest), per accumulator:
    num, den     (row, e)     N and D per row, row = x2 h1 + x1
    zscratch     (row, e)     prefix products of D, then 1 / D
    zval         (row, e)     Z(x) with Z(1, x2) = 1 and Z(next) = Z N / D along the chain (7.2)
    chain_prod   (x2, e)      Z(e1, x2) N(e1, x2) / D(e1, x2), the whole chain's product
    n_end, d_end (x2, e)      N(e1, x2), D(e1, x2): the small grid's lines (smallgrid.mojo)
    z2           (x2 + 1, e)  Z2(1) = 1, Z2(omega2 x2) = Z2(x2) chain_prod(x2) (7.3, (W) for (P)); entry h2 is 1

ponytail: one thread per chain (h1 sequential E products, h2 threads) and one thread for Z2; the
two-level scan of 10.1 when a chain is long enough to matter. A zero D (probability ~ 1 / |E|)
gives 0 from ext_inv0 and a proof the verifier rejects; no abort path.
"""

from std.math import ceildiv
from std.gpu import global_idx
from max.gpu.host import DeviceContext

from caracal7.core.field import E, f_add, ext_mul, ext_inv0, ext_one
from caracal7.core.params import Params
from caracal7.core.backend import BACKEND
from caracal7.core.bytes import Base, Buf, u16
from caracal7.relations.ir import ACC, KIND_LOOKUP
from caracal7.core.arena import Arena



def k_derive_chals(base: Base, chals: Buf[16]):
    """One thread: the derived challenges 1 + beta and (1 + beta) delta after the three sampled ones."""
    if global_idx.x != 0:
        return
    var ob = f_add(chals.load(base, 0), ext_one[4]())
    chals.store(base, 3, ob)
    chals.store(base, 4, ext_mul[4](ob, chals.load(base, 1)))


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


def k_chain_scan[p: Params](base: Base, num: Buf[16], den: Buf[16], scratch: Buf[16], zval: Buf[16], chain_prod: Buf[16],
                            n_end: Buf[16], d_end: Buf[16]):
    """One thread per chain x2: batched inversion of D along the chain (prefix products, one inverse,
    backward pass), then Z(1, x2) = 1, Z(next) = Z N / D, and the chain's whole product."""
    comptime h1 = p.h1()
    var x2 = global_idx.x
    if x2 >= p.h2():
        return
    var row0 = x2 * h1
    var acc = ext_one[4]()
    for i in range(h1):
        acc = ext_mul[4](acc, den.load(base, row0 + i))
        scratch.store(base, row0 + i, acc)
    var inv = ext_inv0[4](acc)
    for i in range(h1 - 1, -1, -1):
        var pref = scratch.load(base, row0 + i - 1) if i > 0 else ext_one[4]()
        var inv_d = ext_mul[4](inv, pref)
        inv = ext_mul[4](inv, den.load(base, row0 + i))
        scratch.store(base, row0 + i, inv_d)
    var z = ext_one[4]()
    for i in range(h1):
        zval.store(base, row0 + i, z)
        z = ext_mul[4](z, ext_mul[4](num.load(base, row0 + i), scratch.load(base, row0 + i)))
    chain_prod.store(base, x2, z)
    n_end.store(base, x2, num.load(base, row0 + h1 - 1))
    d_end.store(base, x2, den.load(base, row0 + h1 - 1))


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


def derive_chals(ctx: DeviceContext, arena: Arena, chals: Int) raises:
    """Complete the CHALS elements at `chals` from the three sampled ones."""
    ctx.enqueue_function[k_derive_chals](arena.buf, Buf[16](chals), grid_dim=1, block_dim=1)


def accumulate[p: Params](ctx: DeviceContext, arena: Arena, trace: Int, acc: Int, chals: Int,
                          num: Int, den: Int, scratch: Int, zval: Int, chain_prod: Int, z2: Int, n_end: Int, d_end: Int) raises:
    """One accumulator: its descriptor at `acc`, the CHALS elements at `chals`, Z into zval (row, e), Z2 into z2 (h2, e),
    the chain-end factors into n_end, d_end (h2, e)."""
    comptime N = p.N()
    comptime B = BACKEND.block
    ctx.enqueue_function[k_factors[p]](arena.buf, Buf[1](trace), Buf[1](acc), Buf[16](chals), Buf[16](num), Buf[16](den),
                                       grid_dim=ceildiv(N, B), block_dim=B)
    ctx.enqueue_function[k_chain_scan[p]](arena.buf, Buf[16](num), Buf[16](den), Buf[16](scratch), Buf[16](zval), Buf[16](chain_prod),
                                          Buf[16](n_end), Buf[16](d_end), grid_dim=ceildiv(p.h2(), B), block_dim=B)
    ctx.enqueue_function[k_z2[p]](arena.buf, Buf[16](chain_prod), Buf[16](z2), grid_dim=1, block_dim=1)
