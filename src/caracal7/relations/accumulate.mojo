"""The Z stage (spec 10 steps 4-6, 6.2, 7.1-7.3): factors, batched inversion, the running product
per chain, and Z2 across chains, for permutation accumulators on witness columns.

Accumulator descriptor (ACC bytes, u16 little-endian): z_col [0, 2) the global index of coordinate
column 0 of Z; w_num [2, 4), w_den [4, 6); num columns [6, 22); den columns [22, 38). The factors are
N = gamma + fp(num), D = gamma + fp(den), fp(c) = sum_j c_j b_j the fingerprint of 6.1 (b_j the unit
vector j of E: no multiplication, the column bytes are the coordinates).

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
from caracal7.core.arena import Arena

comptime ACC = 40
comptime ACC_W_MAX = 8


def k_factors[p: Params](base: Base, trace: Buf[1], acc: Buf[1], gamma: Buf[16], num: Buf[16], den: Buf[16]):
    """One thread per row: num[row] = gamma + fp(num columns at row), den likewise."""
    comptime N = p.N()
    var row = global_idx.x
    if row >= N:
        return
    var g = gamma.load(base, 0)
    var n = g
    var d = g
    var w_num = u16(base, acc.at(2))
    var w_den = u16(base, acc.at(4))
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


def accumulate[p: Params](ctx: DeviceContext, arena: Arena, trace: Int, acc: Int, gamma: Int,
                          num: Int, den: Int, scratch: Int, zval: Int, chain_prod: Int, z2: Int, n_end: Int, d_end: Int) raises:
    """One accumulator: its descriptor at `acc`, Z into zval (row, e), Z2 into z2 (h2, e), the chain-end factors into n_end, d_end (h2, e)."""
    comptime N = p.N()
    comptime B = BACKEND.block
    ctx.enqueue_function[k_factors[p]](arena.buf, Buf[1](trace), Buf[1](acc), Buf[16](gamma), Buf[16](num), Buf[16](den),
                                       grid_dim=ceildiv(N, B), block_dim=B)
    ctx.enqueue_function[k_chain_scan[p]](arena.buf, Buf[16](num), Buf[16](den), Buf[16](scratch), Buf[16](zval), Buf[16](chain_prod),
                                          Buf[16](n_end), Buf[16](d_end), grid_dim=ceildiv(p.h2(), B), block_dim=B)
    ctx.enqueue_function[k_z2[p]](arena.buf, Buf[16](chain_prod), Buf[16](z2), grid_dim=1, block_dim=1)
