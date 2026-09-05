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
from max.gpu.host import DeviceContext

from caracal7.field import E, f_add, ext_mul, ext_inv0, ext_one
from caracal7.params import Params
from caracal7.backend import BACKEND
from caracal7.device import gid, load_e, load_u16

comptime ACC = 40
comptime ACC_W_MAX = 8


def acc_get16(accs: List[UInt8], k: Int, at: Int) -> Int:
    return Int(accs[k * ACC + at]) | Int(accs[k * ACC + at + 1]) << 8


def k_factors[p: Params](base: Pointer[UInt8, MutAnyOrigin], trace: Int64, acc: Int64, gamma: Int64, num: Int64, den: Int64):
    """One thread per row: num[row] = gamma + fp(num columns at row), den likewise."""
    comptime N = p.N()
    var row = gid()
    if row >= N:
        return
    var g = load_e(base, Int(gamma))
    var n = g
    var d = g
    var w_num = load_u16(base, Int(acc) + 2)
    var w_den = load_u16(base, Int(acc) + 4)
    for j in range(w_num):
        n[j] = f_add(n[j], base[unsafe_offset=Int(trace) + load_u16(base, Int(acc) + 6 + 2 * j) * N + row])
    for j in range(w_den):
        d[j] = f_add(d[j], base[unsafe_offset=Int(trace) + load_u16(base, Int(acc) + 22 + 2 * j) * N + row])
    base.unsafe_store[width=16](Int(num) + row * 16, n)
    base.unsafe_store[width=16](Int(den) + row * 16, d)


def k_chain_scan[p: Params](base: Pointer[UInt8, MutAnyOrigin], num: Int64, den: Int64, scratch: Int64, zval: Int64, chain_prod: Int64,
                            n_end: Int64, d_end: Int64):
    """One thread per chain x2: batched inversion of D along the chain (prefix products, one inverse,
    backward pass), then Z(1, x2) = 1, Z(next) = Z N / D, and the chain's whole product."""
    comptime h1 = p.h1()
    var x2 = gid()
    if x2 >= p.h2():
        return
    var row0 = x2 * h1
    var acc = ext_one[4]()
    for i in range(h1):
        acc = ext_mul[4](acc, load_e(base, Int(den) + (row0 + i) * 16))
        base.unsafe_store[width=16](Int(scratch) + (row0 + i) * 16, acc)
    var inv = ext_inv0[4](acc)
    for i in range(h1 - 1, -1, -1):
        var pref = load_e(base, Int(scratch) + (row0 + i - 1) * 16) if i > 0 else ext_one[4]()
        var inv_d = ext_mul[4](inv, pref)
        inv = ext_mul[4](inv, load_e(base, Int(den) + (row0 + i) * 16))
        base.unsafe_store[width=16](Int(scratch) + (row0 + i) * 16, inv_d)
    var z = ext_one[4]()
    for i in range(h1):
        base.unsafe_store[width=16](Int(zval) + (row0 + i) * 16, z)
        z = ext_mul[4](z, ext_mul[4](load_e(base, Int(num) + (row0 + i) * 16), load_e(base, Int(scratch) + (row0 + i) * 16)))
    base.unsafe_store[width=16](Int(chain_prod) + x2 * 16, z)
    base.unsafe_store[width=16](Int(n_end) + x2 * 16, load_e(base, Int(num) + (row0 + h1 - 1) * 16))
    base.unsafe_store[width=16](Int(d_end) + x2 * 16, load_e(base, Int(den) + (row0 + h1 - 1) * 16))


def k_z2[p: Params](base: Pointer[UInt8, MutAnyOrigin], chain_prod: Int64, z2: Int64):
    """One thread: Z2 across the chains, then Z2(omega2^h2) = Z2(1) = 1 at index h2 so the shifted
    line Z2(omega2 X2) of the small grid is the same buffer one element on."""
    if gid() != 0:
        return
    var z = ext_one[4]()
    for x2 in range(p.h2()):
        base.unsafe_store[width=16](Int(z2) + x2 * 16, z)
        z = ext_mul[4](z, load_e(base, Int(chain_prod) + x2 * 16))
    base.unsafe_store[width=16](Int(z2) + p.h2() * 16, ext_one[4]())


def accumulate[p: Params](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin], trace: Int, acc: Int, gamma: Int,
                          num: Int, den: Int, scratch: Int, zval: Int, chain_prod: Int, z2: Int, n_end: Int, d_end: Int) raises:
    """One accumulator: its descriptor at `acc`, Z into zval (row, e), Z2 into z2 (h2, e), the chain-end factors into n_end, d_end (h2, e)."""
    comptime N = p.N()
    ctx.enqueue_function[k_factors[p]](base, Int64(trace), Int64(acc), Int64(gamma), Int64(num), Int64(den),
                                       grid_dim=ceildiv(N, BACKEND.block), block_dim=BACKEND.block)
    ctx.enqueue_function[k_chain_scan[p]](base, Int64(num), Int64(den), Int64(scratch), Int64(zval), Int64(chain_prod),
                                          Int64(n_end), Int64(d_end), grid_dim=ceildiv(p.h2(), BACKEND.block), block_dim=BACKEND.block)
    ctx.enqueue_function[k_z2[p]](base, Int64(chain_prod), Int64(z2), grid_dim=1, block_dim=1)

