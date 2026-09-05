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
    z2           (x2, e)      Z2(1) = 1, Z2(omega2 x2) = Z2(x2) chain_prod(x2) (7.3, (W) for (P))

ponytail: one thread per chain (h1 sequential E products, h2 threads) and one thread for Z2; the
two-level scan of 10.1 when a chain is long enough to matter. A zero D (probability ~ 1 / |E|)
gives 0 from ext_inv0 and a proof the verifier rejects; no abort path.
"""

from std.math import ceildiv
from std.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext

from caracal7.field import E, f_add, ext_mul, ext_inv0, ext_inv
from caracal7.params import Params

comptime ACC = 40
comptime ACC_W_MAX = 8
comptime BLOCK = 256


def acc_get16(accs: List[UInt8], k: Int, at: Int) -> Int:
    return Int(accs[k * ACC + at]) | Int(accs[k * ACC + at + 1]) << 8


@always_inline
def _gid() -> Int:
    return Int(block_idx.x * block_dim.x + thread_idx.x)


@always_inline
def _e(base: Pointer[UInt8, MutAnyOrigin], off: Int) -> E:
    return base.unsafe_load[width=16](off)


@always_inline
def _d16(base: Pointer[UInt8, MutAnyOrigin], at: Int) -> Int:
    return Int(base[unsafe_offset=at]) | Int(base[unsafe_offset=at + 1]) << 8


@always_inline
def _one() -> E:
    var v = E(0)
    v[0] = 1
    return v


def k_factors[p: Params](base: Pointer[UInt8, MutAnyOrigin], trace: Int64, acc: Int64, gamma: Int64, num: Int64, den: Int64):
    """One thread per row: num[row] = gamma + fp(num columns at row), den likewise."""
    comptime N = p.N()
    var row = _gid()
    if row >= N:
        return
    var g = _e(base, Int(gamma))
    var n = g
    var d = g
    var w_num = _d16(base, Int(acc) + 2)
    var w_den = _d16(base, Int(acc) + 4)
    for j in range(w_num):
        n[j] = f_add(n[j], base[unsafe_offset=Int(trace) + _d16(base, Int(acc) + 6 + 2 * j) * N + row])
    for j in range(w_den):
        d[j] = f_add(d[j], base[unsafe_offset=Int(trace) + _d16(base, Int(acc) + 22 + 2 * j) * N + row])
    base.unsafe_store[width=16](Int(num) + row * 16, n)
    base.unsafe_store[width=16](Int(den) + row * 16, d)


def k_chain_scan[p: Params](base: Pointer[UInt8, MutAnyOrigin], num: Int64, den: Int64, scratch: Int64, zval: Int64, chain_prod: Int64):
    """One thread per chain x2: batched inversion of D along the chain (prefix products, one inverse,
    backward pass), then Z(1, x2) = 1, Z(next) = Z N / D, and the chain's whole product."""
    comptime h1 = p.h1()
    var x2 = _gid()
    if x2 >= p.h2():
        return
    var row0 = x2 * h1
    var acc = _one()
    for i in range(h1):
        acc = ext_mul[4](acc, _e(base, Int(den) + (row0 + i) * 16))
        base.unsafe_store[width=16](Int(scratch) + (row0 + i) * 16, acc)
    var inv = ext_inv0[4](acc)
    for i in range(h1 - 1, -1, -1):
        var pref = _e(base, Int(scratch) + (row0 + i - 1) * 16) if i > 0 else _one()
        var inv_d = ext_mul[4](inv, pref)
        inv = ext_mul[4](inv, _e(base, Int(den) + (row0 + i) * 16))
        base.unsafe_store[width=16](Int(scratch) + (row0 + i) * 16, inv_d)
    var z = _one()
    for i in range(h1):
        base.unsafe_store[width=16](Int(zval) + (row0 + i) * 16, z)
        z = ext_mul[4](z, ext_mul[4](_e(base, Int(num) + (row0 + i) * 16), _e(base, Int(scratch) + (row0 + i) * 16)))
    base.unsafe_store[width=16](Int(chain_prod) + x2 * 16, z)


def k_z2[p: Params](base: Pointer[UInt8, MutAnyOrigin], chain_prod: Int64, z2: Int64):
    """One thread: Z2 across the chains."""
    if _gid() != 0:
        return
    var z = _one()
    for x2 in range(p.h2()):
        base.unsafe_store[width=16](Int(z2) + x2 * 16, z)
        z = ext_mul[4](z, _e(base, Int(chain_prod) + x2 * 16))


def accumulate[p: Params](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin], trace: Int, acc: Int, gamma: Int,
                          num: Int, den: Int, scratch: Int, zval: Int, chain_prod: Int, z2: Int) raises:
    """One accumulator: its descriptor at `acc`, Z into zval (row, e), Z2 into z2 (h2, e)."""
    comptime N = p.N()
    ctx.enqueue_function[k_factors[p]](base, Int64(trace), Int64(acc), Int64(gamma), Int64(num), Int64(den),
                                       grid_dim=ceildiv(N, BLOCK), block_dim=BLOCK)
    ctx.enqueue_function[k_chain_scan[p]](base, Int64(num), Int64(den), Int64(scratch), Int64(zval), Int64(chain_prod),
                                          grid_dim=ceildiv(p.h2(), 32), block_dim=32)
    ctx.enqueue_function[k_z2[p]](base, Int64(chain_prod), Int64(z2), grid_dim=1, block_dim=1)


# ---- host side ----

def host_factor(accs: List[UInt8], k: Int, trace: List[UInt8], N: Int, row: Int, gamma: E, den: Bool) -> E:
    """N(row) or D(row) of accumulator k from a host trace (column, row)."""
    var v = gamma
    var w = acc_get16(accs, k, 4 if den else 2)
    for j in range(w):
        v[j] = f_add(v[j], trace[acc_get16(accs, k, (22 if den else 6) + 2 * j) * N + row])
    return v


def host_accumulate[p: Params](accs: List[UInt8], k: Int, trace: List[UInt8], gamma: E) raises -> Tuple[List[UInt8], List[UInt8]]:
    """(Z as (row, e) bytes, Z2 as (x2, e) bytes) by the definitions, for the tests."""
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    comptime N = p.N()
    var z = List[UInt8](length=N * 16, fill=0)
    var z2 = List[UInt8](length=h2 * 16, fill=0)
    var acc2 = _one()
    for x2 in range(h2):
        for t in range(16):
            z2[x2 * 16 + t] = acc2[t]
        var acc = _one()
        for x1 in range(h1):
            var row = x2 * h1 + x1
            for t in range(16):
                z[row * 16 + t] = acc[t]
            acc = ext_mul[4](acc, ext_mul[4](host_factor(accs, k, trace, N, row, gamma, False),
                                             ext_inv[4](host_factor(accs, k, trace, N, row, gamma, True))))
        acc2 = ext_mul[4](acc2, acc)
    return (z^, z2^)
