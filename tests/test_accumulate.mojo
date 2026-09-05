"""The Z stage on the synthetic permutation accumulator, reference profile: the device Z and Z2
against the host definitions, the chain relation Z(next) D = Z N on every row but the chain end,
Z(1, x2) = 1, and the grand product Z2(e2) chain_prod(e2) = 1 (c8 is a permutation of c0)."""

from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceContext, HostBuffer

from caracal7.core.field import E, f_add, ext_mul, ext_inv, ext_one
from caracal7.core.params import Params
from caracal7.core.params import REFERENCE
from caracal7.core.arena import Arena, Bump
from caracal7.relations.accumulate import ACC, accumulate
from caracal7.core.bytes import get_u16, list_e
from caracal7.relations.synthetic import SYNTHETIC_COLUMNS, synthetic_families, synthetic_trace

comptime p = REFERENCE
comptime N = p.N()
comptime h1 = p.h1()
comptime h2 = p.h2()


def _host(ctx: DeviceContext, l: List[UInt8]) raises -> HostBuffer[DType.uint8]:
    var h = ctx.enqueue_create_host_buffer[DType.uint8](len(l))
    ctx.synchronize()
    for i in range(len(l)):
        h[i] = l[i]
    return h^


def _down(ctx: DeviceContext, arena: Arena, off: Int, n: Int) raises -> List[UInt8]:
    var h = ctx.enqueue_create_host_buffer[DType.uint8](n)
    arena.download(ctx, off, h)
    ctx.synchronize()
    var l = List[UInt8](capacity=n)
    for i in range(n):
        l.append(h[i])
    return l^



# ---- host reference (design rule 5) ----

def host_factor(accs: List[UInt8], k: Int, trace: List[UInt8], N: Int, row: Int, gamma: E, den: Bool) -> E:
    """N(row) or D(row) of accumulator k from a host trace (column, row)."""
    var v = gamma
    var w = get_u16(accs, k * ACC + (4 if den else 2))
    for j in range(w):
        v[j] = f_add(v[j], trace[get_u16(accs, k * ACC + (22 if den else 6) + 2 * j) * N + row])
    return v


def host_accumulate[p: Params](accs: List[UInt8], k: Int, trace: List[UInt8], gamma: E) raises -> Tuple[List[UInt8], List[UInt8]]:
    """(Z as (row, e) bytes, Z2 as (x2, e) bytes) by the definitions, for the tests."""
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    comptime N = p.N()
    var z = List[UInt8](length=N * 16, fill=0)
    var z2 = List[UInt8](length=h2 * 16, fill=0)
    var acc2 = ext_one[4]()
    for x2 in range(h2):
        for t in range(16):
            z2[x2 * 16 + t] = acc2[t]
        var acc = ext_one[4]()
        for x1 in range(h1):
            var row = x2 * h1 + x1
            for t in range(16):
                z[row * 16 + t] = acc[t]
            acc = ext_mul[4](acc, ext_mul[4](host_factor(accs, k, trace, N, row, gamma, False),
                                             ext_inv[4](host_factor(accs, k, trace, N, row, gamma, True))))
        acc2 = ext_mul[4](acc2, acc)
    return (z^, z2^)



def test_accumulator_matches_host_and_satisfies_the_relations() raises:
    var f = synthetic_families()
    assert_equal(len(f.accs), 2 * ACC)
    for k in range(2):
        _check(f.accs, k)


def _check(accs: List[UInt8], k: Int) raises:
    var ctx = DeviceContext()
    var trace = synthetic_trace[p](1)
    var gamma = E(0)
    for t in range(16):
        gamma[t] = UInt8((t * 37 + 5) % 127)
    var bump = Bump()
    var o_trace = bump.alloc(SYNTHETIC_COLUMNS * N)
    var o_acc = bump.alloc(ACC)
    var o_gamma = bump.alloc(16)
    var o_num = bump.alloc(N * 16)
    var o_den = bump.alloc(N * 16)
    var o_scratch = bump.alloc(N * 16)
    var o_z = bump.alloc(N * 16)
    var o_prod = bump.alloc(h2 * 16)
    var o_z2 = bump.alloc((h2 + 1) * 16)   # k_z2 stores a trailing 1
    var o_nend = bump.alloc(h2 * 16)
    var o_dend = bump.alloc(h2 * 16)
    var arena = Arena(ctx, bump.used)
    arena.upload(ctx, o_trace, _host(ctx, trace))
    var one_acc = List[UInt8](capacity=ACC)
    for i in range(ACC):
        one_acc.append(accs[k * ACC + i])
    arena.upload(ctx, o_acc, _host(ctx, one_acc))
    var gl = List[UInt8](capacity=16)
    for t in range(16):
        gl.append(gamma[t])
    arena.upload(ctx, o_gamma, _host(ctx, gl))
    accumulate[p](ctx, arena, o_trace, o_acc, o_gamma, o_num, o_den, o_scratch, o_z, o_prod, o_z2, o_nend, o_dend)
    var z = _down(ctx, arena, o_z, N * 16)
    var z2 = _down(ctx, arena, o_z2, (h2 + 1) * 16)
    var prod = _down(ctx, arena, o_prod, h2 * 16)

    var want_z: List[UInt8]
    var want_z2: List[UInt8]
    var got = host_accumulate[p](accs, k, trace, gamma)
    want_z = got[0].copy()
    want_z2 = got[1].copy()
    assert_true(z == want_z, "Z differs from the host definition")
    assert_true(z2[: h2 * 16] == want_z2, "Z2 differs from the host definition")
    assert_true(list_e(z2, h2) == ext_one[4](), "trailing Z2 element is not 1")
    var one = E(0)
    one[0] = 1
    for x2 in range(h2):
        assert_true(list_e(z, x2 * h1) == one, "chain start is not 1")
        for x1 in range(h1 - 1):
            var row = x2 * h1 + x1
            var lhs = ext_mul[4](list_e(z, row + 1), host_factor(accs, k, trace, N, row, gamma, True))
            var rhs = ext_mul[4](list_e(z, row), host_factor(accs, k, trace, N, row, gamma, False))
            assert_true(lhs == rhs, "chain relation fails")
    assert_true(list_e(z2, 0) == one, "Z2(1) is not 1")
    for x2 in range(h2 - 1):
        assert_true(list_e(z2, x2 + 1) == ext_mul[4](list_e(z2, x2), list_e(prod, x2)), "Z2 recurrence fails")
    assert_true(ext_mul[4](list_e(z2, h2 - 1), list_e(prod, h2 - 1)) == one, "grand product is not 1: c8 is not a permutation of c0")
    assert_true(list_e(z, N - 1) != one, "vacuous")
    var nend = _down(ctx, arena, o_nend, h2 * 16)
    var dend = _down(ctx, arena, o_dend, h2 * 16)
    for x2 in range(h2):
        assert_true(list_e(nend, x2) == host_factor(accs, k, trace, N, x2 * h1 + h1 - 1, gamma, False), "chain-end N")
        assert_true(list_e(dend, x2) == host_factor(accs, k, trace, N, x2 * h1 + h1 - 1, gamma, True), "chain-end D")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
