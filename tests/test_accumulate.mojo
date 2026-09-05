"""The Z stage on the synthetic permutation accumulator, reference profile: the device Z and Z2
against the host definitions, the chain relation Z(next) D = Z N on every row but the chain end,
Z(1, x2) = 1, and the grand product Z2(e2) chain_prod(e2) = 1 (c8 is a permutation of c0)."""

from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceContext, HostBuffer

from caracal7.field import E, ext_mul
from caracal7.params import REFERENCE
from caracal7.arena import Arena, Bump
from caracal7.accumulate import ACC, accumulate, host_accumulate, host_factor
from caracal7.residual import SYNTHETIC_COLUMNS, synthetic_families, synthetic_trace

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


def _e(l: List[UInt8], i: Int) -> E:
    var v = E(0)
    for t in range(16):
        v[t] = l[i * 16 + t]
    return v


def test_accumulator_matches_host_and_satisfies_the_relations() raises:
    var ctx = DeviceContext()
    var f = synthetic_families()
    assert_equal(len(f.accs), ACC)
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
    var o_z2 = bump.alloc(h2 * 16)
    var arena = Arena(ctx, bump.used)
    arena.upload(ctx, o_trace, _host(ctx, trace))
    arena.upload(ctx, o_acc, _host(ctx, f.accs))
    var gl = List[UInt8](capacity=16)
    for t in range(16):
        gl.append(gamma[t])
    arena.upload(ctx, o_gamma, _host(ctx, gl))
    accumulate[p](ctx, arena.base(), o_trace, o_acc, o_gamma, o_num, o_den, o_scratch, o_z, o_prod, o_z2)
    var z = _down(ctx, arena, o_z, N * 16)
    var z2 = _down(ctx, arena, o_z2, h2 * 16)
    var prod = _down(ctx, arena, o_prod, h2 * 16)

    var want_z: List[UInt8]
    var want_z2: List[UInt8]
    var got = host_accumulate[p](f.accs, 0, trace, gamma)
    want_z = got[0].copy()
    want_z2 = got[1].copy()
    assert_true(z == want_z, "Z differs from the host definition")
    assert_true(z2 == want_z2, "Z2 differs from the host definition")
    var one = E(0)
    one[0] = 1
    for x2 in range(h2):
        assert_true(_e(z, x2 * h1) == one, "chain start is not 1")
        for x1 in range(h1 - 1):
            var row = x2 * h1 + x1
            var lhs = ext_mul[4](_e(z, row + 1), host_factor(f.accs, 0, trace, N, row, gamma, True))
            var rhs = ext_mul[4](_e(z, row), host_factor(f.accs, 0, trace, N, row, gamma, False))
            assert_true(lhs == rhs, "chain relation fails")
    assert_true(_e(z2, 0) == one, "Z2(1) is not 1")
    for x2 in range(h2 - 1):
        assert_true(_e(z2, x2 + 1) == ext_mul[4](_e(z2, x2), _e(prod, x2)), "Z2 recurrence fails")
    assert_true(ext_mul[4](_e(z2, h2 - 1), _e(prod, h2 - 1)) == one, "grand product is not 1: c8 is not a permutation of c0")
    assert_true(_e(z, N - 1) != one, "vacuous")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
