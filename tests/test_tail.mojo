"""Tail kernels against the host formulas, reference profile: leaf points, expected symbols against
the host encoders, the materialized query against the batched claim (the identity the sumcheck
proves), the three rounds against the verifier's checks, and the fold."""

from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceContext, HostBuffer

from caracal7.field import F4, E, f_add, ext_mul
from caracal7.params import REFERENCE
from caracal7.tables import RsDomain
from caracal7.arena import Arena, Bump
from caracal7.tail import DOM_BYTES, ROUND_THREADS, domain_bytes, points, expected_symbols, tail_materialize, tail_round, tail_fold
from caracal7.tail import host_e, tail_encode_at, fold8_host, quadratic_at
from caracal7.verifier import encode_at

comptime p = REFERENCE
comptime N = p.N()
comptime ROWS = N // 8
comptime Q = 16
comptime L0_TAIL = 4608
comptime M_TAIL = 4


def _rand(n: Int, seed: Int, below: Int) -> List[UInt8]:
    var l = List[UInt8](capacity=n)
    var s = seed
    for _ in range(n):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        l.append(UInt8((s >> 8) % below))
    return l^


def _u32s(vals: List[Int]) -> List[UInt8]:
    var l = List[UInt8]()
    for v in vals:
        for b in range(4):
            l.append(UInt8((v >> (8 * b)) & 255))
    return l^


def _inner(a: List[UInt8], b: List[UInt8], n: Int) -> E:
    var acc = E(0)
    for i in range(n):
        acc = f_add(acc, ext_mul[4](host_e(a, i), host_e(b, i)))
    return acc


def _up(ctx: DeviceContext, arena: Arena, off: Int, l: List[UInt8]) raises:
    var h = ctx.enqueue_create_host_buffer[DType.uint8](len(l))
    ctx.synchronize()
    for i in range(len(l)):
        h[i] = l[i]
    arena.upload(ctx, off, h)


def _down(ctx: DeviceContext, arena: Arena, off: Int, n: Int) raises -> List[UInt8]:
    var h = ctx.enqueue_create_host_buffer[DType.uint8](n)
    arena.download(ctx, off, h)
    ctx.synchronize()
    var l = List[UInt8](capacity=n)
    for i in range(n):
        l.append(h[i])
    return l^


def test_tail_kernels() raises:
    var ctx = DeviceContext()
    var dom1 = RsDomain(p.L0, p.m_cosets)
    var dom2 = RsDomain(L0_TAIL, M_TAIL)
    var y = _rand(N * 16, 1, 127)
    var running = _rand(N * 16, 2, 127)
    var batch = _rand((1 + 4 * Q) * 16, 3, 127)
    var r = _rand(3 * 16, 4, 127)
    var pos_list = List[Int]()
    var pos2_list = List[Int]()
    var s = 99
    for _ in range(Q):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        pos_list.append(s % p.L())
        pos2_list.append((s >> 3) % (L0_TAIL * M_TAIL))
    var batch2 = _rand((1 + Q) * 16, 5, 127)

    var bump = Bump()
    var o_y = bump.alloc(N * 16)
    var o_run = bump.alloc(N * 16)
    var o_batch = bump.alloc((1 + 4 * Q) * 16)
    var o_r = bump.alloc(3 * 16)
    var o_pos = bump.alloc(Q * 4)
    var o_pos2 = bump.alloc(Q * 4)
    var o_dom1 = bump.alloc(DOM_BYTES)
    var o_dom2 = bump.alloc(DOM_BYTES)
    var o_pts = bump.alloc(Q * 4)
    var o_pts2 = bump.alloc(Q * 4)
    var o_v = bump.alloc(4 * Q * 16)
    var o_v2 = bump.alloc(Q * 16)
    var o_w = bump.alloc(N * 16)
    var o_partial = bump.alloc(3 * ROUND_THREADS * 16)
    var o_rounds = bump.alloc(9 * 16)
    var o_ynext = bump.alloc(ROWS * 16)
    var o_wnext = bump.alloc(ROWS * 16)
    var o_batch2 = bump.alloc((1 + Q) * 16)
    var o_w2 = bump.alloc(ROWS * 16)
    var arena = Arena(ctx, bump.used)

    _up(ctx, arena, o_y, y)
    _up(ctx, arena, o_run, running)
    _up(ctx, arena, o_batch, batch)
    _up(ctx, arena, o_r, r)
    _up(ctx, arena, o_pos, _u32s(pos_list))
    _up(ctx, arena, o_pos2, _u32s(pos2_list))
    _up(ctx, arena, o_dom1, domain_bytes(dom1))
    _up(ctx, arena, o_dom2, domain_bytes(dom2))
    _up(ctx, arena, o_batch2, batch2)
    var base = arena.base()

    # level-1 functionals on y
    points(ctx, base, o_pos, Q, o_dom1, p.L0, o_pts)
    expected_symbols[p](ctx, base, True, o_y, 0, o_pts, Q, o_v)
    tail_materialize[p](ctx, base, True, o_run, o_batch, o_pts, Q, N, o_w)
    for d in range(3):
        tail_round(ctx, base, o_w, o_y, N, d, o_r, o_partial, o_rounds + d * 3 * 16)
    tail_fold(ctx, base, o_y, ROWS, o_r, o_ynext)
    tail_fold(ctx, base, o_w, ROWS, o_r, o_wnext)
    # tail-level functionals on y_next
    points(ctx, base, o_pos2, Q, o_dom2, L0_TAIL, o_pts2)
    expected_symbols[p](ctx, base, False, o_ynext, ROWS, o_pts2, Q, o_v2)
    tail_materialize[p](ctx, base, False, o_wnext, o_batch2, o_pts2, Q, ROWS, o_w2)

    var pts = _down(ctx, arena, o_pts, Q * 4)
    var v = _down(ctx, arena, o_v, 4 * Q * 16)
    var w = _down(ctx, arena, o_w, N * 16)
    var rounds = _down(ctx, arena, o_rounds, 9 * 16)
    var ynext = _down(ctx, arena, o_ynext, ROWS * 16)
    var wnext = _down(ctx, arena, o_wnext, ROWS * 16)
    var pts2 = _down(ctx, arena, o_pts2, Q * 4)
    var v2 = _down(ctx, arena, o_v2, Q * 16)
    var w2 = _down(ctx, arena, o_w2, ROWS * 16)

    for q in range(Q):
        var pt = F4(pts[4 * q], pts[4 * q + 1], pts[4 * q + 2], pts[4 * q + 3])
        assert_true(pt == dom1.point(pos_list[q]), "point mismatch")
        var enc = encode_at[p](y, pt)
        for tau in range(4):
            assert_true(host_e(v, 4 * q + tau) == enc[tau], "expected symbol mismatch (level 1)")
    # the batched claim: <y, w~> = batch_0 <y, running> + sum_q batch_q v_q
    var claim = ext_mul[4](host_e(batch, 0), _inner(y, running, N))
    for k in range(4 * Q):
        claim = f_add(claim, ext_mul[4](host_e(batch, 1 + k), host_e(v, k)))
    assert_true(_inner(y, w, N) == claim, "materialized query does not carry the batched claim")
    # sumcheck: s_d(0) + s_d(1) = previous, ending at <fold(y), fold(w~)>
    var prev = claim
    for d in range(3):
        assert_true(f_add(host_e(rounds, 3 * d), host_e(rounds, 3 * d + 1)) == prev, "round " + String(d) + " sum mismatch")
        prev = quadratic_at(rounds, 3 * d, host_e(r, d))
    assert_true(ynext == fold8_host(y, ROWS, r), "fold mismatch")
    assert_true(wnext == fold8_host(w, ROWS, r), "query fold mismatch")
    assert_true(_inner(ynext, wnext, ROWS) == prev, "folded claim mismatch")
    assert_true(prev != E(0), "vacuous")
    # tail level
    for q in range(Q):
        var pt = F4(pts2[4 * q], pts2[4 * q + 1], pts2[4 * q + 2], pts2[4 * q + 3])
        assert_true(pt == dom2.point(pos2_list[q]), "coset point mismatch")
        assert_true(host_e(v2, q) == tail_encode_at(ynext, ROWS, pt), "expected symbol mismatch (tail)")
    var claim2 = ext_mul[4](host_e(batch2, 0), _inner(ynext, wnext, ROWS))
    for q in range(Q):
        claim2 = f_add(claim2, ext_mul[4](host_e(batch2, 1 + q), host_e(v2, q)))
    assert_true(_inner(ynext, w2, ROWS) == claim2, "tail query does not carry the batched claim")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
