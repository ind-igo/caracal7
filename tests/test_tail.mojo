"""Tail kernels against the host formulas, reference profile: leaf points, expected symbols from
committed rows against the host encoder of the folded message, the materialized query against the batched claim (the identity the sumcheck
proves), the three rounds against the verifier's checks, and the fold."""

from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceContext, HostBuffer

from caracal7.core.field import F4, E, f_add, ext_mul, E_LEVEL, E_BYTES
from caracal7.verifier import encode_at
from caracal7.core.params import CLIENT
from caracal7.core.tables import RsDomain, RsTables, build_rs_tables
from caracal7.core.arena import Arena, Bump
from caracal7.pcs.tail import DOM_BYTES, ROUND_THREADS, domain_bytes, points, tail_encode, tail_materialize, tail_round, tail_fold, power_table_len, TAIL_F4
from caracal7.core.bytes import list_e
from caracal7.pcs.tail import host_r3, rbar_at, tail_encode_at, fold8_host, quadratic_at

comptime p = CLIENT.grid(72, 32)
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
        acc = f_add(acc, ext_mul[E_LEVEL](list_e(a, i), list_e(b, i)))
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
    var y = _rand(N * E_BYTES, 1, 127)
    var running = _rand(N * E_BYTES, 2, 127)
    var batch = _rand((1 + 4 * Q) * E_BYTES, 3, 127)
    var r = _rand(3 * E_BYTES, 4, 127)
    var pos_list = List[Int]()
    var pos2_list = List[Int]()
    var s = 99
    for _ in range(Q):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        pos_list.append(s % p.L())
        pos2_list.append((s >> 3) % (L0_TAIL * M_TAIL))
    var batch2 = _rand((1 + Q) * E_BYTES, 5, 127)

    var bump = Bump()
    var o_y = bump.alloc(N * E_BYTES)
    var o_run = bump.alloc(N * E_BYTES)
    var o_batch = bump.alloc((1 + 4 * Q) * E_BYTES)
    var o_r = bump.alloc(3 * E_BYTES)
    var o_pos = bump.alloc(Q * 4)
    var o_pos2 = bump.alloc(Q * 4)
    var o_dom1 = bump.alloc(DOM_BYTES)
    var o_dom2 = bump.alloc(DOM_BYTES)
    var o_pts = bump.alloc(Q * 4)
    var o_pts2 = bump.alloc(Q * 4)
    var o_w = bump.alloc(N * E_BYTES)
    var o_partial = bump.alloc(3 * ROUND_THREADS * E_BYTES)
    var o_rounds = bump.alloc(9 * E_BYTES)
    var o_ynext = bump.alloc(ROWS * E_BYTES)
    var o_wnext = bump.alloc(ROWS * E_BYTES)
    var o_batch2 = bump.alloc((1 + Q) * E_BYTES)
    var o_w2 = bump.alloc(ROWS * E_BYTES)
    var o_ptab = bump.alloc(Q * power_table_len[p]() * 4)
    var rs = RsTables(bump.alloc(0), L0_TAIL, M_TAIL, ROWS)
    _ = bump.alloc(rs.bytes)
    var o_etmp = bump.alloc(L0_TAIL * TAIL_F4 * 4)
    var o_code = bump.alloc(M_TAIL * L0_TAIL * TAIL_F4 * 4)
    var arena = Arena(ctx, bump.used)
    arena.upload(ctx, rs.base, build_rs_tables(ctx, rs, dom2, ROWS))

    _up(ctx, arena, o_y, y)
    _up(ctx, arena, o_run, running)
    _up(ctx, arena, o_batch, batch)
    _up(ctx, arena, o_r, r)
    _up(ctx, arena, o_pos, _u32s(pos_list))
    _up(ctx, arena, o_pos2, _u32s(pos2_list))
    _up(ctx, arena, o_dom1, domain_bytes(dom1))
    _up(ctx, arena, o_dom2, domain_bytes(dom2))
    _up(ctx, arena, o_batch2, batch2)

    # level-1 functionals on y
    points(ctx, arena, o_pos, Q, o_dom1, p.L0, o_pts)
    tail_encode(ctx, arena, o_y, ROWS, L0_TAIL, M_TAIL, o_etmp, o_code, rs)   # y as a tail level: Mat(y) (ROWS, 8, e)
    tail_materialize[p](ctx, arena, True, o_run, o_batch, o_pts, Q, N, o_w, o_ptab)
    for d in range(3):
        tail_round(ctx, arena, o_w, o_y, N, d, o_r, o_partial, o_rounds + d * 3 * E_BYTES)
    tail_fold(ctx, arena, o_y, ROWS, o_r, o_ynext)
    tail_fold(ctx, arena, o_w, ROWS, o_r, o_wnext)
    # tail-level functionals on y_next
    points(ctx, arena, o_pos2, Q, o_dom2, L0_TAIL, o_pts2)
    tail_materialize[p](ctx, arena, False, o_wnext, o_batch2, o_pts2, Q, ROWS, o_w2, o_ptab)

    var pts = _down(ctx, arena, o_pts, Q * 4)
    var w = _down(ctx, arena, o_w, N * E_BYTES)
    var rounds = _down(ctx, arena, o_rounds, 9 * E_BYTES)
    var ynext = _down(ctx, arena, o_ynext, ROWS * E_BYTES)
    var wnext = _down(ctx, arena, o_wnext, ROWS * E_BYTES)
    var pts2 = _down(ctx, arena, o_pts2, Q * 4)
    var w2 = _down(ctx, arena, o_w2, ROWS * E_BYTES)
    var code = _down(ctx, arena, o_code, M_TAIL * L0_TAIL * TAIL_F4 * 4)

    for q in range(Q):
        var pt = F4(pts[4 * q], pts[4 * q + 1], pts[4 * q + 2], pts[4 * q + 3])
        assert_true(pt == dom1.point(pos_list[q]), "point mismatch")
    # the batched claim: <y, w~> = batch_0 <y, running> + sum_q batch_q g_q(y), g_q the level-1 functionals at pt_q
    var claim = ext_mul[E_LEVEL](list_e(batch, 0), _inner(y, running, N))
    for q in range(Q):
        var enc = encode_at[p](y, F4(pts[4 * q], pts[4 * q + 1], pts[4 * q + 2], pts[4 * q + 3]))
        for tau in range(4):
            claim = f_add(claim, ext_mul[E_LEVEL](list_e(batch, 1 + 4 * q + tau), enc[tau]))
    assert_true(_inner(y, w, N) == claim, "materialized query does not carry the batched claim")
    # sumcheck: s_d(0) + s_d(1) = previous, ending at <fold(y), fold(w~)>
    var prev = claim
    for d in range(3):
        assert_true(f_add(list_e(rounds, 3 * d), list_e(rounds, 3 * d + 1)) == prev, "round " + String(d) + " sum mismatch")
        prev = quadratic_at(rounds, 3 * d, list_e(r, d))
    assert_true(ynext == fold8_host(y, ROWS, r), "fold mismatch")
    assert_true(wnext == fold8_host(w, ROWS, r), "query fold mismatch")
    assert_true(_inner(ynext, wnext, ROWS) == prev, "folded claim mismatch")
    assert_true(prev != E(0), "vacuous")
    # tail level
    var claim2 = ext_mul[E_LEVEL](list_e(batch2, 0), _inner(ynext, wnext, ROWS))
    for q in range(Q):
        var pt = F4(pts2[4 * q], pts2[4 * q + 1], pts2[4 * q + 2], pts2[4 * q + 3])
        assert_true(pt == dom2.point(pos2_list[q]), "coset point mismatch")
        # the encoder is linear: the r_bar-combined row of Enc(y) at s is Enc(fold(y))(pt), the verifier's expected symbol
        var row = E(0)
        for a in range(8):
            row = f_add(row, ext_mul[E_LEVEL](rbar_at(host_r3(r), a, 3), list_e(code, pos2_list[q] * 8 + a)))
        assert_true(row == tail_encode_at(ynext, ROWS, pt), "expected symbol from the committed row mismatch")
        claim2 = f_add(claim2, ext_mul[E_LEVEL](list_e(batch2, 1 + q), tail_encode_at(ynext, ROWS, pt)))
    assert_true(_inner(ynext, w2, ROWS) == claim2, "tail query does not carry the batched claim")
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks (decisions.md 2026-09-16)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
