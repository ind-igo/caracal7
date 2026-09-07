"""The Z stage on the synthetic permutation accumulator, reference profile: the device Z and Z2
against the host definitions, the chain relation Z(next) D = Z N on every row but the chain end,
Z(1, x2) = 1, and the grand product Z2(e2) chain_prod(e2) = 1 (c8 is a permutation of c0)."""

from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceContext, HostBuffer

from caracal7.core.field import E, f_add, ext_mul, ext_inv, ext_one
from caracal7.core.params import Params, CLIENT
from caracal7.core.arena import Arena, Bump
from caracal7.relations.accumulate import ACC, accumulate, derive_chals
from caracal7.relations.ir import Families, CHAL, CHAL_MUL, KIND_LOOKUP, lookup_constant, derived_chals, standard_chals, chal_count
from caracal7.relations.sort import counting_sort
from caracal7.core.bytes import get_u16, list_e, append_u32
from caracal7.relations.synthetic import SYNTHETIC_COLUMNS, synthetic_statement, synthetic_trace

comptime p = CLIENT.grid(72, 32)
comptime N = p.N()
comptime h1 = p.h1()
comptime h2 = p.h2()


def _host(ctx: DeviceContext, l: Span[UInt8, _]) raises -> HostBuffer[DType.uint8]:
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

def host_fp(accs: List[UInt8], k: Int, trace: List[UInt8], N: Int, row: Int, den: Bool) -> E:
    var v = E(0)
    for j in range(get_u16(accs, k * ACC + (4 if den else 2))):
        v[j] = trace[get_u16(accs, k * ACC + (22 if den else 6) + 2 * j) * N + row]
    return v


def host_factor(accs: List[UInt8], k: Int, trace: List[UInt8], N: Int, row: Int, chals: List[UInt8], den: Bool) -> E:
    """N(row) or D(row) of accumulator k from a host trace (column, row), by the descriptor's kind."""
    var fp = host_fp(accs, k, trace, N, row, den)
    if Int(accs[k * ACC + 38]) == KIND_LOOKUP:
        if den:
            return f_add(f_add(list_e(chals, 4), fp), ext_mul[4](list_e(chals, 0), host_fp(accs, k, trace, N, (row + 1) % N, True)))
        return ext_mul[4](list_e(chals, 3), f_add(list_e(chals, 1), fp))
    return f_add(list_e(chals, 2), fp)


def host_accumulate[p: Params](accs: List[UInt8], k: Int, trace: List[UInt8], chals: List[UInt8]) raises -> Tuple[List[UInt8], List[UInt8]]:
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
            acc = ext_mul[4](acc, ext_mul[4](host_factor(accs, k, trace, N, row, chals, False),
                                             ext_inv[4](host_factor(accs, k, trace, N, row, chals, True))))
        acc2 = ext_mul[4](acc2, acc)
    return (z^, z2^)



def _chals(table: List[UInt8]) -> List[UInt8]:
    """Three fixed stage-1 elements and the rows of `table`."""
    var c = List[UInt8](capacity=chal_count(table) * 16)
    for i in range(3 * 16):
        c.append(UInt8((i * 37 + 5) % 127))
    derived_chals(c, table)
    return c^


def test_derivation_table_matches_host() raises:
    """A third row, (1 + beta) delta gamma, on top of the standard two: device and host agree element by element."""
    var table = standard_chals()
    table.extend([CHAL_MUL, 4, 2])
    var chals = _chals(table)
    assert_equal(len(chals), 6 * 16)
    assert_true(list_e(chals, 5) == ext_mul[4](list_e(chals, 4), list_e(chals, 2)), "host row 3")
    var ctx = DeviceContext()
    var bump = Bump()
    var o_chals = bump.alloc(6 * 16)
    var o_table = bump.alloc(len(table))
    var arena = Arena(ctx, bump.used)
    arena.upload(ctx, o_chals, _host(ctx, chals[: 3 * 16]))
    arena.upload(ctx, o_table, _host(ctx, table))
    derive_chals(ctx, arena, o_chals, o_table, 3)
    assert_true(_down(ctx, arena, o_chals, 6 * 16) == chals, "derived challenges differ from the host")


def test_accumulator_matches_host_and_satisfies_the_relations() raises:
    var c = synthetic_statement().compile[p]()
    assert_equal(len(c.shape.accs), 2 * ACC)
    var ctx = DeviceContext()
    var trace = synthetic_trace[p](1)
    for k in range(2):
        var got = _run(ctx, c.shape.accs, k, trace.copy(), SYNTHETIC_COLUMNS, List[UInt8](), 0)
        assert_true(got[1] == got[2], "grand product is not 1: c8 is not a permutation of c0")   # lhs = D(e1, e2)


comptime LOOKUP_K = 37


def test_lookup_accumulator_meets_the_table_constant() raises:
    """Records (c0, c1) = table entry idx[i] = (j, 2 j + 1), every entry used at least once; the sorted
    copy in (c2, c3) from the device sort; Z2(e2) Z(e1, e2) N(e1, e2) = C_T."""
    var f = Families()
    f.lookup(0, 4, [0, 1], [2, 3], 0)
    assert_equal(len(f.accs), ACC)
    assert_equal(Int(f.accs[38]), KIND_LOOKUP)
    var trace = List[UInt8](length=4 * N, fill=0)
    var idx = List[UInt8]()
    var s = 3
    for i in range(N):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        var j = i if i < LOOKUP_K else (s >> 8) % LOOKUP_K      # the dummy rule: rows 0..K-1 cover the table
        trace[i] = UInt8(j)
        trace[N + i] = UInt8(2 * j + 1)
        append_u32(idx, j)
    var table = List[UInt8]()
    for j in range(LOOKUP_K):
        table.append(UInt8(j))
        table.append(UInt8(2 * j + 1))
    var ctx = DeviceContext()
    var got = _run(ctx, f.accs, 0, trace^, 4, idx, LOOKUP_K)
    var c_t = lookup_constant(table, 2, _chals(standard_chals()))
    assert_true(got[1] == c_t, "lookup boundary is not C_T")
    assert_true(c_t != ext_one[4](), "vacuous")


def _run(ctx: DeviceContext, accs: List[UInt8], k: Int, var trace: List[UInt8], columns: Int, idx: List[UInt8], table_k: Int) raises -> Tuple[List[UInt8], E, E]:
    """Z stage of descriptor k on `trace` against the host definitions; a lookup (idx non-empty holds the
    advice bytes) is sorted on device first and the host trace takes the sorted columns back. Returns
    (Z bytes, Z2(e2) Z(e1, e2) N(e1, e2), D(e1, e2))."""
    var table = standard_chals()
    var chals = _chals(table)
    var bump = Bump()
    var o_trace = bump.alloc(columns * N)
    var o_acc = bump.alloc(ACC)
    var o_chals = bump.alloc(len(chals))
    var o_table = bump.alloc(len(table))
    var o_num = bump.alloc(N * 16)
    var o_den = bump.alloc(N * 16)
    var o_scratch = bump.alloc(N * 16)
    var o_z = bump.alloc(N * 16)
    var o_prod = bump.alloc(h2 * 16)
    var o_z2 = bump.alloc((h2 + 1) * 16)   # k_z2 stores a trailing 1
    var o_nend = bump.alloc(h2 * 16)
    var o_dend = bump.alloc(h2 * 16)
    var o_idx = bump.alloc(4 * N)
    var o_bins = bump.alloc(4 * (table_k + 1))
    var o_cursor = bump.alloc(4 * table_k)
    var arena = Arena(ctx, bump.used)
    arena.upload(ctx, o_trace, _host(ctx, trace))
    var one_acc = List[UInt8](capacity=ACC)
    for i in range(ACC):
        one_acc.append(accs[k * ACC + i])
    arena.upload(ctx, o_acc, _host(ctx, one_acc))
    arena.upload(ctx, o_chals, _host(ctx, chals[: 3 * 16]))
    arena.upload(ctx, o_table, _host(ctx, table))
    derive_chals(ctx, arena, o_chals, o_table, len(table) // CHAL)
    if len(idx) > 0:
        arena.upload(ctx, o_idx, _host(ctx, idx))
        counting_sort[p](ctx, arena, o_trace, o_acc, o_idx, o_bins, o_cursor, table_k)
        trace = _down(ctx, arena, o_trace, columns * N)
    accumulate[p](ctx, arena, o_trace, o_acc, o_chals, o_num, o_den, o_scratch, o_z, o_prod, o_z2, o_nend, o_dend)
    var z = _down(ctx, arena, o_z, N * 16)
    var z2 = _down(ctx, arena, o_z2, (h2 + 1) * 16)
    var prod = _down(ctx, arena, o_prod, h2 * 16)
    var num = _down(ctx, arena, o_num, N * 16)
    var den = _down(ctx, arena, o_den, N * 16)
    assert_true(_down(ctx, arena, o_chals, len(chals)) == chals, "derived challenges differ from the host")
    for row in [0, 1, h1 - 1, h1, N - 2, N - 1]:
        assert_true(list_e(num, row) == host_factor(accs, k, trace, N, row, chals, False), "N differs from the host at row " + String(row))
        assert_true(list_e(den, row) == host_factor(accs, k, trace, N, row, chals, True), "D differs from the host at row " + String(row))

    var want = host_accumulate[p](accs, k, trace, chals)
    assert_true(z == want[0], "Z differs from the host definition")
    assert_true(z2[: h2 * 16] == want[1], "Z2 differs from the host definition")
    assert_true(list_e(z2, h2) == ext_one[4](), "trailing Z2 element is not 1")
    var one = ext_one[4]()
    for x2 in range(h2):
        assert_true(list_e(z, x2 * h1) == one, "chain start is not 1")
        for x1 in range(h1 - 1):
            var row = x2 * h1 + x1
            var lhs = ext_mul[4](list_e(z, row + 1), host_factor(accs, k, trace, N, row, chals, True))
            var rhs = ext_mul[4](list_e(z, row), host_factor(accs, k, trace, N, row, chals, False))
            assert_true(lhs == rhs, "chain relation fails")
    assert_true(list_e(z2, 0) == one, "Z2(1) is not 1")
    for x2 in range(h2 - 1):
        assert_true(list_e(z2, x2 + 1) == ext_mul[4](list_e(z2, x2), list_e(prod, x2)), "Z2 recurrence fails")
    assert_true(list_e(z, N - 1) != one, "vacuous")
    var nend = _down(ctx, arena, o_nend, h2 * 16)
    var dend = _down(ctx, arena, o_dend, h2 * 16)
    for x2 in range(h2):
        assert_true(list_e(nend, x2) == host_factor(accs, k, trace, N, x2 * h1 + h1 - 1, chals, False), "chain-end N")
        assert_true(list_e(dend, x2) == host_factor(accs, k, trace, N, x2 * h1 + h1 - 1, chals, True), "chain-end D")
    var boundary = ext_mul[4](ext_mul[4](list_e(z2, h2 - 1), list_e(z, N - 1)), list_e(nend, h2 - 1))
    return (z^, boundary, list_e(dend, h2 - 1))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
