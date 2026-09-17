"""The Z stage on the synthetic permutation accumulator, reference profile: the device Z and Z2
against the host definitions, the chain relation Z(next) D = Z N on every row but the chain end,
Z(1, x2) = 1, and the grand product Z2(e2) chain_prod(e2) = 1 (c8 is a permutation of c0)."""

from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceContext, HostBuffer

from caracal7.core.field import E, f_add, f_sub, f_mul, ext_mul, ext_inv, ext_one, E_LEVEL, E_BYTES
from caracal7.core.params import Params, CLIENT
from caracal7.core.arena import Arena, Bump
from caracal7.relations.accumulate import ACC, AccLayout, accumulate, horner, derive_chals
from caracal7.relations.ir import Families, acc_kind, ENTRY, CHAL, CHAL_MUL, KIND_LOOKUP, KIND_HORNER, entry, lookup_constant, derived_chals, standard_chals, chal_count, horner_chain_end, selector_values
from caracal7.relations.statement import Statement, Term, Compiled, BIT, BYTE
from caracal7.prover import Prover, load_trace, load_public
from caracal7.verifier import verify
from caracal7.core.hash import Blake3
from caracal7.relations.sort import counting_sort
from caracal7.core.bytes import get_u16, list_e, append_u32
from caracal7.workloads.synthetic import SYNTHETIC_COLUMNS, synthetic_statement, synthetic_trace, horner_statement, horner_trace

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
    if acc_kind(accs, k) == KIND_LOOKUP:
        if den:
            return f_add(f_add(list_e(chals, 4), fp), ext_mul[E_LEVEL](list_e(chals, 0), host_fp(accs, k, trace, N, (row + 1) % N, True)))
        return ext_mul[E_LEVEL](list_e(chals, 3), f_add(list_e(chals, 1), fp))
    return f_add(list_e(chals, 2), fp)


def host_accumulate[p: Params](accs: List[UInt8], k: Int, trace: List[UInt8], chals: List[UInt8]) raises -> Tuple[List[UInt8], List[UInt8]]:
    """(Z as (row, e) bytes, Z2 as (x2, e) bytes) by the definitions, for the tests."""
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    comptime N = p.N()
    var z = List[UInt8](length=N * E_BYTES, fill=0)
    var z2 = List[UInt8](length=h2 * E_BYTES, fill=0)
    var acc2 = ext_one[E_LEVEL]()
    for x2 in range(h2):
        for t in range(E_BYTES):
            z2[x2 * E_BYTES + t] = acc2[t]
        var acc = ext_one[E_LEVEL]()
        for x1 in range(h1):
            var row = x2 * h1 + x1
            for t in range(E_BYTES):
                z[row * E_BYTES + t] = acc[t]
            acc = ext_mul[E_LEVEL](acc, ext_mul[E_LEVEL](host_factor(accs, k, trace, N, row, chals, False),
                                             ext_inv[E_LEVEL](host_factor(accs, k, trace, N, row, chals, True))))
        acc2 = ext_mul[E_LEVEL](acc2, acc)
    return (z^, z2^)



def _chals(table: List[UInt8]) -> List[UInt8]:
    """Three fixed stage-1 elements and the rows of `table`."""
    var c = List[UInt8](capacity=chal_count(table) * E_BYTES)
    for i in range(3 * E_BYTES):
        c.append(UInt8((i * 37 + 5) % 127))
    derived_chals(c, table)
    return c^


def test_derivation_table_matches_host() raises:
    """A third row, (1 + beta) delta gamma, on top of the standard two: device and host agree element by element."""
    var table = standard_chals()
    table.extend([CHAL_MUL, 4, 2])
    var chals = _chals(table)
    assert_equal(len(chals), 6 * E_BYTES)
    assert_true(list_e(chals, 5) == ext_mul[E_LEVEL](list_e(chals, 4), list_e(chals, 2)), "host row 3")
    var ctx = DeviceContext()
    var bump = Bump()
    var o_chals = bump.alloc(6 * E_BYTES)
    var o_table = bump.alloc(len(table))
    var arena = Arena(ctx, bump.used)
    arena.upload(ctx, o_chals, _host(ctx, chals[: 3 * E_BYTES]))
    arena.upload(ctx, o_table, _host(ctx, table))
    derive_chals(ctx, arena, o_chals, o_table, 3)
    assert_true(_down(ctx, arena, o_chals, 6 * E_BYTES) == chals, "derived challenges differ from the host")
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks (decisions.md 2026-09-16)


def test_accumulator_matches_host_and_satisfies_the_relations() raises:
    var c = synthetic_statement().compile[p]()
    assert_equal(len(c.shape.accs), 2 * ACC)
    var ctx = DeviceContext()
    var trace = synthetic_trace[p](1)
    for k in range(2):
        var got = _run(ctx, c.shape.accs, k, trace.copy(), SYNTHETIC_COLUMNS, List[UInt8](), 0)
        assert_true(got[1] == got[2], "grand product is not 1: c8 is not a permutation of c0")   # lhs = D(e1, e2)
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks (decisions.md 2026-09-16)


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
    assert_true(c_t != ext_one[E_LEVEL](), "vacuous")
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks (decisions.md 2026-09-16)


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
    var A = AccLayout.__init__[p](bump, 1, 1, 0)
    var o_idx = bump.alloc(4 * N)
    var o_bins = bump.alloc(4 * (table_k + 1))
    var o_cursor = bump.alloc(4 * table_k)
    var arena = Arena(ctx, bump.used)
    arena.upload(ctx, o_trace, _host(ctx, trace))
    var one_acc = List[UInt8](capacity=ACC)
    for i in range(ACC):
        one_acc.append(accs[k * ACC + i])
    arena.upload(ctx, o_acc, _host(ctx, one_acc))
    arena.upload(ctx, o_chals, _host(ctx, chals[: 3 * E_BYTES]))
    arena.upload(ctx, o_table, _host(ctx, table))
    derive_chals(ctx, arena, o_chals, o_table, len(table) // CHAL)
    if len(idx) > 0:
        arena.upload(ctx, o_idx, _host(ctx, idx))
        counting_sort[p](ctx, arena, o_trace, o_acc, o_idx, o_bins, o_cursor, table_k)
        trace = _down(ctx, arena, o_trace, columns * N)
    accumulate[p](ctx, arena, o_trace, o_acc, o_chals, A, 0, 0)
    var z = _down(ctx, arena, A.zval, N * E_BYTES)
    var z2 = _down(ctx, arena, A.z2, (h2 + 1) * E_BYTES)
    var prod = _down(ctx, arena, A.chain_prod, h2 * E_BYTES)
    var num = _down(ctx, arena, A.num, N * E_BYTES)
    var den = _down(ctx, arena, A.den, N * E_BYTES)
    assert_true(_down(ctx, arena, o_chals, len(chals)) == chals, "derived challenges differ from the host")
    for row in [0, 1, h1 - 1, h1, N - 2, N - 1]:
        assert_true(list_e(num, row) == host_factor(accs, k, trace, N, row, chals, False), "N differs from the host at row " + String(row))
        assert_true(list_e(den, row) == host_factor(accs, k, trace, N, row, chals, True), "D differs from the host at row " + String(row))

    var want = host_accumulate[p](accs, k, trace, chals)
    assert_true(z == want[0], "Z differs from the host definition")
    assert_true(z2[: h2 * E_BYTES] == want[1], "Z2 differs from the host definition")
    assert_true(list_e(z2, h2) == ext_one[E_LEVEL](), "trailing Z2 element is not 1")
    var one = ext_one[E_LEVEL]()
    for x2 in range(h2):
        assert_true(list_e(z, x2 * h1) == one, "chain start is not 1")
        for x1 in range(h1 - 1):
            var row = x2 * h1 + x1
            var lhs = ext_mul[E_LEVEL](list_e(z, row + 1), host_factor(accs, k, trace, N, row, chals, True))
            var rhs = ext_mul[E_LEVEL](list_e(z, row), host_factor(accs, k, trace, N, row, chals, False))
            assert_true(lhs == rhs, "chain relation fails")
    assert_true(list_e(z2, 0) == one, "Z2(1) is not 1")
    for x2 in range(h2 - 1):
        assert_true(list_e(z2, x2 + 1) == ext_mul[E_LEVEL](list_e(z2, x2), list_e(prod, x2)), "Z2 recurrence fails")
    assert_true(list_e(z, N - 1) != one, "vacuous")
    var nend = _down(ctx, arena, A.n_end, h2 * E_BYTES)
    var dend = _down(ctx, arena, A.d_end, h2 * E_BYTES)
    for x2 in range(h2):
        assert_true(list_e(nend, x2) == host_factor(accs, k, trace, N, x2 * h1 + h1 - 1, chals, False), "chain-end N")
        assert_true(list_e(dend, x2) == host_factor(accs, k, trace, N, x2 * h1 + h1 - 1, chals, True), "chain-end D")
    var boundary = ext_mul[E_LEVEL](ext_mul[E_LEVEL](list_e(z2, h2 - 1), list_e(z, N - 1)), list_e(nend, h2 - 1))
    return (z^, boundary, list_e(dend, h2 - 1))


def _selected_statement() raises -> Statement:
    """horner_statement with every ingest term times the public selector g, a public factor on chain 0 and a wire
    between chains 3 and 4: the accumulators stay at 0 on the chains where g is 0, so the ungated chain end
    holds whatever those chains hold."""
    var st = Statement()
    st.col("a", BIT)
    st.col("b", BIT)
    st.col("c", BYTE)
    st.pub("g", 1)
    var dg = st.derived(CHAL_MUL, 1, 2)
    st.horner("ra", [Term(1, st.read("a", k1=1), st.read("g"))], scale=2)
    st.horner("rb", [Term(3, st.read("b"), st.read("g"), chal=1)], scale=2)
    st.horner("rc", [Term(1, st.read("c"), st.read("g"))], scale=2)
    st.chain_end("mul", [Term(1, st.read("ra"), st.read("rb")), Term(-3, st.read("rc"), chal=dg)])
    var sc = st.slot("rc")
    st.wire(sc, 3, sc, 4)
    st.public_factor("pc", "rc", sc, 0)
    return st^


def _verdict(proof: List[UInt8], c: Compiled, public: List[UInt8]) raises -> String:
    var fam = c.families.copy()
    try:
        _ = verify[p, Blake3](proof.copy(), c.shape, List[UInt8](), fam, public)
    except e:
        return String(e)
    return String("accepted")


def test_selected_ingest_masks_the_chains_of_another_group() raises:
    """The selector g is 1 on the lower half of the chains: the product trace there, garbage above (random
    bytes in c, the bits left as they are), and the proof is accepted; with g raised on one garbage chain the
    residual fails; the public factor is fingerprinted with the selector on its own chain."""
    var ctx = DeviceContext()
    var c = _selected_statement().compile[p]()
    var trace = horner_trace[p](1)
    for col in range(3):                     # chain 4 copies chain 3: the wire
        for x1 in range(h1):
            trace[col * N + 4 * h1 + x1] = trace[col * N + 3 * h1 + x1]
    var s = 7
    for x2 in range(h2 // 2, h2):
        for x1 in range(h1):
            s = (s * 1103515245 + 12345) & 0x7FFFFFFF
            trace[2 * N + x2 * h1 + x1] = UInt8((s >> 8) % 127)
    var block = List[UInt8](length=N, fill=0)
    for x2 in range(h2 // 2):
        for x1 in range(h1):
            block[x2 * h1 + x1] = 1
    var prover = Prover[p, Blake3](ctx, _selected_statement().compile[p]().take_shape(), c.families.copy())
    load_trace[p, Blake3](ctx, prover, trace)
    load_public[p, Blake3](ctx, prover, block)
    var proof = prover.prove(ctx, List[UInt8]())
    var public = block.copy()
    for x1 in range(h1):
        public.append(trace[2 * N + x1])          # the factor: c on chain 0
    assert_equal(_verdict(proof, c, public), "accepted")
    var raised = public.copy()
    for x1 in range(h1):
        raised[(h2 - 1) * h1 + x1] = 1
    assert_equal(_verdict(proof, c, raised), "residual identity fails at z")
    var wrong = public.copy()
    wrong[N + 3] = (wrong[N + 3] + 1) % 127
    assert_equal(_verdict(proof, c, wrong), "wiring grand product is not the public factor")
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks (decisions.md 2026-09-16)


def test_selected_ingest_device_matches_the_fingerprint() raises:
    """Device R of the selected accumulators: 0 at every chain end where g is 0, and at chain 0 the verifier's
    fingerprint of chain 0's column with the selector."""
    var c = _selected_statement().compile[p]()
    var chals = _chals(c.shape.chals.copy())
    var trace = horner_trace[p](1)
    var block = List[UInt8](length=N, fill=0)
    for x2 in range(h2 // 2):
        for x1 in range(h1):
            block[x2 * h1 + x1] = 1
    var ctx = DeviceContext()
    var bump = Bump()
    var o_trace = bump.alloc(3 * N)
    var o_fam = bump.alloc(len(c.families))
    var o_acc = bump.alloc(3 * ACC)
    var o_chals = bump.alloc(len(chals))
    var o_pub = bump.alloc(N)
    var A = AccLayout.__init__[p](bump, 3, 0, 0)
    var arena = Arena(ctx, bump.used)
    arena.upload(ctx, o_trace, _host(ctx, trace))
    arena.upload(ctx, o_fam, _host(ctx, c.families))
    arena.upload(ctx, o_acc, _host(ctx, c.shape.accs))
    arena.upload(ctx, o_chals, _host(ctx, chals))
    arena.upload(ctx, o_pub, _host(ctx, block))
    var pub_at = c.shape.columns_w + c.shape.columns_z
    horner[p](ctx, arena, o_trace, o_fam, o_acc, o_chals, A, 3, o_pub, pub_at)
    for k in range(3):
        var got = _down(ctx, arena, A.zval_at(k), N * E_BYTES)
        for x2 in range(h2 // 2, h2):
            assert_true(list_e(got, x2 * h1 + h1 - 1) == E(0), "R is not neutral on an unselected chain")
        var cols = List[UInt8]()
        for x1 in range(h1):
            cols.append(trace[k * N + x1])
        var sel = selector_values[p](c.families, c.shape.accs, k, c.shape.publics, block, pub_at, 0)
        assert_true(horner_chain_end[p](c.families, c.shape.accs, k, cols, chals, sel) == list_e(got, h1 - 1), "fingerprint differs at chain 0 for accumulator " + String(k))
        assert_true(list_e(got, h1 - 1) != E(0), "vacuous")
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks (decisions.md 2026-09-16)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def host_horner[p: Params](families: List[UInt8], accs: List[UInt8], k: Int, trace: List[UInt8], chals: List[UInt8]) -> List[UInt8]:
    """R as (row, e) bytes by the definition in accumulate.mojo."""
    comptime h1 = p.h1()
    comptime N = p.N()
    var out = List[UInt8](length=N * E_BYTES, fill=0)
    var first = get_u16(accs, k * ACC + 2)
    var count = get_u16(accs, k * ACC + 4)
    var scale = ext_one[E_LEVEL]()
    if accs[k * ACC + 7] != 0:
        scale = list_e(chals, Int(accs[k * ACC + 7]) - 1)
    for x2 in range(p.h2()):
        var r = E(0)
        r[0] = accs[k * ACC + 6]
        for x1 in range(h1):
            var row = x2 * h1 + x1
            for t in range(E_BYTES):
                out[row * E_BYTES + t] = r[t]
            var s = E(0)
            for i in range(first, first + count):
                var en = entry(families, i)
                var v = E(0)
                v[0] = trace[en.col_a * N + x2 * h1 + (x1 + en.dj1_a // 2) % h1]
                v = f_mul(v, E(UInt8(en.coef)))
                if en.chal != 0:
                    v = ext_mul[E_LEVEL](v, list_e(chals, en.chal - 1))
                s = f_add(s, v)
            r = f_sub(ext_mul[E_LEVEL](scale, r), s)
    return out^


def test_horner_chain_end_fingerprints_a_chain() raises:
    """The verifier's public-factor fingerprint is the host recurrence's chain-end value: ingest columns of one
    chain in entry order (the shifted read included)."""
    var c = horner_statement().compile[p]()
    var chals = _chals(c.shape.chals.copy())
    var trace = horner_trace[p](5)
    for k in range(3):
        var full = host_horner[p](c.families, c.shape.accs, k, trace, chals)
        var count = get_u16(c.shape.accs, k * ACC + 4)
        for x2 in [0, 3, h2 - 1]:
            var cols = List[UInt8]()
            for i in range(count):
                var col = entry(c.families, get_u16(c.shape.accs, k * ACC + 2) + i).col_a
                for x1 in range(h1):
                    cols.append(trace[col * N + x2 * h1 + x1])
            assert_true(horner_chain_end[p](c.families, c.shape.accs, k, cols, chals) == list_e(full, x2 * h1 + h1 - 1), "fingerprint differs at chain " + String(x2))


def test_horner_accumulator_matches_host_and_meets_the_chain_end() raises:
    """Device R against the host definition for the three accumulators of the mulmod instance, and
    R_A R_B = 3 delta gamma R_C at every chain end."""
    var c = horner_statement().compile[p]()
    assert_equal(len(c.shape.accs), 3 * ACC)
    var table = c.shape.chals.copy()
    var chals = _chals(table)
    var trace = horner_trace[p](1)
    var ctx = DeviceContext()
    var bump = Bump()
    var o_trace = bump.alloc(3 * N)
    var o_fam = bump.alloc(len(c.families))
    var o_acc = bump.alloc(3 * ACC)
    var o_chals = bump.alloc(len(chals))
    var A = AccLayout.__init__[p](bump, 3, 0, 0)
    var arena = Arena(ctx, bump.used)
    arena.upload(ctx, o_trace, _host(ctx, trace))
    arena.upload(ctx, o_fam, _host(ctx, c.families))
    arena.upload(ctx, o_acc, _host(ctx, c.shape.accs))
    arena.upload(ctx, o_chals, _host(ctx, chals))
    var ends = List[E]()
    horner[p](ctx, arena, o_trace, o_fam, o_acc, o_chals, A, 3, o_trace, 1 << 15)   # no public columns: the tile is never read
    for k in range(3):
        assert_equal(acc_kind(c.shape.accs, k), KIND_HORNER)
        var got = _down(ctx, arena, A.zval_at(k), N * E_BYTES)
        assert_true(got == host_horner[p](c.families, c.shape.accs, k, trace, chals), "R differs from the host for accumulator " + String(k))
        for x2 in range(h2):
            ends.append(list_e(got, x2 * h1 + h1 - 1))
    var w = f_mul(ext_mul[E_LEVEL](list_e(chals, 1), list_e(chals, 2)), E(3))     # 3 delta gamma
    for x2 in range(h2):
        assert_true(ext_mul[E_LEVEL](ends[x2], ends[h2 + x2]) == ext_mul[E_LEVEL](w, ends[2 * h2 + x2]), "chain end fails at x2 = " + String(x2))
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks (decisions.md 2026-09-16)
