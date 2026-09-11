"""Poseidon-M31: the expander's test vector, every bit family on the host, the prover round trip for one
and two chunks, a wrong digest rejected."""

from std.testing import assert_equal, assert_true, assert_false, assert_raises, TestSuite
from max.gpu.host import DeviceContext

from caracal7.core.params import CLIENT, Params
from caracal7.core.hash import Blake3
from caracal7.relations import entry, ENTRY, NONE, NO_BASIS
from caracal7.workloads.poseidon import Poseidon, poseidon_m31, poseidon_statement, poseidon_trace, poseidon_public_columns, column_names, ROWS, T, M31
from caracal7.prover import Prover, load_trace, load_advice, load_public
from caracal7.relations import value_bytes
from caracal7.relations.statement import advice
from caracal7.workload import prove_workload, verify_workload

comptime p = CLIENT.grid(64, 368)          # 16 segments of 24 rounds: one chunk
comptime p2 = CLIENT.grid(64, 720)         # 16 segments of 56 rounds: two chunks


def inputs(n: Int) -> List[Int]:
    var v = List[Int](capacity=n)
    for i in range(n):
        v.append((i * 2654435761 + 12345) % M31)
    return v^


def test_reference_vector() raises:
    """The vector of circuit-std-rs/tests/poseidon_m31.rs: eight copies of 114514."""
    var d = poseidon_m31(List[Int](length=8, fill=114514))
    assert_equal(d[0], 1021105124)
    assert_equal(d[1], 1342990709)
    assert_equal(d[15], 1441052621)
    var d2 = poseidon_m31(List[Int](length=16, fill=114514))
    assert_equal(d2[0], 1510043913)
    assert_equal(d2[15], 535675968)


def _at[q: Params](trace: List[UInt8], pubs: List[List[UInt8]], w: Int, col: Int, k1: Int, k2: Int, x1: Int, x2: Int) -> Int:
    comptime N = q.N()
    comptime h1 = q.h1()
    var row = ((x2 + k2) % q.h2()) * h1 + (x1 + k1) % h1
    if col < w:
        return Int(trace[col * N + row])
    return Int(pubs[col - w][row])


def _families_hold[q: Params](fams: List[UInt8], cz: Int, cw: Int, trace: List[UInt8], pubs: List[List[UInt8]]) raises -> Int:
    """Every family without a challenge or a Z read sums to zero on every row; returns the number skipped."""
    comptime N = q.N()
    comptime h1 = q.h1()
    var count = len(fams) // ENTRY
    var families = 0
    for k in range(count):
        families = max(families, entry(fams, k).family + 1)
    var skip = List[Bool](length=families, fill=False)
    for k in range(count):
        var en = entry(fams, k)
        if en.chal != 0 or en.basis != NO_BASIS or (en.col_a >= cw and en.col_a < cw + cz) or (en.col_b != NONE and en.col_b >= cw and en.col_b < cw + cz):
            skip[en.family] = True
    var sums = List[Int](length=families * N, fill=0)
    var skipped = 0
    for i in range(families):
        if skip[i]:
            skipped += 1
    for k in range(count):
        var en = entry(fams, k)
        if skip[en.family]:
            continue
        for x2 in range(q.h2()):
            for x1 in range(h1):
                if (en.mult == 1 and x1 == h1 - 1) or (en.mult == 2 and x2 == q.h2() - 1):
                    continue
                var col_a = en.col_a if en.col_a < cw else en.col_a - cz
                var v = en.coef * _at[q](trace, pubs, cw, col_a, en.dj1_a // 2, en.dj2_a // 2, x1, x2)
                if en.col_b != NONE:
                    var col_b = en.col_b if en.col_b < cw else en.col_b - cz
                    v *= _at[q](trace, pubs, cw, col_b, en.dj1_b // 2, en.dj2_b // 2, x1, x2)
                sums[en.family * N + x2 * h1 + x1] = (sums[en.family * N + x2 * h1 + x1] + v) % 127
    for i in range(families * N):
        if sums[i] != 0:
            raise Error("family " + String(i // N) + " fails at chain " + String((i % N) // h1) + " row " + String(i % h1))
    return skipped


def _check[q: Params](n: Int) raises:
    comptime N = q.N()
    var c = poseidon_statement[q]().compile[q]()
    var w = Poseidon(inputs(n))
    var trace = poseidon_trace[q](c.layout, w.inputs)
    var pubs = List[List[UInt8]]()
    for period in poseidon_public_columns[q](n, poseidon_m31(w.inputs)):
        var full = List[UInt8](capacity=N)
        while len(full) < N:
            full.extend(period.copy())
        pubs.append(full^)
    # only the four Horner accumulators are left to the prover (the chain-end identity lives on the small grid)
    assert_equal(_families_hold[q](c.families, c.shape.columns_z, c.layout.columns_w(), trace, pubs), 4)
    # the S-box output of the last round is the digest: segment s holds lane -s there
    var digest = poseidon_m31(w.inputs)
    var col = c.layout.col("v")
    comptime R = q.h2() // T
    var last = 22 * (1 if n <= 8 else 2)
    for s in range(T):
        var v = 0
        for r in range(ROWS - 1):
            v |= Int(trace[col * N + (s * R + last) * ROWS + r]) << (ROWS - 2 - r)
        assert_equal(v, digest[(T - s) % T])


def test_trace_satisfies_every_bit_family() raises:
    _check[p](8)
    _check[p](3)
    _check[p2](16)


def test_prover_round_trip() raises:
    var ctx = DeviceContext()
    var w = Poseidon(inputs(8))
    var proof = prove_workload[p, Blake3, Poseidon](ctx, w)
    assert_true(verify_workload[p, Blake3, Poseidon](proof^, w, w.public_inputs[p]()))


def test_prover_round_trip_two_chunks() raises:
    var ctx = DeviceContext()
    var w = Poseidon(inputs(16))
    var proof = prove_workload[p2, Blake3, Poseidon](ctx, w)
    assert_true(verify_workload[p2, Blake3, Poseidon](proof^, w, w.public_inputs[p2]()))


def test_wrong_digest_is_rejected() raises:
    """A proof built against a wrong digest (the trace honest, the public dg column not) does not verify."""
    var ctx = DeviceContext()
    var w = Poseidon(inputs(2))
    var c = w.statement[p]().compile[p]()
    var trace = w.trace[p](c.layout)
    var idx = advice[p](c.layout, trace)
    var wrong = w.public_inputs[p]()
    wrong[1] ^= 1
    var data = Poseidon.public_data[p](c.layout, wrong)
    var n_blocks = value_bytes(c.layout.publics, p.h1(), p.h2())
    var blocks = List[UInt8](capacity=n_blocks)
    for i in range(n_blocks):
        blocks.append(data[i])
    var prover = Prover[p, Blake3](ctx, c^)
    load_trace[p, Blake3](ctx, prover, trace)
    load_advice[p, Blake3](ctx, prover, idx)
    load_public[p, Blake3](ctx, prover, blocks)
    var proof = prover.prove(ctx, wrong)
    var ok: Bool
    try:
        ok = verify_workload[p, Blake3, Poseidon](proof^, w, wrong)
    except e:
        ok = False
    assert_false(ok)


def test_wrong_coefficient_is_rejected() raises:
    """A trace whose product coefficients are off by one on the S-box-free lanes satisfies every row family
    and the digest; only the chain-end fingerprint identity rejects it."""
    var ctx = DeviceContext()
    var w = Poseidon(inputs(4))
    var c = w.statement[p]().compile[p]()
    var trace = poseidon_trace[p](c.layout, w.inputs, cheat=True)
    var inputs_ = w.public_inputs[p]()
    var data = Poseidon.public_data[p](c.layout, inputs_)
    var pubs = List[List[UInt8]]()
    for period in poseidon_public_columns[p](4, poseidon_m31(w.inputs)):
        var full = List[UInt8](capacity=p.N())
        while len(full) < p.N():
            full.extend(period.copy())
        pubs.append(full^)
    _ = _families_hold[p](c.families, c.shape.columns_z, c.layout.columns_w(), trace, pubs)
    var idx = advice[p](c.layout, trace)
    var n_blocks = value_bytes(c.layout.publics, p.h1(), p.h2())
    var blocks = List[UInt8](capacity=n_blocks)
    for i in range(n_blocks):
        blocks.append(data[i])
    var prover = Prover[p, Blake3](ctx, c^)
    load_trace[p, Blake3](ctx, prover, trace)
    load_advice[p, Blake3](ctx, prover, idx)
    load_public[p, Blake3](ctx, prover, blocks)
    var proof = prover.prove(ctx, inputs_)
    var ok: Bool
    try:
        ok = verify_workload[p, Blake3, Poseidon](proof^, w, inputs_)
    except e:
        ok = False
    assert_false(ok)


def test_public_data_rejects_a_bad_digest_word() raises:
    var c = poseidon_statement[p]().compile[p]()
    var w = Poseidon(inputs(2))
    var bad = w.public_inputs[p]()
    for i in range(4):
        bad[1 + i] = 0xFF
    with assert_raises():
        _ = Poseidon.public_data[p](c.layout, bad)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
