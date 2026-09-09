"""Keccak-256 on the builder: the reference digest, the trace against every compiled family on the host, and
the prover round trip on the 64 x 24 grid with a wrong digest rejected."""

from std.testing import assert_equal, assert_true, assert_false, TestSuite
from max.gpu.host import DeviceContext

from caracal7.core.params import CLIENT, Params
from caracal7.core.hash import Blake3
from caracal7.relations import entry, ENTRY, NONE, NO_BASIS
from caracal7.workloads.keccak import Keccak, keccak256, keccak_statement, keccak_trace, keccak_public_values, chain_words, digest_words, rc, KECCAK_COLUMNS, ABSORB, LANES, ROUNDS
from caracal7.prover import Prover, load_trace, load_advice, load_public
from caracal7.relations import value_bytes
from caracal7.relations.statement import advice
from caracal7.workload import prove_workload, verify_workload

comptime p = CLIENT.grid(64, 24)


def hex(bytes: List[UInt8]) -> String:
    var s = String("")
    for b in bytes:
        s += String("0123456789abcdef"[byte=Int(b >> 4)]) + String("0123456789abcdef"[byte=Int(b & 15)])
    return s^


def message(n: Int) -> List[UInt8]:
    var m = List[UInt8](capacity=n)
    for i in range(n):
        m.append(UInt8((i * 37 + 11) % 256))
    return m^


def test_reference_digests() raises:
    assert_equal(hex(keccak256(List[UInt8]())), "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470")
    assert_equal(hex(keccak256([UInt8(97), UInt8(98), UInt8(99)])), "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45")


def test_trace_satisfies_every_family() raises:
    """Every entry summed per family is zero on every row its gate admits, public reads included; on the
    exact grid and on one with 24 idle chains before the live ones. 28 opening points: z, its two restriction
    lines, 24 cyclic shifts, the next-chain point (no accumulators, so no boundary points)."""
    _check_families[p](128)
    _check_families[CLIENT.grid(64, 48)](128)


def test_idle_chains_keep_the_digest() raises:
    """With 24 idle chains first, the last chain's x lanes are still the digest (lane 0 before RC[23])."""
    var msg = message(128)
    var w = chain_words(msg, 48)
    var d = digest_words(msg)
    var off = 47 * KECCAK_COLUMNS + KECCAK_COLUMNS - LANES
    assert_equal(w[off] ^ rc()[ROUNDS - 1], d[0])
    for l in range(1, 4):
        assert_equal(w[off + l], d[l])
    for i in range(24 * KECCAK_COLUMNS):
        assert_equal(w[i], 0)


def _check_families[q: Params](bytes: Int) raises:
    comptime N = q.N()
    comptime h1 = q.h1()
    comptime h2 = q.h2()
    var c = keccak_statement().compile[q]()
    assert_equal(c.shape.points, 28)
    var msg = message(bytes)
    var trace = keccak_trace[q](c.layout, msg)
    var pubs = List[List[UInt8]]()
    for l in range(ABSORB):
        pubs.append(keccak_public_values[q](msg, l))
    var w = c.layout.columns_w()
    var count = len(c.families) // ENTRY
    var families = 0
    for k in range(count):
        families = max(families, entry(c.families, k).family + 1)
    var sums = List[Int](length=families * N, fill=0)
    for k in range(count):
        var en = entry(c.families, k)
        assert_equal(en.chal, 0)
        assert_equal(en.basis, NO_BASIS)
        for x2 in range(h2):
            for x1 in range(h1):
                if (en.mult == 1 and x1 == h1 - 1) or (en.mult == 2 and x2 == h2 - 1):
                    continue
                var v = en.coef * _at[q](trace, pubs, w, en.col_a, en.dj1_a // 2, en.dj2_a // 2, x1, x2)
                if en.col_b != NONE:
                    v *= _at[q](trace, pubs, w, en.col_b, en.dj1_b // 2, en.dj2_b // 2, x1, x2)
                sums[en.family * N + x2 * h1 + x1] = (sums[en.family * N + x2 * h1 + x1] + v) % 127
    for i in range(families * N):
        if sums[i] != 0:
            raise Error("family " + String(i // N) + " fails at row " + String(i % N))


def _at[q: Params](trace: List[UInt8], pubs: List[List[UInt8]], w: Int, col: Int, k1: Int, k2: Int, x1: Int, x2: Int) -> Int:
    """A read at (x1 + k1, x2 + k2); public columns follow the W columns (no Z columns in this statement)."""
    comptime N = q.N()
    comptime h1 = q.h1()
    var row = ((x2 + k2) % q.h2()) * h1 + (x1 + k1) % h1
    if col < w:
        return Int(trace[col * N + row])
    return Int(pubs[col - w][row])


def test_prover_round_trip() raises:
    var ctx = DeviceContext()
    var w = Keccak(message(128))
    var proof = prove_workload[p, Blake3, Keccak](ctx, w)
    assert_true(verify_workload[p, Blake3, Keccak](proof^, w, w.public_inputs[p]()))


def test_wrong_digest_is_rejected() raises:
    """A prover that claims a wrong digest derives its restriction lines from it and fails the restriction check."""
    var ctx = DeviceContext()
    var w = Keccak(message(128))
    var c = w.statement().compile[p]()
    var trace = w.trace[p](c.layout)
    var idx = advice[p](c.layout, trace)
    var wrong = w.public_inputs[p]()
    wrong[len(wrong) - 1] ^= 1
    var data = Keccak.public_data[p](c.layout, wrong)
    var n_blocks = value_bytes(c.layout.publics, p.h1(), p.h2())
    var blocks = List[UInt8](capacity=n_blocks)
    for i in range(n_blocks):
        blocks.append(data[i])
    var families = c.families.copy()
    var prover = Prover[p, Blake3](ctx, c^.take_shape(), families^)
    load_trace[p, Blake3](ctx, prover, trace)
    load_advice[p, Blake3](ctx, prover, idx)
    load_public[p, Blake3](ctx, prover, blocks)
    var proof = prover.prove(ctx, wrong)
    var ok: Bool
    try:
        ok = verify_workload[p, Blake3, Keccak](proof^, w, wrong)
    except e:
        assert_equal(String(e), "restriction fails")
        ok = False
    assert_false(ok)


def test_public_data_rejects_a_message_past_the_grid() raises:
    """A 136-byte message needs 48 chains; on 24 the verifier's public data must refuse it rather than skip a
    block (Codex, 2026-09-07)."""
    var c = keccak_statement().compile[p]()
    var w = Keccak(message(136))
    var inputs = w.public_inputs[p]()
    var rejected = False
    try:
        _ = Keccak.public_data[p](c.layout, inputs)
    except e:
        rejected = True
    assert_true(rejected)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
