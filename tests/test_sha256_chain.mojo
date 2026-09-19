"""SHA-256 hash chain on the builder: the reference chain digest, the trace against every compiled family on
the host (idle chains included, one lane and three lanes), and the prover round trips on the 32 x 224 grid
(3 hashes) and the 96 x 224 grid (3 lanes, 9 hashes) with a wrong digest rejected."""

from std.testing import assert_equal, assert_true, assert_false, TestSuite
from max.gpu.host import DeviceContext

from core.params import CLIENT, Params
from core.hash import Blake3
from workloads.sha256 import Sha256Chain, sha256, sha256_chain, chain_statement, chain_trace, chain_public_values, hashes_of, chain_hashes, CHAIN_PUBLICS
from prover import Prover, load_trace, load_advice, load_public
from relations.statement import advice
from workload import prove_workload, verify_workload, check_families

comptime p = CLIENT.grid(32, 193)          # 3 hashes of 64 chains and the settled state: 224 chains


def message() -> List[UInt8]:
    var m = List[UInt8](capacity=32)
    for i in range(32):
        m.append(UInt8((i * 37 + 11) % 256))
    return m^


def test_chain_digest_is_the_iterated_hash() raises:
    assert_equal(hashes_of(p.h2()), 3)
    var d = sha256(message())
    d = sha256(d)
    d = sha256(d)
    var c = sha256_chain(message(), 3)
    for i in range(32):
        assert_equal(c[i], d[i])


def test_trace_satisfies_every_family() raises:
    """On the 224-chain grid (31 idle chains), on one with 400 (6 hashes, 15 idle chains), and on three lanes."""
    _check_families[p]()
    _check_families[CLIENT.grid(32, 400)]()
    _check_families[CLIENT.grid(96, 193)]()


def _check_families[q: Params]() raises:
    comptime N = q.N()
    var c = chain_statement[q]().compile[q]()
    var msg = message()
    var trace = chain_trace[q](c.layout, msg)
    var pubs = List[List[UInt8]]()
    for l in range(CHAIN_PUBLICS):
        var period = chain_public_values[q](msg, l)
        var full = List[UInt8](capacity=N)
        while len(full) < N:
            full.extend(period.copy())
        pubs.append(full^)
    check_families[q](c, trace, pubs)


def test_prover_round_trip() raises:
    var ctx = DeviceContext()
    var w = Sha256Chain(message())
    var proof = prove_workload[p, Blake3, Sha256Chain](ctx, w)
    assert_true(verify_workload[p, Blake3, Sha256Chain](proof^, w, w.public_inputs[p]()))
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks (decisions.md 2026-09-16)


def test_prover_round_trip_on_three_lanes() raises:
    comptime q = CLIENT.grid(96, 193)
    assert_equal(chain_hashes[q](), 9)
    var ctx = DeviceContext()
    var w = Sha256Chain(message())
    var proof = prove_workload[q, Blake3, Sha256Chain](ctx, w)
    assert_true(verify_workload[q, Blake3, Sha256Chain](proof^, w, w.public_inputs[q]()))
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks (decisions.md 2026-09-16)


def test_prover_round_trip_with_split_tail_levels() raises:
    """224 x 8064: the tail splits into codewords past level 0, so every level's power table is sized
    by the rows of one codeword (tail_materialize), not by the level's whole length."""
    comptime q = CLIENT.grid(224, 8064)
    var ctx = DeviceContext()
    var w = Sha256Chain(message())
    var proof = prove_workload[q, Blake3, Sha256Chain](ctx, w)
    assert_true(verify_workload[q, Blake3, Sha256Chain](proof^, w, w.public_inputs[q]()))
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks (decisions.md 2026-09-16)


def test_wrong_digest_is_rejected() raises:
    var ctx = DeviceContext()
    var w = Sha256Chain(message())
    var c = w.statement[p]().compile[p]()
    var trace = w.trace[p](c.layout)
    var idx = advice[p](c.layout, trace)
    var wrong = w.public_inputs[p]()
    wrong[len(wrong) - 1] ^= 1
    var data = Sha256Chain.public_data[p](c.layout, wrong)
    var prover = Prover[p, Blake3](ctx, c^)
    load_trace[p, Blake3](ctx, prover, trace)
    load_advice[p, Blake3](ctx, prover, idx)
    load_public[p, Blake3](ctx, prover, data)
    var proof = prover.prove(ctx, wrong)
    var ok: Bool
    try:
        ok = verify_workload[p, Blake3, Sha256Chain](proof^, w, wrong)
    except e:
        assert_equal(String(e), "restriction fails")
        ok = False
    assert_false(ok)
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks (decisions.md 2026-09-16)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
