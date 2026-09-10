"""SHA-256 on the builder: the reference digest, the trace against every compiled family on the host (idle
chains included), and the prover round trip on the 32 x 224 grid with a wrong digest rejected."""

from std.testing import assert_equal, assert_true, assert_false, TestSuite
from max.gpu.host import DeviceContext

from caracal7.core.params import CLIENT, Params
from caracal7.core.hash import Blake3
from caracal7.workloads.sha256 import Sha256, sha256, sha256_statement, sha256_trace, sha256_public_values, chain_words, PUBLICS, SHA_COLUMNS, WORDS, ROWS
from caracal7.prover import Prover, load_trace, load_advice, load_public
from caracal7.relations import value_bytes
from caracal7.relations.statement import advice
from caracal7.workload import prove_workload, verify_workload, check_families

comptime p = CLIENT.grid(32, 193)          # 3 blocks of 64 chains and the settled state: 224 chains


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
    assert_equal(hex(sha256(List[UInt8]())), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    assert_equal(hex(sha256([UInt8(97), UInt8(98), UInt8(99)])), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    var m = List[UInt8](capacity=56)
    for ch in "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".as_bytes():
        m.append(ch)
    assert_equal(hex(sha256(m)), "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1")


def test_trace_satisfies_every_family() raises:
    """Every entry summed per family is zero on every row its gate admits, on the 224-chain grid (31 idle
    chains) and on one with 800 (a whole idle block and more)."""
    _check_families[p](128)
    _check_families[CLIENT.grid(32, 800)](128)


def test_idle_chains_keep_the_state() raises:
    var w = chain_words(message(128), 224)
    for t in range(193, 224):
        for i in range(WORDS):
            assert_equal(w[t * SHA_COLUMNS + i], w[192 * SHA_COLUMNS + WORDS + i])


def _check_families[q: Params](bytes: Int) raises:
    comptime N = q.N()
    var c = sha256_statement[q]().compile[q]()
    assert_equal(c.shape.points, 20)
    var msg = message(bytes)
    var trace = sha256_trace[q](c.layout, msg)
    var pubs = List[List[UInt8]]()
    for l in range(PUBLICS):
        var period = sha256_public_values[q](msg, l)
        var full = List[UInt8](capacity=N)
        while len(full) < N:
            full.extend(period.copy())
        pubs.append(full^)
    check_families[q](c, trace, pubs)


def test_prover_round_trip() raises:
    var ctx = DeviceContext()
    var w = Sha256(message(128))
    var proof = prove_workload[p, Blake3, Sha256](ctx, w)
    assert_true(verify_workload[p, Blake3, Sha256](proof^, w, w.public_inputs[p]()))


def test_wrong_digest_is_rejected() raises:
    var ctx = DeviceContext()
    var w = Sha256(message(128))
    var c = w.statement[p]().compile[p]()
    var trace = w.trace[p](c.layout)
    var idx = advice[p](c.layout, trace)
    var wrong = w.public_inputs[p]()
    wrong[len(wrong) - 1] ^= 1
    var data = Sha256.public_data[p](c.layout, wrong)
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
        ok = verify_workload[p, Blake3, Sha256](proof^, w, wrong)
    except e:
        assert_equal(String(e), "restriction fails")
        ok = False
    assert_false(ok)


def test_public_data_rejects_a_message_past_the_grid() raises:
    """A 192-byte message needs 4 blocks and 257 chains; on 224 the verifier's public data must refuse it."""
    var c = sha256_statement[p]().compile[p]()
    var w = Sha256(message(192))
    var inputs = w.public_inputs[p]()
    var rejected = False
    try:
        _ = Sha256.public_data[p](c.layout, inputs)
    except e:
        rejected = True
    assert_true(rejected)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
