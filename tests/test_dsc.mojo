"""The DSC workload (workloads/dsc.mojo): the certificate body (100 bytes, the 64-byte DSC key at byte 17), the
commitment group (n_dsc || r) and a 512-bit RSA verify (2 limbs, e = 3) of the CSCA's signature on a 96-chain
grid, 88 chains live. The proof verifies against public inputs without the body, s, n_dsc or r; a body whose
key is not the committed one, random bytes the public digest does not hash, and a body whose digest is not
the signed limb are refused at the wire. The commitment is the one test_sod's SOD opens. Fixture: OpenSSL
genrsa -3 512 twice (CSCA, DSC), s = m^d mod n_csca for m = (random upper limb || sha256(body))."""

from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceContext

from core.params import CLIENT
from core.hash import Blake3
from workloads.bigint import Big
from workloads.sha256 import sha256
from workloads.dsc import DSC
from workloads.sod import SOD, sha_chains, commitment_length, commitment_message
from workloads.rsa import chain_count
from workload import prove_workload, prove_prepared, verify_workload
from prover import Prover

comptime p = CLIENT.grid(144, 96)
comptime N_CSCA_HEX = "ae9f2da27d4ee55e9425452a5176d7b526e0fef149c1e61963323dea4370ba86b2f03201a478b591bbe3ceaffc6c1ee32e9296af427cc27b8d1c421e9fb884d3"
comptime N_DSC_HEX = "c6c2c947fb9983e3f183b3f232619049cf007a2a00ea640b93ca8c2a11b3ed503dc31cbb8c4e06dd5baf9e24dd239c6bc07e3bf53eb92f746873f6e09dd44ccf"
comptime S_DSC_HEX = "64095ef87e816f7dc92dfa7301e47aa78d4d4562e75f0c8b81f8fb971c9e83aee40a379c42a63f739aaf110f1daec4b8f3efdd66638b86d9a8bef2e783acf524"
comptime M_DSC_HEX = "35c6ec8cae67e797fd49ac313af557bd255ce59f48c684f7259c0a9fafa884c1ada08f1654d6e081b86d9e88e613d709cc9d305f7b3ac966796edafbd5bf9729"
comptime TBS_HEX = "d589943e9e5a7d20720bef000f8d68ff37c6c2c947fb9983e3f183b3f232619049cf007a2a00ea640b93ca8c2a11b3ed503dc31cbb8c4e06dd5baf9e24dd239c6bc07e3bf53eb92f746873f6e09dd44ccfaa3545c004eb6f6299603762c7eeef70d6caf4"
comptime N_OFFSET = 17
comptime R_HEX = "a584972e6adad837d3db75d7685592f3de517be3afce80bf3883008fa9c10714"


def _message(hex: String) raises -> List[UInt8]:
    """Big-endian bytes in message order."""
    var v = Big.from_hex(hex).bytes(hex.byte_length() // 2)
    v.reverse()
    return v^


def _verdict(proof: List[UInt8], w: DSC, inputs: List[UInt8]) raises -> String:
    try:
        if verify_workload[p, Blake3, DSC](proof.copy(), w, inputs):
            return "accepted"
        return "rejected"
    except e:
        return String(e)


def _prove_as(ctx: DeviceContext, w: DSC, inputs: List[UInt8]) raises -> List[UInt8]:
    """Prove `w` under the public inputs `inputs` (a prover whose witness does not match what it claims)."""
    var c = w.statement[p]().compile[p]()
    var layout = c.layout.copy()
    var prover = Prover[p, Blake3](ctx, c^)
    return prove_prepared[p, Blake3, DSC](ctx, prover, w, layout, inputs)


def test_dsc_verifies_and_wires_hold() raises:
    var ctx = DeviceContext()
    var s = Big.from_hex(S_DSC_HEX)
    var n_csca = Big.from_hex(N_CSCA_HEX)
    var n_dsc = Big.from_hex(N_DSC_HEX)
    var m = Big.from_hex(M_DSC_HEX)
    var tbs = _message(TBS_HEX)
    var r = _message(R_HEX)
    assert_equal(m.low(256), Big.from_bytes(sha256(tbs)[::-1]))
    var nb = n_dsc.bytes(64)
    nb.reverse()
    for i in range(64):
        assert_equal(tbs[N_OFFSET + i], nb[i])
    assert_equal(sha_chains(len(tbs)) + sha_chains(commitment_length(2)) + chain_count(2, 2), 88)
    var w = DSC(2, 2, s, n_csca, m, len(tbs), N_OFFSET, tbs.copy(), n_dsc, r.copy())
    var inputs = w.public_inputs[p]()
    var proof = prove_workload[p, Blake3, DSC](ctx, w)
    var verifier = DSC(2, 2, Big(), n_csca, m.shr(256).shl(256), len(tbs), N_OFFSET)      # no body, s, n_dsc, r or low limb
    assert_equal(_verdict(proof, verifier, inputs), "accepted")
    # the commitment the SOD proof opens is the same bytes
    var sod = SOD(2, 2, Big(), n_dsc, Big(), [1, 2, 3], [0, 0], List[List[UInt8]](), r.copy())
    var c = sha256(commitment_message(n_dsc, 2, r))
    for i in range(32):
        assert_equal(inputs[len(inputs) - 32 + i], c[i])
    _ = sod
    # the body carries another key than the committed one: the window wires break
    var bad = tbs.copy()
    bad[N_OFFSET + 40] ^= 1
    var wb = DSC(2, 2, s, n_csca, m, len(tbs), N_OFFSET, bad^, n_dsc, r.copy())
    assert_equal(_verdict(prove_workload[p, Blake3, DSC](ctx, wb), verifier, inputs), "wiring grand product is not the public factor")
    # other random bytes: the commitment digest is not the public one
    var r1 = r.copy()
    r1[7] ^= 1
    var wr = DSC(2, 2, s, n_csca, m, len(tbs), N_OFFSET, tbs.copy(), n_dsc, r1^)
    assert_equal(_verdict(_prove_as(ctx, wr, inputs), verifier, inputs), "wiring grand product is not the public factor")
    # another committed key with the body and the signature honest: only the window wires break
    var wk = DSC(2, 2, s, n_csca, m, len(tbs), N_OFFSET, tbs.copy(), n_dsc + Big(1), r.copy())
    assert_equal(_verdict(prove_workload[p, Blake3, DSC](ctx, wk), verifier, wk.public_inputs[p]()), "wiring grand product is not the public factor")
    # the body changed outside the key: its digest is not the signed limb
    var bad2 = tbs.copy()
    bad2[3] ^= 1
    var wb2 = DSC(2, 2, s, n_csca, m, len(tbs), N_OFFSET, bad2^, n_dsc, r.copy())
    assert_equal(_verdict(prove_workload[p, Blake3, DSC](ctx, wb2), verifier, inputs), "wiring grand product is not the public factor")
    # another signature: the prover refuses
    var ws = DSC(2, 2, s + Big(1), n_csca, m, len(tbs), N_OFFSET, tbs.copy(), n_dsc, r.copy())
    var refused = String("")
    try:
        _ = prove_workload[p, Blake3, DSC](ctx, ws)
    except e:
        refused = String(e)
    assert_equal(refused, "the signature does not verify")
    # the CSCA key differs: the transcript binds the public inputs
    var other = inputs.copy()
    other[6] ^= 1
    assert_equal(_verdict(proof, verifier, other), "public inputs differ")
    # another offset: the statement pins it
    var off = inputs.copy()
    off[4] = 18
    assert_equal(_verdict(proof, verifier, off), "public inputs do not start with the statement's pinned bytes")
    # more limbs than the statement's: refused before any public data is indexed
    var limbs = inputs.copy()
    limbs[0] = 3
    limbs.extend(List[UInt8](length=64, fill=0))
    assert_equal(_verdict(proof, verifier, limbs), "the public inputs' limbs are not the statement's")
    # a length the statement was not compiled with is refused before any public data is indexed
    var lng = inputs.copy()
    lng[2] = 1
    assert_equal(_verdict(proof, verifier, lng), "the public inputs' length and offset are not the statement's")
    _ = ctx


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
