"""The SOD workload (workloads/sod.mojo): three SHA-256 groups (DG1, eContent, signed attributes of the
real ASN.1 shape), the commitment group (n || r) and a 512-bit RSA verify (2 limbs, e = 3) on a 128-chain
grid, 122 chains live. The proof verifies against public inputs without the messages, s or n; a DG1 whose
digest the eContent does not carry, signed attributes whose digest is not the signed limb, a modulus the
commitment does not hold, and random bytes the public digest does not hash are refused at the wire. Fixture:
OpenSSL genrsa -3 512 (the DSC key of test_dsc), s = m^d mod n for m = (random upper limb || sha256(attrs))."""

from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceContext

from core.params import CLIENT
from core.hash import Blake3
from workloads.bigint import Big
from workloads.sha256 import sha256
from workloads.sod import SOD, sha_chains, commitment_length
from workloads.rsa import chain_count
from workload import prove_workload, prove_prepared, verify_workload
from prover import Prover

comptime p = CLIENT.grid(144, 128)
comptime N_HEX = "c6c2c947fb9983e3f183b3f232619049cf007a2a00ea640b93ca8c2a11b3ed503dc31cbb8c4e06dd5baf9e24dd239c6bc07e3bf53eb92f746873f6e09dd44ccf"
comptime S_HEX = "8006fef715c7c0ddecd6eff468fa7d2a3a68c3cb3f4c6135db4d344b49601f3c6f5c893133ffe5ae144c01ece68e73735ad65dff3d6b87443230e8478b910bcd"
comptime M_HEX = "05eba8ddf8be24f7fcd1908e2bd623d6530ae64739280d5e23971213e29041c8bf8080d253cc543c9c0647462570f1a2c992cbf3679a80eb99271400f1bbadd4"
comptime S_FORGED_HEX = "000000000000000000000000000000000000000000000000000000000000000000000000000000000000051dc07c20aa1270f38fb5cbb5cf52d813c6552d288d"
comptime N_FORGED_HEX = "800000000000000000000000000000000000000000675b234f7754adf696de93768cae91bd0ba698c3bd3e6f903dabdda1e25d70226b410c8650ec9541b55041"
comptime R_HEX = "a584972e6adad837d3db75d7685592f3de517be3afce80bf3883008fa9c10714"
comptime DG1_HEX = "a54dca182530bb1d6d132cded6237b2ed91e3f721fcb1971174494d6493c9d5c3460be31201e69fe"
comptime ECONTENT_HEX = "daa0eee8b9997f5c920764db0e493c5a26904c9f1af12aa4469dd3472cf7cc469452934f43401ddf7c2999fdafe59325"
comptime ATTRS_HEX = "3147301506092a864886f70d010903310806066781080101302f06092a864886f70d0109043122042017588c29fd9dc7826cf2031d1a2d666e5ffe7264ca7c4da46b03849d348b7ef1"
comptime EMBED_1 = 8
comptime EMBED_2 = 41


def _bytes(hex: String) raises -> List[UInt8]:
    return Big.from_hex(hex).bytes(hex.byte_length() // 2)


def _message(hex: String) raises -> List[UInt8]:
    """Big-endian bytes in message order."""
    var v = _bytes(hex)
    v.reverse()
    return v^


def _messages() raises -> List[List[UInt8]]:
    return [_message(DG1_HEX), _message(ECONTENT_HEX), _message(ATTRS_HEX)]


def _verdict(proof: List[UInt8], w: SOD, inputs: List[UInt8]) raises -> String:
    try:
        if verify_workload[p, Blake3, SOD](proof.copy(), w, inputs):
            return "accepted"
        return "rejected"
    except e:
        return String(e)


def _prove_as(ctx: DeviceContext, w: SOD, inputs: List[UInt8]) raises -> List[UInt8]:
    """Prove `w` under the public inputs `inputs` (a prover whose witness does not match what it claims)."""
    var c = w.statement[p]().compile[p]()
    var layout = c.layout.copy()
    var prover = Prover[p, Blake3](ctx, c^)
    return prove_prepared[p, Blake3, SOD](ctx, prover, w, layout, inputs)


def test_sod_verifies_and_wires_hold() raises:
    var ctx = DeviceContext()
    var s = Big.from_hex(S_HEX)
    var n = Big.from_hex(N_HEX)
    var m = Big.from_hex(M_HEX)
    var msgs = _messages()
    assert_equal(m.low(256), Big.from_bytes(sha256(msgs[2])[::-1]))
    var lengths: List[Int] = [len(msgs[0]), len(msgs[1]), len(msgs[2])]
    var embeds: List[Int] = [EMBED_1, EMBED_2]
    var r = _message(R_HEX)
    assert_equal(sha_chains(lengths[0]) + sha_chains(lengths[1]) + sha_chains(lengths[2]) + sha_chains(commitment_length(2)) + chain_count(2, 2), 122)
    var w = SOD(2, 2, s, n, m, lengths.copy(), embeds.copy(), msgs.copy(), r.copy())
    var inputs = w.public_inputs[p]()
    var proof = prove_workload[p, Blake3, SOD](ctx, w)
    var verifier = SOD(2, 2, Big(), Big(), m.shr(256).shl(256), lengths.copy(), embeds.copy())     # no messages, s, n, r or low limb
    assert_equal(_verdict(proof, verifier, inputs), "accepted")
    # DG1 changed: its digest is not the one the eContent carries
    var bad = msgs.copy()
    bad[0][5] ^= 1
    var wb = SOD(2, 2, s, n, m, lengths.copy(), embeds.copy(), bad^, r.copy())
    assert_equal(_verdict(prove_workload[p, Blake3, SOD](ctx, wb), verifier, inputs), "wiring grand product is not the public factor")
    # eContent changed: both of its wires break (the digest it carries and the digest it has)
    var bad1 = msgs.copy()
    bad1[1][0] ^= 1
    var wb1 = SOD(2, 2, s, n, m, lengths.copy(), embeds.copy(), bad1^, r.copy())
    assert_equal(_verdict(prove_workload[p, Blake3, SOD](ctx, wb1), verifier, inputs), "wiring grand product is not the public factor")
    # signed attributes changed: their digest is not the signed limb
    var bad2 = msgs.copy()
    bad2[2][60] ^= 1
    var wb2 = SOD(2, 2, s, n, m, lengths.copy(), embeds.copy(), bad2^, r.copy())
    assert_equal(_verdict(prove_workload[p, Blake3, SOD](ctx, wb2), verifier, inputs), "wiring grand product is not the public factor")
    # other random bytes: the commitment digest is not the public one
    var r1 = r.copy()
    r1[3] ^= 1
    var wr = SOD(2, 2, s, n, m, lengths.copy(), embeds.copy(), msgs.copy(), r1^)
    assert_equal(_verdict(_prove_as(ctx, wr, inputs), verifier, inputs), "wiring grand product is not the public factor")
    # another key pair with the same m (s'^3 = m + n'): the RSA lane holds, the committed n is not its modulus
    var wn = SOD(2, 2, Big.from_hex(S_FORGED_HEX), Big.from_hex(N_FORGED_HEX), m, lengths.copy(), embeds.copy(), msgs.copy(), r.copy(), committed=n)
    assert_equal(_verdict(prove_workload[p, Blake3, SOD](ctx, wn), verifier, inputs), "wiring grand product is not the public factor")
    # more limbs than the statement's: refused before any public data is indexed
    var limbs = inputs.copy()
    limbs[0] = 3
    limbs.extend(List[UInt8](length=32, fill=0))
    assert_equal(_verdict(proof, verifier, limbs), "the public inputs' limbs are not the statement's")
    # the padding limb differs: the transcript binds the public inputs
    var other = inputs.copy()
    other[len(other) - 1] ^= 1
    assert_equal(_verdict(proof, verifier, other), "public inputs differ")
    # a length the statement was not compiled with is refused before any public data is indexed
    var long = inputs.copy()
    long[3] = 1
    assert_equal(_verdict(proof, verifier, long), "the public inputs' lengths and offsets are not the statement's")
    _ = ctx


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
