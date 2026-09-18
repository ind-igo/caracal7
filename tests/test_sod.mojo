"""The SOD workload (workloads/sod.mojo): three SHA-256 groups (DG1, eContent, signed attributes of the
real ASN.1 shape) and a 512-bit RSA verify (2 limbs, e = 3) on a 96-chain grid, 89 chains live. The proof
verifies against public inputs without the messages; a DG1 whose digest the eContent does not carry, and
signed attributes whose digest is not the signed limb, are refused at the wire. Fixture: OpenSSL genrsa -3
512, s = m^d mod n for m = (random upper limb || sha256(attrs))."""

from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceContext

from core.params import CLIENT
from core.hash import Blake3
from workloads.bigint import Big
from workloads.sha256 import sha256
from workloads.sod import SOD, sha_chains
from workloads.rsa import chain_count
from workload import prove_workload, verify_workload

comptime p = CLIENT.grid(144, 96)
comptime N_HEX = "d7aec95e774314c3f834d5be00b9c3992fd390a8f28efb81eb5b92756aaf3b3760b3b14c60f271756621756bfc55c71ee878fd115d348cf8066a045a4d09aeb7"
comptime S_HEX = "100cf98a149960acd468db77214c092f274fa125f8438e3757272ff5316bc8c6106ce64779297d2860628e53a10398b4ba1c6c0aaa6f41515d8ea6e5bc4c7620"
comptime M_HEX = "8ede0d7ac3baea9e13deef86ab1031d0f646e1f40a097c976bf46c697d2caf83bf8080d253cc543c9c0647462570f1a2c992cbf3679a80eb99271400f1bbadd4"
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


def test_sod_verifies_and_wires_hold() raises:
    var ctx = DeviceContext()
    var s = Big.from_hex(S_HEX)
    var n = Big.from_hex(N_HEX)
    var m = Big.from_hex(M_HEX)
    var msgs = _messages()
    assert_equal(m.low(256), Big.from_bytes(sha256(msgs[2])[::-1]))
    var lengths: List[Int] = [len(msgs[0]), len(msgs[1]), len(msgs[2])]
    var embeds: List[Int] = [EMBED_1, EMBED_2]
    assert_equal(sha_chains(lengths[0]) + sha_chains(lengths[1]) + sha_chains(lengths[2]) + chain_count(2, 2), 89)
    var w = SOD(2, 2, s, n, m, lengths.copy(), embeds.copy(), msgs.copy())
    var inputs = w.public_inputs[p]()
    var proof = prove_workload[p, Blake3, SOD](ctx, w)
    var verifier = SOD(2, 2, s, n, m.shr(256).shl(256), lengths.copy(), embeds.copy())     # no messages, no low limb
    assert_equal(_verdict(proof, verifier, inputs), "accepted")
    # DG1 changed: its digest is not the one the eContent carries
    var bad = msgs.copy()
    bad[0][5] ^= 1
    var wb = SOD(2, 2, s, n, m, lengths.copy(), embeds.copy(), bad^)
    assert_equal(_verdict(prove_workload[p, Blake3, SOD](ctx, wb), verifier, inputs), "wiring grand product is not the public factor")
    # eContent changed: both of its wires break (the digest it carries and the digest it has)
    var bad1 = msgs.copy()
    bad1[1][0] ^= 1
    var wb1 = SOD(2, 2, s, n, m, lengths.copy(), embeds.copy(), bad1^)
    assert_equal(_verdict(prove_workload[p, Blake3, SOD](ctx, wb1), verifier, inputs), "wiring grand product is not the public factor")
    # signed attributes changed: their digest is not the signed limb
    var bad2 = msgs.copy()
    bad2[2][60] ^= 1
    var wb2 = SOD(2, 2, s, n, m, lengths.copy(), embeds.copy(), bad2^)
    assert_equal(_verdict(prove_workload[p, Blake3, SOD](ctx, wb2), verifier, inputs), "wiring grand product is not the public factor")
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
