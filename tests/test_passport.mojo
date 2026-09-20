"""One passport in one proof (workloads/passport.mojo): the SOD fixture of test_sod and the DSC fixture of
test_dsc, whose certificate body carries the SOD's DSC key. Two limbs and two modmuls (e = 3) per verify;
the messages, the body, both signatures and the DSC key are witness. Checks the accepted proof, the
verifier's check of the public inputs, and that a body with another key, another CSCA key or a forged
signature is refused."""

from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceContext

from core.params import CLIENT
from core.hash import Blake3
from workloads.bigint import Big
from workloads.sha256 import sha256
from workloads.passport import Passport, passport_chains, HEAD
from workloads.csca import Registry, passport_check_one
from workloads.mrz import WINDOW_OFFSET
from workload import prove_workload, prove_prepared, verify_workload
from prover import Prover

comptime p = CLIENT.grid(144, 192)
# the SOD (test_sod): s1^3 = m1 mod n_dsc, m1's low limb the digest of the signed attributes
comptime N_DSC_HEX = "c6c2c947fb9983e3f183b3f232619049cf007a2a00ea640b93ca8c2a11b3ed503dc31cbb8c4e06dd5baf9e24dd239c6bc07e3bf53eb92f746873f6e09dd44ccf"
comptime S1_HEX = "a4ffd94cde96743b4c2ea345612da61d62539c51f34fb1e63f1a2d37aa54d57578a3ee2a104f96000f74846c5ffc2f7c7a34dfd37c6e751cf5a6a89739720171"
comptime M1_HEX = "7a5bc125f007aa97c0b78533c9a6b4866f5d74360f19219e4ae344c2964f337449a08d34a4d6ba4d1299c3f31cecf3e76de7c155c283bd3cb65c65ad04fb64c4"
comptime DG1_HEX = "615b5f1f58503c55544f4552494b53534f4e3c3c414e4e413c4d415249413c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c4c38393839303243333655544f3734303831323246313230343135395a45313834323236423c3c3c3c3c3130"
comptime ECONTENT_HEX = "585b93f8e779c6c8432bc07d1c637793f4d77e0b756865f7aec3756f98d6ec6eb767eda371904651274d0750e87265aa"
comptime ATTRS_HEX = "3147301506092a864886f70d010903310806066781080101302f06092a864886f70d0109043122042023c09199ea90aa65aab358e441333b56dc09a1d9ca18425aadbe0f43429e1b31"
comptime EMBED_1 = 8
comptime EMBED_2 = 41
comptime SCOPE_HEX = "5b3e2b1e0a76d2e7e1d5a01d5b0d0b6b7d2a3c9e0f1e2d3c4b5a69788796a5b4"
# the DSC check (test_dsc): s2^3 = m2 mod n_csca, m2's low limb the digest of the body, n_dsc at byte 17 of it
comptime N_CSCA_HEX = "ae9f2da27d4ee55e9425452a5176d7b526e0fef149c1e61963323dea4370ba86b2f03201a478b591bbe3ceaffc6c1ee32e9296af427cc27b8d1c421e9fb884d3"
comptime S2_HEX = "64095ef87e816f7dc92dfa7301e47aa78d4d4562e75f0c8b81f8fb971c9e83aee40a379c42a63f739aaf110f1daec4b8f3efdd66638b86d9a8bef2e783acf524"
comptime M2_HEX = "35c6ec8cae67e797fd49ac313af557bd255ce59f48c684f7259c0a9fafa884c1ada08f1654d6e081b86d9e88e613d709cc9d305f7b3ac966796edafbd5bf9729"
comptime TBS_HEX = "d589943e9e5a7d20720bef000f8d68ff37c6c2c947fb9983e3f183b3f232619049cf007a2a00ea640b93ca8c2a11b3ed503dc31cbb8c4e06dd5baf9e24dd239c6bc07e3bf53eb92f746873f6e09dd44ccfaa3545c004eb6f6299603762c7eeef70d6caf4"
comptime N_OFFSET = 17


def _message(hex: String) raises -> List[UInt8]:
    """Big-endian bytes in message order."""
    var v = Big.from_hex(hex).bytes(hex.byte_length() // 2)
    v.reverse()
    return v^


def _verdict(proof: List[UInt8], w: Passport, inputs: List[UInt8]) raises -> String:
    try:
        if verify_workload[p, Blake3, Passport](proof.copy(), w, inputs):
            return "accepted"
        return "rejected"
    except e:
        return String(e)


def _prove_as(ctx: DeviceContext, w: Passport, inputs: List[UInt8]) raises -> List[UInt8]:
    """Prove `w` under the public inputs `inputs` (a prover whose witness does not match what it claims)."""
    var c = w.statement[p]().compile[p]()
    var layout = c.layout.copy()
    var prover = Prover[p, Blake3](ctx, c^)
    return prove_prepared[p, Blake3, Passport](ctx, prover, w, layout, inputs)


def _check(inputs: List[UInt8], registry: Registry) raises -> String:
    try:
        passport_check_one(inputs, registry)
        return "trusted"
    except e:
        return String(e)


def test_passport_verifies_in_one_proof() raises:
    var ctx = DeviceContext()
    var n_dsc = Big.from_hex(N_DSC_HEX)
    var n_csca = Big.from_hex(N_CSCA_HEX)
    var m1 = Big.from_hex(M1_HEX)
    var m2 = Big.from_hex(M2_HEX)
    var msgs: List[List[UInt8]] = [_message(DG1_HEX), _message(ECONTENT_HEX), _message(ATTRS_HEX)]
    var tbs = _message(TBS_HEX)
    assert_equal(m1.low(256), Big.from_bytes(sha256(msgs[2])[::-1]))
    assert_equal(m2.low(256), Big.from_bytes(sha256(tbs)[::-1]))
    var lengths: List[Int] = [len(msgs[0]), len(msgs[1]), len(msgs[2])]
    var embeds: List[Int] = [EMBED_1, EMBED_2]
    var scope = _message(SCOPE_HEX)
    assert_equal(passport_chains(lengths, len(tbs), 2, 2), 183)
    var w = Passport(2, 2, Big.from_hex(S1_HEX), n_dsc, m1, Big.from_hex(S2_HEX), n_csca, m2, lengths.copy(), embeds.copy(),
                     WINDOW_OFFSET, len(tbs), N_OFFSET, scope.copy(), msgs.copy(), tbs.copy())
    var inputs = w.public_inputs[p]()
    var proof = prove_workload[p, Blake3, Passport](ctx, w)
    # the verifier: no messages, body, signatures or DSC key; the padding limbs of m1 and m2, n_csca public
    var verifier = Passport(2, 2, Big(), Big(), m1.shr(256).shl(256), Big(), n_csca, m2.shr(256).shl(256), lengths.copy(), embeds.copy(),
                            WINDOW_OFFSET, len(tbs), N_OFFSET, scope.copy())
    assert_equal(_verdict(proof, verifier, inputs), "accepted")
    assert_equal(sha256(sha256(msgs[1]) + scope.copy()), List[UInt8](inputs[len(inputs) - 32:]))
    # the verifier's side: the CSCA key on the list (the padding limbs are not canonical in this fixture)
    var registry = Registry()
    registry.add(n_csca, 3)
    assert_equal(_check(inputs, registry), "the padding limbs are not PKCS#1 v1.5 with a SHA-256 DigestInfo")
    var empty = Registry()
    assert_equal(_check(inputs, empty), "the CSCA key is not in the registry")
    # the body carries another DSC key: the SOD verify's n is not the body's windows
    var bad = tbs.copy()
    bad[N_OFFSET + 3] ^= 1
    var wb = Passport(2, 2, Big.from_hex(S1_HEX), n_dsc, m1, Big.from_hex(S2_HEX), n_csca, m2, lengths.copy(), embeds.copy(),
                      WINDOW_OFFSET, len(tbs), N_OFFSET, scope.copy(), msgs.copy(), bad^)
    assert_equal(_verdict(_prove_as(ctx, wb, inputs), verifier, inputs), "wiring grand product is not the public factor")
    # another CSCA key claimed: the second verify's n is a public factor
    var other = inputs.copy()
    other[HEAD + 32 + 5] ^= 1
    assert_equal(_verdict(_prove_as(ctx, w, other), verifier, other), "wiring grand product is not the public factor")
    # a forged CSCA signature
    var ws = Passport(2, 2, Big.from_hex(S1_HEX), n_dsc, m1, Big.from_hex(S2_HEX) + Big(1), n_csca, m2, lengths.copy(), embeds.copy(),
                      WINDOW_OFFSET, len(tbs), N_OFFSET, scope.copy(), msgs.copy(), tbs.copy())
    var forged: String
    try:
        _ = _prove_as(ctx, ws, inputs)
        forged = "proved"
    except e:
        forged = String(e)
    assert_equal(forged, "the signature does not verify")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
