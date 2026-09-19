"""The SOD workload (workloads/sod.mojo): three SHA-256 groups (the ICAO sample DG1, an eContent, signed
attributes of the real ASN.1 shape), the commitment group (n || r), the nullifier group and a 512-bit RSA
verify (2 limbs, e = 3) on a 192-chain grid, 171 chains live. The proof verifies against public inputs without
the messages, s or n, and the verifier reads the MRZ fields from the disclosed window; a DG1 whose digest the
eContent does not carry, signed attributes whose digest is not the signed limb, a modulus the commitment does
not hold, random bytes the public digest does not hash, a window that is not DG1's and a scope the nullifier
does not hash are refused at the wire. Fixture: OpenSSL genrsa -3 512 (the DSC key of test_dsc), s = m^d mod n
for m = (random upper limb || sha256(attrs))."""

from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceContext

from core.params import CLIENT
from core.hash import Blake3
from workloads.bigint import Big
from workloads.sha256 import sha256
from workloads.sod import SOD, sha_chains, commitment_length, NULL_LENGTH, HEAD
from workloads.mrz import mrz_fields, Date, WINDOW_OFFSET
from workloads.rsa import chain_count
from workload import prove_workload, prove_prepared, verify_workload
from prover import Prover

comptime p = CLIENT.grid(144, 192)
comptime N_HEX = "c6c2c947fb9983e3f183b3f232619049cf007a2a00ea640b93ca8c2a11b3ed503dc31cbb8c4e06dd5baf9e24dd239c6bc07e3bf53eb92f746873f6e09dd44ccf"
comptime S_HEX = "a4ffd94cde96743b4c2ea345612da61d62539c51f34fb1e63f1a2d37aa54d57578a3ee2a104f96000f74846c5ffc2f7c7a34dfd37c6e751cf5a6a89739720171"
comptime M_HEX = "7a5bc125f007aa97c0b78533c9a6b4866f5d74360f19219e4ae344c2964f337449a08d34a4d6ba4d1299c3f31cecf3e76de7c155c283bd3cb65c65ad04fb64c4"
comptime S_FORGED_HEX = "000000000000000000000000000000000000000000000000000000000000000000000000000000000000064d7821c43dd01460a217e9113a5b688cfaf28468dc"
comptime N_FORGED_HEX = "8000000000000000000000000000000000000000000c6efcc1784c7bdae68a223c0406fad3a6122ce7cf6aeaa19d359531f7f40c96008e267a157f5ba3d294fc"
comptime R_HEX = "a584972e6adad837d3db75d7685592f3de517be3afce80bf3883008fa9c10714"
comptime DG1_HEX = "615b5f1f58503c55544f4552494b53534f4e3c3c414e4e413c4d415249413c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c4c38393839303243333655544f3734303831323246313230343135395a45313834323236423c3c3c3c3c3130"
comptime ECONTENT_HEX = "585b93f8e779c6c8432bc07d1c637793f4d77e0b756865f7aec3756f98d6ec6eb767eda371904651274d0750e87265aa"
comptime ATTRS_HEX = "3147301506092a864886f70d010903310806066781080101302f06092a864886f70d0109043122042023c09199ea90aa65aab358e441333b56dc09a1d9ca18425aadbe0f43429e1b31"
comptime EMBED_1 = 8
comptime EMBED_2 = 41
comptime SCOPE_HEX = "5b3e2b1e0a76d2e7e1d5a01d5b0d0b6b7d2a3c9e0f1e2d3c4b5a69788796a5b4"


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
    var scope = _message(SCOPE_HEX)
    assert_equal(sha_chains(lengths[0]) + sha_chains(lengths[1]) + sha_chains(lengths[2]) + sha_chains(commitment_length(2)) + sha_chains(NULL_LENGTH) + chain_count(2, 2), 166)
    var w = SOD(2, 2, s, n, m, lengths.copy(), embeds.copy(), WINDOW_OFFSET, scope.copy(), msgs.copy(), r.copy())
    var inputs = w.public_inputs[p]()
    var proof = prove_workload[p, Blake3, SOD](ctx, w)
    var verifier = SOD(2, 2, Big(), Big(), m.shr(256).shl(256), lengths.copy(), embeds.copy(), WINDOW_OFFSET, scope.copy())     # no messages, s, n, r or low limb
    assert_equal(_verdict(proof, verifier, inputs), "accepted")
    # the verifier's predicates on the disclosed window: Anna Maria Eriksson of Utopia, born 1974-08-12, expired 2012-04-15
    var window = List[UInt8]()
    for i in range(32):
        window.append(inputs[HEAD + 32 + 32 + i])
    var today = Date(2026, 9, 19)
    var fields = mrz_fields(window, WINDOW_OFFSET, lengths[0], today)
    assert_equal(fields.nationality, "UTO")
    assert_equal(fields.sex, "F")
    assert_equal(fields.birth, Date(1974, 8, 12))
    assert_equal(fields.expiry, Date(2012, 4, 15))
    assert_equal(fields.age_on(today), 52)
    assert_true(fields.expired_on(today))
    assert_equal(sha256(sha256(msgs[1]) + scope.copy()), List[UInt8](inputs[len(inputs) - 32:]))
    # DG1 changed: its digest is not the one the eContent carries
    var bad = msgs.copy()
    bad[0][5] ^= 1
    var wb = SOD(2, 2, s, n, m, lengths.copy(), embeds.copy(), WINDOW_OFFSET, scope.copy(), bad^, r.copy())
    assert_equal(_verdict(prove_workload[p, Blake3, SOD](ctx, wb), verifier, inputs), "wiring grand product is not the public factor")
    # eContent changed: its three wires break (the digest it carries, the digest it has, the nullifier's window)
    var bad1 = msgs.copy()
    bad1[1][0] ^= 1
    var wb1 = SOD(2, 2, s, n, m, lengths.copy(), embeds.copy(), WINDOW_OFFSET, scope.copy(), bad1^, r.copy())
    assert_equal(_verdict(_prove_as(ctx, wb1, inputs), verifier, inputs), "wiring grand product is not the public factor")
    # signed attributes changed: their digest is not the signed limb
    var bad2 = msgs.copy()
    bad2[2][60] ^= 1
    var wb2 = SOD(2, 2, s, n, m, lengths.copy(), embeds.copy(), WINDOW_OFFSET, scope.copy(), bad2^, r.copy())
    assert_equal(_verdict(prove_workload[p, Blake3, SOD](ctx, wb2), verifier, inputs), "wiring grand product is not the public factor")
    # other random bytes: the commitment digest is not the public one
    var r1 = r.copy()
    r1[3] ^= 1
    var wr = SOD(2, 2, s, n, m, lengths.copy(), embeds.copy(), WINDOW_OFFSET, scope.copy(), msgs.copy(), r1^)
    assert_equal(_verdict(_prove_as(ctx, wr, inputs), verifier, inputs), "wiring grand product is not the public factor")
    # another key pair with the same m (s'^3 = m + n'): the RSA lane holds, the committed n is not its modulus
    var wn = SOD(2, 2, Big.from_hex(S_FORGED_HEX), Big.from_hex(N_FORGED_HEX), m, lengths.copy(), embeds.copy(), WINDOW_OFFSET, scope.copy(), msgs.copy(), r.copy(), committed=n)
    assert_equal(_verdict(prove_workload[p, Blake3, SOD](ctx, wn), verifier, inputs), "wiring grand product is not the public factor")
    # the disclosed window is not DG1's: the nationality byte in the public inputs differs from the trace's
    var claim = inputs.copy()
    claim[HEAD + 32 + 32] ^= 1
    assert_equal(_verdict(_prove_as(ctx, w, claim), verifier, claim), "wiring grand product is not the public factor")
    var claim1 = inputs.copy()                        # the window's last byte: the wd segment (s0 = 88, the last 11 bytes)
    claim1[HEAD + 32 + 32 + 31] ^= 1
    assert_equal(_verdict(_prove_as(ctx, w, claim1), verifier, claim1), "wiring grand product is not the public factor")
    # another scope in the trace than in the public inputs: the scope window and the nullifier both break
    var scope1 = scope.copy()
    scope1[0] ^= 1
    var ws = SOD(2, 2, s, n, m, lengths.copy(), embeds.copy(), WINDOW_OFFSET, scope1^, msgs.copy(), r.copy())
    assert_equal(_verdict(_prove_as(ctx, ws, inputs), verifier, inputs), "wiring grand product is not the public factor")
    var scope2 = scope.copy()                         # the scope's last byte: the u1 segment
    scope2[31] ^= 1
    var ws2 = SOD(2, 2, s, n, m, lengths.copy(), embeds.copy(), WINDOW_OFFSET, scope2^, msgs.copy(), r.copy())
    assert_equal(_verdict(_prove_as(ctx, ws2, inputs), verifier, inputs), "wiring grand product is not the public factor")
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
