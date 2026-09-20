"""The CSCA registry: the verifier's list of trusted CSCA keys. A DSC proof names its CSCA key in the public
inputs (`dsc.mojo`: n_csca after the head, little endian); the verifier accepts the proof only when the key's
id is in the registry with the same public exponent. The id of a key is the SHA-256 of its modulus in
big-endian bytes without leading zeros, the form `openssl rsa -modulus` prints (without the `Modulus=`
prefix, hex decoded): the proof's limb count pads the modulus, the id does not depend on it.

The registry is a text file: one key per line as 64 hex digits, then the public exponent in decimal
(65537 when absent); `#` starts a comment, blank lines are skipped. It comes from the ICAO PKD master list
or from a national list; parsing the CMS master list is not in scope, the operator extracts the moduli
with OpenSSL. The exponent is part of the key: the proof's `muls` selects e = 2^(muls - 1) + 1, and a
registry that ignored it would trust a signature under another exponent of a listed modulus.

`passport_check` is the verifier's whole check of the two proofs' public inputs: the CSCA key on the list,
the PKCS#1 v1.5 padding limbs of both m canonical (the proof shows s^e = m for the public upper limbs of m
and the low limb wired to the digest; a prover chooses the upper limbs, so the verifier pins them to the
encoding: 00 01 FF..FF 00 DigestInfo(SHA-256)), and the one commitment digest in both. `passport_check_one`
is the same for the passport in one proof (`passport.mojo`), where the DSC key is a wire and there is no
commitment. Both run beside `verify`, never instead of it: the proof binds the public inputs, this checks
their values.

This check is on the verifier's side: a wrong CSCA key does not break the proof, it breaks the trust in it,
so it costs no chains. A Merkle path inside the proof would hide which CSCA signed, but the disclosed MRZ
window names the country anyway; that path waits for the masking layer."""

from workloads.bigint import Big
from workloads.dsc import HEAD
from workloads.sod import HEAD as SOD_HEAD
from workloads.passport import HEAD as ONE_HEAD
from workloads.rsa import LIMB
from workloads.sha256 import sha256


def _id(var little: List[UInt8]) raises -> List[UInt8]:
    """SHA-256 of the big-endian bytes without leading zeros."""
    while len(little) > 0 and little[len(little) - 1] == 0:
        _ = little.pop()
    little.reverse()
    return sha256(little)


def csca_key_id(n: Big) raises -> List[UInt8]:
    """The id of the modulus n."""
    return _id(n.bytes((n.bit_length() + 7) // 8))


def exponent_muls(e: Int) raises -> Int:
    """The `muls` of the RSA statement for the exponent e = 2^k + 1."""
    var k = 0
    while (1 << k) + 1 < e:
        k += 1
    if (1 << k) + 1 != e or k == 0:
        raise Error("the exponent is 2^k + 1 for k > 0: " + String(e))
    return k + 1


def csca_key_id_of(public_inputs: List[UInt8]) raises -> List[UInt8]:
    """The id of the CSCA key a DSC proof's public inputs name."""
    if len(public_inputs) < HEAD:
        raise Error("public inputs start with limbs, muls, the TBS length and the offset")
    var lb = LIMB // 8 * Int(public_inputs[0])
    if len(public_inputs) < HEAD + lb:
        raise Error("public inputs hold n_csca after the head")
    return _id(List[UInt8](public_inputs[HEAD:HEAD + lb]))


struct Registry(Copyable, Movable):
    """The trusted CSCA keys: (id, muls) pairs."""
    var ids: List[List[UInt8]]
    var muls: List[Int]

    def __init__(out self):
        self.ids = []
        self.muls = []

    @staticmethod
    def parse(text: String) raises -> Registry:
        """One key per line: 64 hex digits, then the exponent in decimal (65537 when absent); `#` comments."""
        var r = Registry()
        for raw in text.splitlines():
            var line = String(raw.split("#")[0]).strip()
            if line.byte_length() == 0:
                continue
            var words = line.split()
            if len(words) > 2 or words[0].byte_length() != 64:
                raise Error("a CSCA key is 64 hex digits and an optional exponent: " + String(line))
            r.ids.append(Big.from_hex(String(words[0])).bytes(32)[::-1])
            r.muls.append(exponent_muls(Int(String(words[1])) if len(words) == 2 else 65537))
        return r^

    def add(mut self, n: Big, e: Int = 65537) raises:
        self.ids.append(csca_key_id(n))
        self.muls.append(exponent_muls(e))

    def trusts(self, public_inputs: List[UInt8]) raises -> Bool:
        """The DSC proof's CSCA key and exponent are in the registry. Runs before `verify` or after it, never
        instead: the proof binds the public inputs, this checks their values."""
        return self.trusts_key(csca_key_id_of(public_inputs), Int(public_inputs[1]))

    def trusts_key(self, id: List[UInt8], muls: Int) -> Bool:
        for i in range(len(self.ids)):
            if self.ids[i] == id and self.muls[i] == muls:
                return True
        return False


comptime DIGEST_INFO = "3031300d060960864801650304020105000420"    # DER prefix of a SHA-256 DigestInfo


def pkcs1_upper(limbs: Int) raises -> List[UInt8]:
    """The upper limbs of a PKCS#1 v1.5 SHA-256 encoding in `limbs` limbs, little endian as the public
    inputs carry them: everything above the 32-byte digest."""
    var lb = LIMB // 8 * limbs
    var info = Big.from_hex(DIGEST_INFO).bytes(DIGEST_INFO.byte_length() // 2)      # little endian
    var v = info.copy()
    v.append(0)
    for _ in range(lb - 3 - len(info) - 32):
        v.append(0xFF)
    v.append(1)
    v.append(0)
    return v^


def passport_check(sod_inputs: List[UInt8], dsc_inputs: List[UInt8], registry: Registry) raises:
    """Refuses a passport whose public inputs are not the verifier's: the DSC proof's CSCA key is not on the
    list, a padding limb is not canonical, or the two proofs do not open one commitment."""
    if not registry.trusts(dsc_inputs):
        raise Error("the CSCA key is not in the registry")
    var lb = LIMB // 8 * Int(dsc_inputs[0])
    if Int(sod_inputs[0]) != Int(dsc_inputs[0]):
        raise Error("the two proofs have different limb counts")
    var upper = pkcs1_upper(Int(dsc_inputs[0]))
    if len(sod_inputs) < SOD_HEAD + lb or len(dsc_inputs) < HEAD + 2 * lb:
        raise Error("public inputs are cut short")
    for i in range(lb - 32):
        if sod_inputs[SOD_HEAD + i] != upper[i] or dsc_inputs[HEAD + lb + i] != upper[i]:
            raise Error("the padding limbs are not PKCS#1 v1.5 with a SHA-256 DigestInfo")
    for i in range(32):
        if sod_inputs[SOD_HEAD + lb - 32 + i] != dsc_inputs[HEAD + 2 * lb - 32 + i]:
            raise Error("the two proofs do not open one commitment")


def passport_check_one(inputs: List[UInt8], registry: Registry) raises:
    """`passport_check` for a passport in one proof (`passport.mojo`): the CSCA key on the list and both m's
    padding limbs canonical; the DSC key is a wire inside the proof, so there is no commitment to compare."""
    if len(inputs) < ONE_HEAD:
        raise Error("public inputs are cut short")
    var lb = LIMB // 8 * Int(inputs[0])
    if len(inputs) < ONE_HEAD + 3 * lb - 64:
        raise Error("public inputs are cut short")
    var n_at = ONE_HEAD + lb - 32
    if not registry.trusts_key(_id(List[UInt8](inputs[n_at:n_at + lb])), Int(inputs[1])):
        raise Error("the CSCA key is not in the registry")
    var upper = pkcs1_upper(Int(inputs[0]))
    for i in range(lb - 32):
        if inputs[ONE_HEAD + i] != upper[i] or inputs[n_at + lb + i] != upper[i]:
            raise Error("the padding limbs are not PKCS#1 v1.5 with a SHA-256 DigestInfo")
