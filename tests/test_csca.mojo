"""The verifier's CSCA registry and the passport's public-input check (workloads/csca.mojo), host only: the
bench fixtures' public inputs (bench_sod, bench_dsc: an RSA-2048 CSCA and DSC key from OpenSSL) without a
proof."""

from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from workloads.bigint import Big
from workloads.sha256 import sha256
from workloads.csca import Registry, csca_key_id, csca_key_id_of, exponent_muls, pkcs1_upper, passport_check
from workloads.dsc import dsc_head, HEAD
from workloads.sod import sod_head, HEAD as SOD_HEAD

comptime N_CSCA_HEX = "e033cfbd808b0902536d51580f7e53665c1bf9da78076eee38f95709579ed0a1985729e35da0b0ccb18c936e045c51e21087ed7031c3f94e9cc8b7e400d95cb7cf46bf553fbe439c1cbc7220b2a2afcfdafb13cc8314029da6ff30e82d09451e1857c6a88a8002569e6f0e7405a58dff739f76ea2c12098afa41e26cddd318fcbe1eceed0a4a8089d0c16b4db58b88a20c28a8737a44e3e82fcbe97f5fe0d888032af37b36dfca514d3c1d70b4dc68ceafa9de77fa7a89378f5dde7f8f6536b5f8c919fc09ef65e36a88532f2614e948980f503d968d329162a8d575e7e2218c63408313275562fdcd4ff578fcd267ea2146904c6b52191f5f921bf3dabb3735"
comptime M_HEX = "0001ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff003031300d0609608648016503040201050004209adac772ef5475b6b2624243c606915e185503a5858d3738f3040ca325ab554b"


def _hex(v: List[UInt8]) raises -> String:
    comptime D = "0123456789abcdef"
    var s = String("")
    for b in v:
        s += String(D[byte=Int(b >> 4)]) + String(D[byte=Int(b & 15)])
    return s


def _sod_inputs() raises -> List[UInt8]:
    var sod = sod_head(8, 17, [93, 100, 73], [29, 41], 59)
    sod.extend(Big.from_hex(M_HEX).shr(256).bytes(224))
    sod.extend(sha256([1, 2, 3]))
    sod.extend(List[UInt8](length=96, fill=7))
    return sod^


def _dsc_inputs(n_csca: Big) raises -> List[UInt8]:
    var dsc = dsc_head(8, 17, 1000, 300)
    dsc.extend(n_csca.bytes(256))
    dsc.extend(Big.from_hex(M_HEX).shr(256).bytes(224))
    dsc.extend(sha256([1, 2, 3]))
    return dsc^


def test_registry_and_passport_check() raises:
    var n = Big.from_hex(N_CSCA_HEX)
    var nb = n.bytes((n.bit_length() + 7) // 8)
    nb.reverse()
    assert_equal(csca_key_id(n), sha256(nb))
    var sod = _sod_inputs()
    var dsc = _dsc_inputs(n)
    assert_equal(csca_key_id_of(dsc), sha256(nb))
    # the padding limbs of the OpenSSL fixture are the canonical encoding
    assert_equal(pkcs1_upper(8), Big.from_hex(M_HEX).shr(256).bytes(224))
    assert_equal(exponent_muls(3), 2)
    assert_equal(exponent_muls(65537), 17)
    var reg = Registry.parse("# list\r\n" + _hex(sha256(nb)).upper() + " 65537 # csca\r\n\r\n" + _hex(sha256([9])) + "\n")
    assert_equal(len(reg.ids), 2)
    assert_true(reg.trusts(dsc))
    passport_check(sod, dsc, reg)
    # another exponent, a key off the list
    var e3 = dsc.copy()
    e3[1] = 2
    assert_true(not reg.trusts(e3))
    with assert_raises(contains="not in the registry"):
        passport_check(sod, e3, Registry.parse(_hex(sha256([9]))))
    # a padding limb the prover chose
    var pad = dsc.copy()
    pad[HEAD + 256 + 100] ^= 1
    with assert_raises(contains="PKCS#1"):
        passport_check(sod, pad, reg)
    var pad2 = sod.copy()
    pad2[SOD_HEAD] ^= 1
    with assert_raises(contains="PKCS#1"):
        passport_check(pad2, dsc, reg)
    # two commitments
    var cm = sod.copy()
    cm[SOD_HEAD + 224 + 31] ^= 1
    with assert_raises(contains="one commitment"):
        passport_check(cm, dsc, reg)
    with assert_raises(contains="cut short"):
        passport_check(sod, List[UInt8](dsc[:HEAD + 300]), reg)
    with assert_raises(contains="64 hex digits"):
        _ = Registry.parse("abc")
    with assert_raises(contains="2^k + 1"):
        _ = Registry.parse(_hex(sha256(nb)) + " 7")
    with assert_raises(contains="hex"):
        _ = Registry.parse("zz" + String("0") * 62)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
