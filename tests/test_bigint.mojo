from std.testing import assert_equal, assert_true, TestSuite
from caracal7.relations.bigint import Big


def test_arithmetic() raises:
    var a = Big.from_hex("fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f")     # secp256k1 p
    assert_equal(a.bit_length(), 256)
    var two256 = Big(1).shl(256)
    assert_equal(two256 - a, Big((1 << 32) + 977))
    assert_equal(String(Big(-1234567890123)), "-1234567890123")
    var sq = a * a
    var qr = sq.divmod(a)
    assert_equal(qr[0], a)
    assert_true(qr[1].is_zero())
    var neg = Big(-7).divmod(Big(3))
    assert_equal(neg[0], Big(-3))
    assert_equal(neg[1], Big(2))
    assert_equal((Big(-7) + Big(7)).to_int(), 0)
    assert_equal((Big(5) - Big(9)).to_int(), -4)
    var x = Big.from_hex("123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
    var inv = x.inv_mod(a)
    assert_equal(x.mulmod(inv, a), Big(1))
    assert_equal(Big.from_bytes(x.bytes(32)), x)
    assert_equal(Big.from_bits(x.bits(256)), x)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
