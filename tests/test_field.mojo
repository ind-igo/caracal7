from std.testing import assert_equal, assert_true, assert_raises, TestSuite
from caracal7.field import *


def test_f_add_sub_wrap() raises:
    assert_equal(f_add(SIMD[DType.uint8, 1](126), SIMD[DType.uint8, 1](3))[0], 2)
    assert_equal(f_sub(SIMD[DType.uint8, 1](2), SIMD[DType.uint8, 1](3))[0], 126)
    assert_equal(f_neg(SIMD[DType.uint8, 1](0))[0], 0)
    assert_equal(f_neg(SIMD[DType.uint8, 1](5))[0], 122)


def test_f_mul_all_pairs() raises:
    for a in range(127):
        for b in range(127):
            assert_equal(Int(f_mul(SIMD[DType.uint8, 1](a), SIMD[DType.uint8, 1](b))[0]), (a * b) % 127)


def test_f_reduce_matches_mod() raises:
    for x in range(0, 1 << 21, 997):
        assert_equal(Int(f_reduce(SIMD[DType.uint32, 1](x))[0]), x % 127)
    assert_equal(Int(f_reduce(SIMD[DType.uint32, 1]((1 << 21) - 1))[0]), ((1 << 21) - 1) % 127)


def test_f_inv_all_nonzero() raises:
    for a in range(1, 127):
        var x = Scalar[DType.uint8](a)
        assert_equal(f_mul(x, f_inv(x))[0], 1)
    with assert_raises():
        _ = f_inv(0)


def test_simd_lanes_agree_with_scalar() raises:
    var a = SIMD[DType.uint8, 16]()
    var b = SIMD[DType.uint8, 16]()
    for t in range(16):
        a[t] = UInt8((t * 37 + 11) % 127)
        b[t] = UInt8((t * 53 + 100) % 127)
    var m = f_mul(a, b)
    var s = f_add(a, b)
    for t in range(16):
        assert_equal(Int(m[t]), (Int(a[t]) * Int(b[t])) % 127)
        assert_equal(Int(s[t]), (Int(a[t]) + Int(b[t])) % 127)


def test_tower_constants_are_nonsquares() raises:
    # x is a square in F_{q} iff x^((q-1)/2) == 1. Check C1..C3 directly; C4 is
    # implied (u^((127^8-1)/2) = j^((127^8-1)/4) = -1 by the same argument).
    var one1 = SIMD[DType.uint8, 1](1)
    assert_equal(f_pow(C1, 63), SIMD[DType.uint8, 1](126))
    var m1 = f_neg(ext_embed[1, 1](one1))
    assert_equal(ext_pow[1](C2, (127 * 127 - 1) // 2), m1)
    var q4 = 127 * 127 * 127 * 127
    assert_equal(ext_pow[2](C3, (q4 - 1) // 2), ext_embed[2, 1](SIMD[DType.uint8, 1](126)))


def test_i_squared_is_minus_one() raises:
    var i = F2(0, 1)
    assert_equal(ext_mul[1](i, i), F2(126, 0))


def _sample_e(seed: Int) -> E:
    var v = E()
    for t in range(16):
        v[t] = UInt8((seed * (t + 1) * 31 + t * t * 7 + 3) % 127)
    return v


def test_e_ring_identities() raises:
    var a = _sample_e(1)
    var b = _sample_e(2)
    var c = _sample_e(3)
    assert_equal(ext_mul[4](a, b), ext_mul[4](b, a))
    assert_equal(ext_mul[4](ext_mul[4](a, b), c), ext_mul[4](a, ext_mul[4](b, c)))
    assert_equal(ext_mul[4](a, f_add(b, c)), f_add(ext_mul[4](a, b), ext_mul[4](a, c)))
    var one = ext_embed[4, 1](SIMD[DType.uint8, 1](1))
    assert_equal(ext_mul[4](a, one), a)


def test_e_inverse() raises:
    var one = ext_embed[4, 1](SIMD[DType.uint8, 1](1))
    for s in range(1, 6):
        var a = _sample_e(s)
        assert_equal(ext_mul[4](a, ext_inv[4](a)), one)
    with assert_raises():
        _ = ext_inv[4](E())


def test_f4_scalar_action_is_coordinatewise() raises:
    # (F4 scalar) * (E element) == 4 independent F4 products on the F4 coordinates.
    var s = F4(3, 5, 7, 11)
    var a = _sample_e(9)
    var full = ext_mul[4](ext_embed[4, 4](s), a)
    comptime for t in range(4):
        var coord = a.slice[4, offset = 4 * t]()
        assert_equal(full.slice[4, offset = 4 * t](), ext_mul[2](s, coord))


def test_f4_mac_wide_matches_ext_mul() raises:
    var acc = SIMD[DType.int32, 4](0)
    var want_sum = F4(0)
    for n in range(F4_MAC_MAX):
        var a = F4(UInt8((n * 37 + 5) % 127), UInt8((n * 91 + 126) % 127), UInt8((n * 13) % 127), UInt8((n * 71 + 100) % 127))
        var b = F4(UInt8((n * 59 + 2) % 127), UInt8(126), UInt8((n * 17 + 60) % 127), UInt8((n * 3 + 125) % 127))
        f4_mac_wide(acc, a, b)
        want_sum = f_add(want_sum, ext_mul[2](a, b))
    assert_equal(f_reduce_signed(acc), want_sum)
    # a worst-case negative accumulator still reduces
    var neg = SIMD[DType.int32, 4](0)
    for _ in range(F4_MAC_MAX):
        f4_mac_wide(neg, F4(0, 126, 0, 126), F4(0, 126, 126, 0))
    var want = F4(0)
    for _ in range(F4_MAC_MAX):
        want = f_add(want, ext_mul[2](F4(0, 126, 0, 126), F4(0, 126, 126, 0)))
    assert_equal(f_reduce_signed(neg), want)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
