from std.testing import assert_equal, assert_true, assert_raises, TestSuite
from core.field import *


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
    # x is a square in F_{q} iff x^((q-1)/2) == 1. Check C1..C3 directly.
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
    for t in range(E_BYTES):
        v[t] = UInt8((seed * (t + 1) * 31 + t * t * 7 + 3) % 127)
    return v


def _e_mul_reference(a: E20, b: E20) -> E20:
    """Polynomial product of the limbs modulo v^5 - G5, limb by limb in F4."""
    var c = InlineArray[F4, 9](fill=F4(0))
    for m in range(5):
        for n in range(5):
            var am = F4(a[4 * m], a[4 * m + 1], a[4 * m + 2], a[4 * m + 3])
            var bn = F4(b[4 * n], b[4 * n + 1], b[4 * n + 2], b[4 * n + 3])
            c[m + n] = f_add(c[m + n], ext_mul[2](am, bn))
    var r = E20(0)
    for t in range(5):
        var lo = c[t]
        if t < 4:
            lo = f_add(lo, ext_mul[2](G5, c[t + 5]))
        for l in range(4):
            r[4 * t + l] = lo[l]
    return r


def test_g5_is_not_a_fifth_power_and_zeta() raises:
    comptime if not E16:
        var q4 = 127 * 127 * 127 * 127
        var one = F4(1, 0, 0, 0)
        var zeta = ext_pow[2](G5, (q4 - 1) // 5)
        assert_true(zeta != one)
        assert_equal(zeta, ZETA1)
        assert_equal(ext_mul[2](ZETA1, ZETA1), ZETA2)
        assert_equal(ext_mul[2](ZETA2, ZETA1), ZETA3)
        assert_equal(ext_mul[2](ZETA3, ZETA1), ZETA4)
        assert_equal(ext_mul[2](ZETA4, ZETA1), one)


def test_e_ring_identities() raises:
    var a = _sample_e(1)
    var b = _sample_e(2)
    var c = _sample_e(3)
    assert_equal(ext_mul[E_LEVEL](a, b), ext_mul[E_LEVEL](b, a))
    assert_equal(ext_mul[E_LEVEL](ext_mul[E_LEVEL](a, b), c), ext_mul[E_LEVEL](a, ext_mul[E_LEVEL](b, c)))
    assert_equal(ext_mul[E_LEVEL](a, f_add(b, c)), f_add(ext_mul[E_LEVEL](a, b), ext_mul[E_LEVEL](a, c)))
    var one = ext_one[E_LEVEL]()
    assert_equal(ext_mul[E_LEVEL](a, one), a)


def test_e_product_matches_the_reference_and_pads_zero() raises:
    comptime if not E16:
        for s in range(64):
            var a = _sample_e(s)
            var b = _sample_e(s * 7 + 11)
            var got = ext_mul[E_LEVEL](a, b)
            assert_equal(rebind[E20](got), _e_mul_reference(rebind[E20](a), rebind[E20](b)))
            for t in range(E_BYTES, E_WIDTH):
                assert_equal(Int(got[t]), 0)
            var fp = fp_canonical(fp_ext_mul[E_LEVEL](a.cast[DType.float32](), b.cast[DType.float32]()))
            assert_equal(fp, got)
            var c = fp_canonical(fp_ext_mul[E_LEVEL](fp_center(a), fp_center(b)))
            assert_equal(c, got)
        var one = ext_one[E_LEVEL]()
        assert_equal(ext_mul[E_LEVEL](one, one), one)


def test_e_frobenius_is_the_q_power() raises:
    comptime if not E16:
        var q4 = 127 * 127 * 127 * 127
        var a = rebind[E20](_sample_e(5))
        assert_equal(e_frobenius[1](a), rebind[E20](ext_pow[E_LEVEL](rebind[E](a), q4)))
        assert_equal(e_frobenius[2](a), e_frobenius[1](e_frobenius[1](a)))
        assert_equal(e_frobenius[4](a), e_frobenius[2](e_frobenius[2](a)))
        assert_equal(e_frobenius[1](e_frobenius[4](a)), a)


def test_e_inverse() raises:
    var one = ext_one[E_LEVEL]()
    for s in range(1, 6):
        var a = _sample_e(s)
        assert_equal(ext_mul[E_LEVEL](a, ext_inv[E_LEVEL](a)), one)
    assert_equal(ext_mul[E_LEVEL](ext_embed[E_LEVEL, 4](F4(3, 5, 7, 11)), ext_inv[E_LEVEL](ext_embed[E_LEVEL, 4](F4(3, 5, 7, 11)))), one)
    with assert_raises():
        _ = ext_inv[E_LEVEL](E())
    assert_equal(ext_inv0[E_LEVEL](E()), E())


def test_f4_scalar_action_is_coordinatewise() raises:
    comptime if not E16:
        # (F4 scalar) * (E element) == 5 independent F4 products on the limbs.
        var s = F4(3, 5, 7, 11)
        var a = _sample_e(9)
        var full = ext_mul[E_LEVEL](ext_embed[E_LEVEL, 4](s), a)
        assert_equal(rebind[E20](full), e_scale(s, rebind[E20](a)))
        comptime for t in range(5):
            var coord = a.slice[4, offset = 4 * t]()
            assert_equal(full.slice[4, offset = 4 * t](), ext_mul[2](s, coord))


def test_e16_product_matches_the_schoolbook() raises:
    comptime if E16:
        var one = ext_one[E_LEVEL]()
        for s in range(64):
            var a = rebind[E16T](_sample_e(s))
            var b = rebind[E16T](_sample_e(s * 7 + 11))
            var a0 = a.slice[8]()
            var a1 = a.slice[8, offset=8]()
            var b0 = b.slice[8]()
            var b1 = b.slice[8, offset=8]()
            var lo = f_add(ext_mul[3](a0, b0), ext_mul[3](C4, ext_mul[3](a1, b1)))
            var hi = f_add(ext_mul[3](a0, b1), ext_mul[3](a1, b0))
            assert_equal(ext_mul[E_LEVEL](rebind[E](a), rebind[E](b)), rebind[E](lo.join(hi)))
            assert_equal(e_from_power(e_to_power(a)), a)
            var fp = fp_canonical(fp_ext_mul[E_LEVEL](rebind[E](a).cast[DType.float32](), rebind[E](b).cast[DType.float32]()))
            assert_equal(fp, rebind[E](lo.join(hi)))
        assert_equal(ext_mul[E_LEVEL](one, one), one)


def test_e16_f4_scalar_action_is_coordinatewise() raises:
    comptime if E16:
        var s = F4(3, 5, 7, 11)
        var a = _sample_e(9)
        var full = ext_mul[E_LEVEL](ext_embed[E_LEVEL, 4](s), a)
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
