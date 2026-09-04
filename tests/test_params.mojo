from std.testing import assert_equal, assert_true, assert_raises, TestSuite
from caracal7.params import Params, REFERENCE


def test_reference_profile() raises:
    REFERENCE.check()
    assert_equal(REFERENCE.h1(), 72)
    assert_equal(REFERENCE.h2(), 32)
    assert_equal(REFERENCE.N(), 2304)
    assert_equal(REFERENCE.L(), 80640)
    assert_equal(REFERENCE.leaf_columns_max(), 256)
    # rate 2304 / (4 * 80640) = 1/140; queries = ceil(103 / log2(2 / (1 + 1/140))) = ceil(104.07) = 105
    assert_equal(REFERENCE.queries(), 105)


def test_query_formula_matches_spec_rows() raises:
    # Synthetic profiles exercise the rate and query formulas.
    var q = Params(e=16, a1=7, m1=63, a2=5, m2=5, L0=161280, m_cosets=4, leaf_bytes=1024,
                   tail_digits=3, tail_clear_max=2500, lambda_bits=103)
    assert_equal(q.N(), 8064 * 160)
    assert_equal(q.L(), 645120)
    assert_equal(q.rate(), 0.5)  # N / (4 L) = 1290240 / 2580480
    q.m2 = 3
    # N = 8064 * 96 = 774144; rate = 0.3
    assert_true(q.queries() > 104)


def test_check_rejects_bad_profiles() raises:
    var p = REFERENCE
    p.a1 = 1
    with assert_raises():
        p.check()
    p = REFERENCE
    p.L0 = 1000
    with assert_raises():
        p.check()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
