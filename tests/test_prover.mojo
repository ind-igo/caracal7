"""The skeleton: the arena plan builds for the reference profile and the prover stops at the first
stage that does not exist, in spec order."""

from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceContext

from caracal7.params import REFERENCE, Params
from caracal7.hash import Blake3
from caracal7.proof import Shape, tail_schedule
from caracal7.prover import Prover, ProverLayout, load_trace
from caracal7.verifier import verify
from caracal7.residual import synthetic_families, synthetic_trace, SYNTHETIC_COLUMNS

comptime p = REFERENCE


def test_tail_schedule_reference_is_clear_at_level_2() raises:
    var s = tail_schedule[p]()
    assert_equal(len(s), 0)          # N = 2304 <= tail_clear_max: y_2 is the clear vector
    var shape = Shape.__init__[p](53, synthetic_families().bytes)
    assert_equal(shape.clear_length, p.N())
    assert_equal(shape.columns(), 53 + 48)


def test_tail_schedule_folds_a_larger_grid() raises:
    # 288 x 128 (spec 9.5 worked row): N = 36864 -> 4608 rows on 161280 (rate 1/35) -> 576 rows on 18432 = 4 x 4608 (1/32) -> clear
    comptime big = Params(e=16, a1=5, m1=9, a2=7, m2=1, L0=161280, m_cosets=1, leaf_bytes=1024,
                          tail_digits=3, tail_clear_max=2500, lambda_bits=103)
    var s = tail_schedule[big]()
    assert_equal(len(s), 2)
    assert_equal(s[0].rows, 4608)
    assert_equal(s[0].L, 161280)
    assert_equal(s[0].cosets, 1)
    assert_equal(s[1].rows, 576)
    assert_equal(s[1].L, 18432)
    assert_equal(s[1].cosets, 4)
    assert_true(s[0].queries >= 100 and s[0].queries <= 115)
    var shape = Shape.__init__[big](357, synthetic_families().bytes)
    assert_equal(shape.clear_length, 576)


def test_tail_schedule_stops_when_binary_digits_run_out() raises:
    # spec 9.5 narrow row, 1008 x 252: 6 binary digits -> two folds -> 3969 in the clear (odd digit only)
    comptime narrow = Params(e=16, a1=4, m1=63, a2=2, m2=63, L0=161280, m_cosets=4, leaf_bytes=1024,
                             tail_digits=3, tail_clear_max=2500, lambda_bits=103)
    var s = tail_schedule[narrow]()
    assert_equal(len(s), 2)
    assert_equal(s[0].rows, 31752)
    assert_equal(s[0].L, 645120)          # the largest domain, rate 1/20: the cap
    assert_equal(s[0].cosets, 4)
    assert_equal(s[1].rows, 3969)
    assert_equal(s[1].L, 129024)         # 4 cosets of 32256, rate 1/32.5
    var shape = Shape.__init__[narrow](43, synthetic_families().bytes)
    assert_equal(shape.clear_length, 3969)


def test_layout_plans_the_arena() raises:
    var shape = Shape.__init__[p](53, synthetic_families().bytes)
    var L = ProverLayout.__init__[p, Blake3](shape)
    assert_true(L.bytes > 0)
    assert_equal(len(L.tail), 0)
    # every offset is inside the arena and 256-aligned
    for off in [L.tree_w, L.tree_q, L.families, L.shifts, L.ltmp, L.lde, L.residual, L.quotient, L.w_z, L.openings, L.fold_y, L.positions, L.proof_stage,
                L.prefix, L.stage1, L.z, L.beta_gamma, L.batch, L.r]:
        assert_true(off < L.bytes and off % 256 == 0)
    print("arena for 53 + 48 columns:", L.bytes // (1 << 20), "MiB")


def test_prove_and_verify_reach_the_tail() raises:
    """The reference profile has no committed tail level, so the level-1 stages are the whole proof:
    prove finishes, the verifier passes steps 1, 2 and 5 and stops at the tail; a tampered opening
    fails the residual identity."""
    var ctx = DeviceContext()
    var f = synthetic_families()
    var shape = Shape.__init__[p](SYNTHETIC_COLUMNS, f.bytes)
    assert_equal(shape.points, 4)                 # z, (omega1 z1, z2), (omega1^3 z1, z2), (z1, omega2 z2)
    var prover = Prover[p, Blake3](ctx, Shape.__init__[p](SYNTHETIC_COLUMNS, f.bytes), f.bytes.copy())
    load_trace[p, Blake3](ctx, prover, synthetic_trace[p](1))
    var proof = prover.prove(ctx, List[UInt8]())
    var fixed = shape.fixed_bytes[p, 32](0)
    assert_true(len(proof) > fixed, "proof shorter than its fixed part")
    print("proof bytes:", len(proof), " fixed:", fixed)
    var stopped = String("")
    try:
        _ = verify[p, Blake3](proof.copy(), shape, List[UInt8](), f.bytes)
    except e:
        stopped = String(e)
    assert_equal(stopped, "not implemented: verifier step 7 (tail)")
    var bad = proof.copy()
    bad[8 + 64 + 5] ^= 1                          # inside the openings
    stopped = ""
    try:
        _ = verify[p, Blake3](bad^, shape, List[UInt8](), f.bytes)
    except e:
        stopped = String(e)
    assert_equal(stopped, "residual identity fails at z")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
