"""The skeleton: the arena plan builds for the reference profile and the prover stops at the first
stage that does not exist, in spec order."""

from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceContext

from caracal7.params import REFERENCE, Params
from caracal7.hash import Blake3
from caracal7.proof import Shape, tail_schedule
from caracal7.prover import Prover, ProverLayout, load_trace
from caracal7.verifier import verify

comptime p = REFERENCE


def test_tail_schedule_reference_is_clear_at_level_2() raises:
    var s = tail_schedule[p]()
    assert_equal(len(s), 0)          # N = 2304 <= tail_clear_max: y_2 is the clear vector
    var shape = Shape.__init__[p](53, 34)
    assert_equal(shape.clear_length, p.N())
    assert_equal(shape.columns(), 53 + 48)


def test_tail_schedule_folds_a_larger_grid() raises:
    # 288 x 128: N = 36864 -> level 2 rows 4608 on L = 80640 -> level 3 rows 576 on L = 10080 -> clear
    comptime big = Params(e=16, a1=5, m1=9, a2=7, m2=1, L0=161280, m_cosets=1, leaf_bytes=1024,
                          tail_digits=3, tail_clear_max=2500, lambda_bits=103)
    var s = tail_schedule[big]()
    assert_equal(len(s), 2)
    assert_equal(s[0].rows, 4608)
    assert_equal(s[0].L, 80640)
    assert_equal(s[1].rows, 576)
    assert_equal(s[1].L, 10080)
    assert_true(s[0].queries >= 100 and s[0].queries <= 115)
    var shape = Shape.__init__[big](357, 34)
    assert_equal(shape.clear_length, 576)


def test_layout_plans_the_arena() raises:
    var shape = Shape.__init__[p](53, 34)
    var L = ProverLayout.__init__[p, Blake3](shape)
    assert_true(L.bytes > 0)
    assert_equal(len(L.tail), 0)
    # every offset is inside the arena and 256-aligned
    for off in [L.tree_w, L.tree_q, L.lde, L.residual, L.quotient, L.w_z, L.openings, L.fold_y, L.positions, L.proof_stage]:
        assert_true(off < L.bytes and off % 256 == 0)
    print("arena for 53 + 48 columns:", L.bytes // (1 << 20), "MiB")


def test_prove_stops_at_first_missing_stage() raises:
    var ctx = DeviceContext()
    var prover = Prover[p, Blake3](ctx, Shape.__init__[p](53, 34))
    var trace = List[UInt8](length=53 * p.N(), fill=0)
    load_trace[p, Blake3](ctx, prover, trace)
    var stopped = String("")
    try:
        _ = prover.prove(ctx, List[UInt8]())
    except e:
        stopped = String(e)
    assert_equal(stopped, "not implemented: lde")


def test_verify_stops_at_first_missing_step() raises:
    var shape = Shape.__init__[p](53, 34)
    var bytes = List[UInt8]()
    for b in [1, 0, 0, 0, 0, 0, 0, 0]:          # version 1, empty public inputs
        bytes.append(UInt8(b))
    var stopped = String("")
    try:
        _ = verify[p, Blake3](bytes^, shape, List[UInt8]())
    except e:
        stopped = String(e)
    assert_equal(stopped, "not implemented: verifier step 1 (prefix)")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
