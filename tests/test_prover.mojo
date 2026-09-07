"""The skeleton: the arena plan builds for the reference profile and the prover stops at the first
stage that does not exist, in spec order."""

from std.testing import assert_equal, assert_true, assert_false, TestSuite
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from caracal7.core.params import REFERENCE, WIDE, CLIENT, Params
from caracal7.core.hash import Blake3
from caracal7.proof import Shape, tail_schedule
from caracal7.prover import Prover, ProverLayout, load_trace, load_advice, load_public
from caracal7.verifier import verify
from caracal7.relations import shift_points, standard_chals, POINT, CHAL_MUL, CHAL_ADD, CHAL_ONE
from caracal7.core.bytes import set_u16
from caracal7.relations.statement import restriction_line, chain_values
from caracal7.workload import prove_workload, verify_workload
from caracal7.relations.synthetic import Synthetic
from caracal7.relations.synthetic import synthetic_statement, synthetic_trace, synthetic_table, synthetic_advice, synthetic_public_block, SYNTHETIC_COLUMNS, SYNTHETIC_LOOKUP_COLUMNS, SYNTHETIC_PUBLIC_COLUMNS

comptime p = REFERENCE


def test_tail_schedule_reference_is_clear_at_level_2() raises:
    var s = tail_schedule[p]()
    assert_equal(len(s), 0)          # N = 2304 <= tail_clear_max: y_2 is the clear vector
    var shape = synthetic_statement(53).compile[p]().take_shape()
    assert_equal(shape.clear_length, p.N())
    assert_equal(shape.columns(), 53 + 32 + 48)


def test_tail_schedule_folds_a_larger_grid() raises:
    # 288 x 128 (spec 9.5 worked row): N = 36864 -> 4608 rows on 161280 (rate 1/35) -> 576 rows on 18432 = 4 x 4608 (1/32) -> clear
    comptime big = WIDE
    var s = tail_schedule[big]()
    assert_equal(len(s), 2)
    assert_equal(s[0].rows, 4608)
    assert_equal(s[0].L, 161280)
    assert_equal(s[0].cosets, 1)
    assert_equal(s[1].rows, 576)
    assert_equal(s[1].L, 18432)
    assert_equal(s[1].cosets, 4)
    assert_true(s[0].queries >= 100 and s[0].queries <= 115)
    var shape = synthetic_statement(357).compile[big]().take_shape()
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
    var shape = synthetic_statement(43).compile[narrow]().take_shape()
    assert_equal(shape.clear_length, 3969)


def test_layout_plans_the_arena() raises:
    var shape = synthetic_statement(53).compile[p]().take_shape()
    var L = ProverLayout.__init__[p, Blake3](shape)
    assert_true(L.bytes > 0)
    assert_equal(len(L.tail), 0)
    # every offset is inside the arena and 256-aligned
    for off in [L.tree_w, L.tree_z, L.tree_q, L.families, L.accs, L.shifts, L.num, L.den, L.zscratch, L.zval, L.chain_prod, L.z2, L.n_end, L.d_end,
                L.sg, L.q3, L.ltmp, L.lde, L.residual, L.quotient, L.w_z, L.openings, L.open_partial, L.fold_y, L.running0, L.dom1, L.pts, L.partial,
                L.positions, L.proof_stage, L.prefix, L.stage1, L.alpha, L.z, L.beta_gamma, L.batch, L.r]:
        assert_true(off < L.bytes and off % 256 == 0)
    print("arena for 53 + 32 + 48 columns:", L.bytes // (1 << 20), "MiB")


def test_client_profile_plans_but_does_not_fit_16gb() raises:
    """The client grid (2016 x 576) plans: two tail levels at five digits each, 223 queries at n_cw = 1, and an
    arena of 5.3 GB at 91 W columns, 11.7 GB at 357 (params.mojo). The numbers are the ceilings the perf pass
    lifts (codeword split, arena reuse); the test pins the plan so a change shows."""
    CLIENT.check()
    assert_equal(CLIENT.queries(), 223)
    assert_equal(len(tail_schedule[CLIENT]()), 2)
    var L = ProverLayout.__init__[CLIENT, Blake3](synthetic_statement(91, with_lookup=True).compile[CLIENT]().take_shape())
    assert_true(L.bytes > 5000 * (1 << 20) and L.bytes < 5500 * (1 << 20))


def test_prove_and_verify() raises:
    """The reference profile has no committed tail level, so the level-1 stages are the whole proof:
    it verifies end to end, and one flipped byte in each region fails the check that owns it."""
    var ctx = DeviceContext()
    var c = synthetic_statement().compile[p]()
    var shape = synthetic_statement().compile[p]().take_shape()
    assert_equal(shape.points, 8)                 # z, the four boundary points, then the reads (omega1 z1, z2), (omega1^3 z1, z2), (z1, omega2 z2)
    var prover = Prover[p, Blake3](ctx, synthetic_statement().compile[p]().take_shape(), c.families.copy())
    load_trace[p, Blake3](ctx, prover, synthetic_trace[p](1))
    var proof = prover.prove(ctx, List[UInt8]())
    var fixed = shape.fixed_bytes[p, 32](0)
    assert_true(len(proof) > fixed, "proof shorter than its fixed part")
    print("proof bytes:", len(proof), " fixed:", fixed)
    assert_true(verify[p, Blake3](proof.copy(), shape, List[UInt8](), c.families))
    var q3_bytes = 8 + 3 * 32 + shape.accumulators() * p.h2() * p.e
    var openings = q3_bytes + 2 * p.h2() * p.e
    var clear = openings + shape.points * shape.columns() * p.e
    var multiproof = clear + shape.clear_length * p.e
    # a changed clear vector moves S, so the multiproof no longer parses; the consistency check
    # itself only sees a dishonest y with a matching frontier, which no byte flip produces
    var z2_bytes = 8 + 2 * 32
    var z_open = openings + (shape.columns() + shape.columns_w) * p.e         # Z coordinate 0 at point 1 = (1, z2)
    for tamper in [(openings + 5, "residual identity fails at z"),
                   (z2_bytes + 1, "Z2(1) is not 1"),
                   (z2_bytes + (p.h2() - 1) * p.e + 2, "accumulator grand product is not 1"),
                   (z_open + 3, "accumulator chain start is not 1"),
                   (q3_bytes + 7, "small grid identity fails at z2"),
                   (clear + 3, "multiproof"),          # truncated or trailing bytes, by where S lands
                   (multiproof + 4 + 7, "multiproof root mismatch"),
                   (len(proof) - 1, "multiproof root mismatch")]:
        var bad = proof.copy()
        bad[tamper[0]] ^= 1
        var stopped = String("")
        try:
            _ = verify[p, Blake3](bad^, shape, List[UInt8](), c.families)
        except e:
            stopped = String(e)
        assert_true(stopped.startswith(tamper[1]), stopped)


def test_prove_and_verify_without_accumulators() raises:
    """No Z tree: a statement with no accumulator (Keccak-128 by the statement layer) carries no Z root,
    no Z2, no Q3, and opens two trees."""
    var ctx = DeviceContext()
    var c = synthetic_statement(with_accumulator=False).compile[p]()
    var shape = synthetic_statement(with_accumulator=False).compile[p]().take_shape()
    assert_equal(shape.columns_z, 0)
    assert_equal(shape.points, 4)                 # z and the three reads: no boundary points without accumulators
    var prover = Prover[p, Blake3](ctx, synthetic_statement(with_accumulator=False).compile[p]().take_shape(), c.families.copy())
    load_trace[p, Blake3](ctx, prover, synthetic_trace[p](1, with_accumulator=False))
    var proof = prover.prove(ctx, List[UInt8]())
    assert_true(verify[p, Blake3](proof^, shape, List[UInt8](), c.families))


def test_invalid_permutation_is_rejected() raises:
    """An honest prover on a witness whose sorted copy is not a permutation: every commitment and
    challenge is fresh, and the grand product fails."""
    var ctx = DeviceContext()
    var c = synthetic_statement().compile[p]()
    var shape = synthetic_statement().compile[p]().take_shape()
    var prover = Prover[p, Blake3](ctx, synthetic_statement().compile[p]().take_shape(), c.families.copy())
    var trace = synthetic_trace[p](1)
    trace[8 * p.N() + 7] = UInt8((Int(trace[8 * p.N() + 7]) + 1) % 127)
    load_trace[p, Blake3](ctx, prover, trace)
    var proof = prover.prove(ctx, List[UInt8]())
    var stopped = String("")
    try:
        _ = verify[p, Blake3](proof^, shape, List[UInt8](), c.families)
    except e:
        stopped = String(e)
    assert_equal(stopped, "accumulator grand product is not 1")


def _point(dj1: Int, dj2: Int) -> List[UInt8]:
    var pt = List[UInt8](length=POINT, fill=0)
    set_u16(pt, 0, dj1)
    set_u16(pt, 2, dj2)
    return pt^


def test_point_list_and_derivation_table_are_artifact_inputs() raises:
    """An explicit list with one unused extra point proves and verifies; a list missing a read is rejected;
    so is an entry naming a challenge past the table, a row over a later element, and a table without the
    accumulator rows. A longer table on a statement with accumulators proves and verifies."""
    var ctx = DeviceContext()
    var c = synthetic_statement().compile[p]()
    var extra = shift_points(c.families)
    extra.extend(_point(4, 4))
    var table = standard_chals()
    table.extend([CHAL_MUL, 4, 2])
    var shape = Shape.__init__[p](SYNTHETIC_COLUMNS, c.families, c.shape.accs, points=extra, chals=table)
    assert_equal(shape.points, 9)
    assert_equal(shape.chal_count(), 6)
    var prover = Prover[p, Blake3](ctx, Shape.__init__[p](SYNTHETIC_COLUMNS, c.families, c.shape.accs, points=extra, chals=table), c.families.copy())
    load_trace[p, Blake3](ctx, prover, synthetic_trace[p](1))
    var proof = prover.prove(ctx, List[UInt8]())
    assert_true(verify[p, Blake3](proof^, shape, List[UInt8](), c.families))
    var short = shift_points(c.families)
    short.resize(len(short) - POINT, 0)
    var bad_fam = c.families.copy()
    bad_fam[32] = UInt8(7)                        # entry 0 names element 6
    assert_equal(_rejected(short, standard_chals(), c.families, c.shape.accs), "opening points must include every read shift, restriction line, and accumulator boundary point")
    assert_equal(_rejected(shift_points(c.families), standard_chals(), bad_fam, c.shape.accs), "family entry names a challenge element past the derivation table")
    assert_equal(_rejected(shift_points(c.families), [CHAL_ADD, 0, CHAL_ONE, CHAL_MUL, 5, 1], c.families, c.shape.accs), "challenge derivation row must add or multiply earlier elements")
    assert_equal(_rejected(shift_points(c.families), [CHAL_ADD, 0, CHAL_ONE], c.families, c.shape.accs), "accumulators need the derivation table to start with 1 + beta and (1 + beta) delta")


def _rejected(points: List[UInt8], chals: List[UInt8], families: List[UInt8], accs: List[UInt8]) -> String:
    try:
        _ = Shape.__init__[p](SYNTHETIC_COLUMNS, families, accs, points=points, chals=chals)
    except e:
        return String(e)
    return String("")


def _lookup_case(ctx: DeviceContext, var trace: List[UInt8], var advice: List[UInt8]) raises -> String:
    """Prove the lookup instance on `trace` with `advice` and verify; the verifier's error, or empty."""
    var c = synthetic_statement(SYNTHETIC_LOOKUP_COLUMNS, with_lookup=True).compile[p]()
    var shape = synthetic_statement(SYNTHETIC_LOOKUP_COLUMNS, with_lookup=True).compile[p]().take_shape()
    var prover = Prover[p, Blake3](ctx, synthetic_statement(SYNTHETIC_LOOKUP_COLUMNS, with_lookup=True).compile[p]().take_shape(), c.families.copy())
    load_trace[p, Blake3](ctx, prover, trace)
    load_advice[p, Blake3](ctx, prover, advice)
    var proof = prover.prove(ctx, List[UInt8]())
    try:
        _ = verify[p, Blake3](proof^, shape, List[UInt8](), c.families)
    except e:
        return String(e)
    return String("")


def test_prove_and_verify_with_lookup() raises:
    """The lookup instance verifies; a record outside the table fails the table constant; advice that
    puts two records in the wrong bins fails it too (the sorted copy is out of table order)."""
    var ctx = DeviceContext()
    assert_equal(_lookup_case(ctx, synthetic_trace[p](1, with_lookup=True), synthetic_advice[p]()), "")
    var bad = synthetic_trace[p](1, with_lookup=True)
    bad[11 * p.N() + 100] = UInt8((Int(bad[11 * p.N() + 100]) + 1) % 127)
    assert_equal(_lookup_case(ctx, bad^, synthetic_advice[p]()), "lookup product is not the table constant")
    var swapped = synthetic_advice[p]()
    var t = swapped[4 * 100]
    swapped[4 * 100] = swapped[4 * 101]
    swapped[4 * 101] = t
    assert_equal(_lookup_case(ctx, synthetic_trace[p](1, with_lookup=True), swapped^), "lookup product is not the table constant")
    var c = synthetic_statement(SYNTHETIC_LOOKUP_COLUMNS, with_lookup=True).compile[p]()
    var short: List[List[UInt8]] = [List[UInt8](length=3, fill=0)]
    var stopped = String("")
    try:
        _ = Shape.__init__[p](SYNTHETIC_LOOKUP_COLUMNS, c.families, c.shape.accs, short)
    except e:
        stopped = String(e)
    assert_equal(stopped, "lookup descriptor needs a table of its record width")
    var noncanonical: List[List[UInt8]] = [[0, 127]]
    stopped = ""
    try:
        _ = Shape.__init__[p](SYNTHETIC_LOOKUP_COLUMNS, c.families, c.shape.accs, noncanonical)
    except e:
        stopped = String(e)
    assert_equal(stopped, "lookup table bytes must be canonical field elements (< 127)")


def _public_case(ctx: DeviceContext, proof: List[UInt8], shape: Shape, families: List[UInt8], public: List[UInt8]) raises -> String:
    var fam = families.copy()
    try:
        _ = verify[p, Blake3](proof.copy(), shape, List[UInt8](), fam, public)
    except e:
        return String(e)
    return String("")


def test_prove_and_verify_with_public_column_and_restriction() raises:
    """One public column (c10 = c0 pub) and one restriction (c1 on the last chain equals its interpolant):
    accept; a changed public coefficient fails the residual identity; a changed restriction polynomial
    fails the restriction check. The restriction adds the point (z1, e2)."""
    var ctx = DeviceContext()
    var c = synthetic_statement(SYNTHETIC_PUBLIC_COLUMNS, with_public=True).compile[p]()
    var trace = synthetic_trace[p](1, with_public=True)
    var block = synthetic_public_block[p]()
    var poly = restriction_line[p](c.layout, 0, chain_values[p](c.layout, trace, 0))
    var shape = synthetic_statement(SYNTHETIC_PUBLIC_COLUMNS, with_public=True).compile[p]().take_shape()
    assert_equal(shape.points, len(shift_points(c.families)) // 4 + 1)
    var prover = Prover[p, Blake3](ctx, synthetic_statement(SYNTHETIC_PUBLIC_COLUMNS, with_public=True).compile[p]().take_shape(), c.families.copy())
    load_trace[p, Blake3](ctx, prover, trace)
    load_public[p, Blake3](ctx, prover, block)
    var proof = prover.prove(ctx, List[UInt8]())
    var public = block.copy()
    public.extend(poly.copy())
    assert_equal(_public_case(ctx, proof, shape, c.families, public), "")
    var bad = public.copy()
    bad[5] = (bad[5] + 1) % 127
    assert_equal(_public_case(ctx, proof, shape, c.families, bad), "residual identity fails at z")
    var bad2 = public.copy()
    bad2[len(block) + 2] = (bad2[len(block) + 2] + 1) % 127
    assert_equal(_public_case(ctx, proof, shape, c.families, bad2), "restriction fails")
    var short = public.copy()
    _ = short.pop()
    assert_equal(_public_case(ctx, proof, shape, c.families, short), "public data has the wrong size")
    var stopped = String("")
    try:
        var bad_pub = List[UInt8](length=4, fill=0)
        bad_pub[0] = 8            # m = 8, d2 = h2 / 4: 8 (h2 / 4 - 1) >= h2
        bad_pub[2] = UInt8(p.h2() // 4)
        _ = Shape.__init__[p](SYNTHETIC_PUBLIC_COLUMNS, c.families, c.shape.accs, List[List[UInt8]](), bad_pub, c.shape.restrictions)
    except e:
        stopped = String(e)
    assert_equal(stopped, "public block does not fit the grid: need m >= 1, d2 >= 1, m (d2 - 1) < h2")


def test_prove_and_verify_with_tail() raises:
    """288 x 128: two committed tail levels (4608 rows on 161280, 576 rows on 4 x 4608), 576 in the clear."""
    comptime big = WIDE
    var ctx = DeviceContext()
    var c = synthetic_statement().compile[big]()
    var shape = synthetic_statement().compile[big]().take_shape()
    assert_equal(len(shape.tail), 2)
    var prover = Prover[big, Blake3](ctx, synthetic_statement().compile[big]().take_shape(), c.families.copy())
    load_trace[big, Blake3](ctx, prover, synthetic_trace[big](1))
    var t0 = perf_counter_ns()
    var proof = prover.prove(ctx, List[UInt8]())
    var t1 = perf_counter_ns()
    assert_true(verify[big, Blake3](proof.copy(), shape, List[UInt8](), c.families))
    var t2 = perf_counter_ns()
    print("proof bytes (tail):", len(proof), " fixed:", shape.fixed_bytes[big, 32](0),
          " prove", (t1 - t0) // 1000000, "ms  verify", (t2 - t1) // 1000000, "ms (host, direct form)")
    # a flipped byte in the first level's sumcheck messages: after its root and the three level-1 multiproofs
    var pos = 8 + 3 * 32 + shape.accumulators() * big.h2() * big.e + 2 * big.h2() * big.e + shape.points * shape.columns() * big.e + 32
    for _ in range(3):
        var n = Int(proof[pos]) | Int(proof[pos + 1]) << 8 | Int(proof[pos + 2]) << 16 | Int(proof[pos + 3]) << 24
        pos += 4 + n
    var bad = proof.copy()
    bad[pos + 3] ^= 1
    var stopped = String("")
    try:
        _ = verify[big, Blake3](bad^, shape, List[UInt8](), c.families)
    except e:
        stopped = String(e)
    assert_equal(stopped, "sumcheck fails at a tail level")



def test_workload_driver_round_trips_every_synthetic_variant() raises:
    """The one path every frontend runs: prove_workload then verify_workload, for the plain, lookup, and
    public variants; a changed public input fails on the verifier's side."""
    var ctx = DeviceContext()
    var plain = Synthetic(SYNTHETIC_COLUMNS, True, False, False, 1)
    assert_true(verify_workload[p, Blake3](prove_workload[p, Blake3](ctx, plain), plain, List[UInt8]()))
    var lookup = Synthetic(SYNTHETIC_LOOKUP_COLUMNS, True, True, False, 1)
    assert_true(verify_workload[p, Blake3](prove_workload[p, Blake3](ctx, lookup), lookup, List[UInt8]()))
    var public = Synthetic(SYNTHETIC_PUBLIC_COLUMNS, True, False, True, 1)
    var proof = prove_workload[p, Blake3](ctx, public)
    var inputs = public.public_inputs[p]()
    assert_true(verify_workload[p, Blake3](proof.copy(), public, inputs))
    var bad = inputs.copy()
    bad[3] = (bad[3] + 1) % 127
    var ok: Bool
    try:
        ok = verify_workload[p, Blake3](proof^, public, bad)
    except e:
        ok = False
    assert_false(ok)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
