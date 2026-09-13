"""The skeleton: the arena plan builds for the reference profile and the prover stops at the first
stage that does not exist, in spec order."""

from std.testing import assert_raises, assert_equal, assert_true, assert_false, TestSuite
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from caracal7.core.field import E_BYTES
from caracal7.core.params import CLIENT, Params, Profile, REGIME_UNIQUE
from caracal7.core.hash import Blake3
from caracal7.proof import Shape, ProofReader, tail_schedule
from caracal7.prover import Prover, ProverLayout, load_trace, load_advice, load_public
from caracal7.verifier import verify
from caracal7.relations import shift_points, standard_chals, POINT, CHAL_MUL, CHAL_ADD, CHAL_ONE, ENTRY, HORNER_TRANSITIONS, ENT_CHAL, ENT_A, ENT_COEF, ENT_BASIS
from caracal7.core.bytes import set_u16
from caracal7.relations.statement import restriction_line, chain_values
from caracal7.workload import prove_workload, verify_workload
from caracal7.workloads.synthetic import Synthetic, SyntheticHorner, SyntheticWiring, horner_statement, horner_trace, wiring_statement, wiring_trace
from caracal7.workloads.synthetic import synthetic_statement, synthetic_trace, synthetic_table, synthetic_advice, synthetic_public_values, SYNTHETIC_COLUMNS, SYNTHETIC_LOOKUP_COLUMNS, SYNTHETIC_PUBLIC_COLUMNS

comptime SPEC95 = Profile(e=E_BYTES, leaf_bytes=1024, tail_digits=3, tail_clear_max=0, lambda_bits=112, grind_bits=20, regime=REGIME_UNIQUE, eta_inv=16, rate_inv=4, tail_rate_inv=32)   # CLIENT before 2026-09-14: the spec 9.5 worked rows
comptime FLAT = Profile(e=E_BYTES, leaf_bytes=1024, tail_digits=3, tail_clear_max=2500, lambda_bits=103, grind_bits=0, regime=REGIME_UNIQUE, eta_inv=16, rate_inv=4, tail_rate_inv=32)   # the reference grid stays clear at level 2: the byte-offset tests below rely on it
comptime p = FLAT.grid(72, 32)


def test_tail_schedule_reference_is_clear_at_level_2() raises:
    var s = tail_schedule[p]()
    assert_equal(len(s), 0)          # N = 2304 <= FLAT's tail_clear_max: y_2 is the clear vector
    var shape = synthetic_statement(53).compile[p]().take_shape()
    assert_equal(shape.clear_length, p.N())
    assert_equal(shape.columns(), 53 + 2 * E_BYTES + 3 * E_BYTES)
    # the spec 9.5 rule (unique regime, tail rate 1/32) folds while binary digits remain: 8 -> two folds -> 36 = 9 x 4 in the clear
    comptime spec = SPEC95.grid(72, 32)
    var f = tail_schedule[spec]()
    assert_equal(len(f), 2)
    assert_equal(f[0].rows, 288)
    assert_equal(f[0].L, 9216)
    assert_equal(f[0].cosets, 2)
    assert_equal(f[1].rows, 36)
    assert_equal(f[1].L, 1152)
    # CLIENT (Johnson regime, tail rate 1/8) keeps the rows and takes the smaller domains
    comptime client = CLIENT.grid(72, 32)
    var g = tail_schedule[client]()
    assert_equal(len(g), 2)
    assert_equal(g[0].L, 2304)
    assert_equal(g[0].cosets, 1)
    assert_equal(g[1].L, 288)
    assert_true(g[0].queries >= 60 and g[0].queries <= 80)   # miss sqrt(1/8) + 1/16 at 92 bits: 73


def test_tail_schedule_folds_a_larger_grid() raises:
    # 288 x 128 (spec 9.5 worked row): N = 36864 -> 4608 rows on 161280 (rate 1/35) -> 576 rows on 18432 = 4 x 4608 (1/32)
    # -> 72 on 2304 -> 9 on 288 -> clear (14 binary digits, four folds, the odd digit 9 plus two binary digits remain)
    comptime big = SPEC95.grid(288, 128)
    var s = tail_schedule[big]()
    assert_equal(len(s), 4)
    assert_equal(s[0].rows, 4608)
    assert_equal(s[0].L, 161280)
    assert_equal(s[0].cosets, 1)
    assert_equal(s[1].rows, 576)
    assert_equal(s[1].L, 18432)
    assert_equal(s[1].cosets, 4)
    assert_equal(s[2].rows, 72)
    assert_equal(s[3].rows, 9)
    assert_true(s[0].queries >= 90 and s[0].queries <= 105)
    var shape = synthetic_statement(357).compile[big]().take_shape()
    assert_equal(shape.clear_length, 9)


def test_tail_schedule_stops_when_binary_digits_run_out() raises:
    # spec 9.5 narrow row, 1008 x 252: 6 binary digits -> two folds -> 3969 in the clear (odd digit only).
    # The tail schedule does not depend on level 1 (63,504 symbols per column: two cosets of 161,280 at the
    # profile's rate 1/4; the spec's 1/32 rule needed the codeword split).
    comptime narrow = SPEC95.grid(1008, 252)
    var s = tail_schedule[narrow]()
    assert_equal(len(s), 2)
    assert_equal(s[0].rows, 31752)
    assert_equal(s[0].L, 645120)          # the largest domain, rate 1/20: the cap
    assert_equal(s[0].cosets, 4)
    assert_equal(s[1].rows, 3969)
    assert_equal(s[1].L, 129024)         # 4 cosets of 32256, rate 1/32.5
    assert_equal(narrow.L(), 322560)
    assert_equal(narrow.m_cosets, 2)
    narrow.check()


def test_layout_plans_the_arena() raises:
    var shape = synthetic_statement(53).compile[p]().take_shape()
    var L = ProverLayout.__init__[p, Blake3](shape)
    assert_true(L.bytes > 0)
    assert_equal(len(L.tail), 0)
    # every offset is inside the arena and 256-aligned
    for off in [L.w.tree, L.z.tree, L.q.tree, L.families, L.accs, L.shifts, L.acc.num, L.acc.den, L.acc.scratch, L.acc.zval, L.acc.chain_prod,
                L.acc.z2, L.acc.n_end, L.acc.d_end, L.sg.lines, L.sg.q3, L.lde.ltmp, L.lde.lde, L.lde.residual, L.lde.quotient, L.open.w_tab,
                L.open.w_z, L.open.openings, L.open.open_partial, L.open.fold_y, L.open.running0, L.query.dom1, L.query.pts, L.query.partial,
                L.query.positions, L.query.stage, L.prefix, L.chal.stage1, L.chal.alpha, L.chal.z, L.chal.beta_gamma, L.chal.batch, L.chal.r]:
        assert_true(off < L.bytes and off % 256 == 0)
    print("arena for 53 + 5 e columns:", L.bytes // (1 << 20), "MiB")


def test_derived_grids() raises:
    """The derivation rounds each axis up to a legal size and picks the level-1 domain by the rate rule; the
    spec's throughput proxy (2016 x 576) needs the codeword split, which check() names."""
    # geometry is regime-independent; the query counts are the unique regime's (SPEC95)
    comptime g = SPEC95.grid(70, 30)
    assert_equal(g.h1(), 72)
    assert_equal(g.h2(), 32)
    assert_equal(g.L(), 2304)                         # 576 symbols per column: one coset of 2304, rate 1/4
    comptime wide = SPEC95.grid(288, 128)
    assert_equal(wide.L(), 40320)                     # 9216 symbols: one coset of 2^7 x 315, rate 0.229
    assert_equal(wide.m_cosets, 1)
    assert_equal(wide.queries(), 131)                # ceil((112 - 20) / log2(2 / 1.229))
    comptime proxy = SPEC95.grid(2016, 576)
    assert_equal(proxy.N(), 1161216)
    var stopped = String("")
    try:
        proxy.check()
    except e:
        stopped = String(e)
    assert_true(stopped.startswith("grid needs the codeword split"))


def test_reader_field_bytes() raises:
    var canonical = List[UInt8]()
    for b in range(127):
        canonical.append(UInt8(b))
    var r = ProofReader(canonical.copy())
    assert_equal(r.field_bytes(127), canonical)
    assert_equal(len(r.field_bytes(0)), 0)
    r.done()
    with assert_raises(contains="truncated"):
        _ = r.field_bytes(1)
    for b in range(127, 256):
        var bytes = List[UInt8](length=16, fill=0)
        bytes[15] = UInt8(b)
        var bad = ProofReader(bytes^)
        with assert_raises(contains="noncanonical field byte"):
            _ = bad.field_bytes(16)
    # Raw length words and payloads are not field elements: length 255, then 255 opaque bytes.
    var raw: List[UInt8] = [255, 0, 0, 0]
    raw.extend(List[UInt8](length=255, fill=255))
    r = ProofReader(raw^)
    assert_equal(r.prefixed(), List[UInt8](length=255, fill=255))
    r.done()


def _reject_noncanonical[p: Params](proof: List[UInt8], shape: Shape, mut families: List[UInt8],
                                    regions: List[Tuple[Int, Int]]) raises:
    """Every direct field region rejects both edge coordinates before algebra/transcript failures."""
    for region in regions:
        for offset in [region[0], region[0] + region[1] - 1]:
            for byte in [127, 128, 255]:
                var bad = proof.copy()
                bad[offset] = UInt8(byte)
                with assert_raises(contains="noncanonical field byte"):
                    _ = verify[p, Blake3](bad^, shape, List[UInt8](), families)


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
    _reject_noncanonical[p](proof, shape, c.families,
        [(z2_bytes, shape.products() * p.h2() * p.e), (q3_bytes, 2 * p.h2() * p.e),
         (openings, shape.points * shape.columns() * p.e), (clear, shape.clear_length * p.e)])


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
    bad_fam[ENT_CHAL] = UInt8(7)                  # entry 0 names element 6
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
    accept; a changed public value fails the residual identity; a changed restriction polynomial
    fails the restriction check. The restriction adds the point (z1, e2)."""
    var ctx = DeviceContext()
    var c = synthetic_statement(SYNTHETIC_PUBLIC_COLUMNS, with_public=True).compile[p]()
    var trace = synthetic_trace[p](1, with_public=True)
    var block = synthetic_public_values[p]()
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
    for offset in [0, len(block), len(public) - 1]:
        var noncanonical = public.copy()
        noncanonical[offset] = 127
        assert_equal(_public_case(ctx, proof, shape, c.families, noncanonical), "noncanonical field byte (expected < 127)")
    var stopped = String("")
    try:
        var bad_pub = List[UInt8](length=2, fill=0)
        bad_pub[0] = 5            # m = 5 does not divide h2
        _ = Shape.__init__[p](SYNTHETIC_PUBLIC_COLUMNS, c.families, c.shape.accs, List[List[UInt8]](), bad_pub, c.shape.restrictions)
    except e:
        stopped = String(e)
    assert_equal(stopped, "public column period divides h2: need m >= 1, h2 % m == 0")


def test_prove_and_verify_with_tail() raises:
    """288 x 128: four committed tail levels (4608 rows on 161280, 576 on 4 x 4608, 72, 9), 9 in the clear."""
    comptime big = SPEC95.grid(288, 128)
    var ctx = DeviceContext()
    var c = synthetic_statement().compile[big]()
    var shape = synthetic_statement().compile[big]().take_shape()
    assert_equal(len(shape.tail), 4)
    var prover = Prover[big, Blake3](ctx, synthetic_statement().compile[big]().take_shape(), c.families.copy())
    load_trace[big, Blake3](ctx, prover, synthetic_trace[big](1))
    var t0 = perf_counter_ns()
    var proof = prover.prove(ctx, List[UInt8]())
    var t1 = perf_counter_ns()
    assert_true(verify[big, Blake3](proof.copy(), shape, List[UInt8](), c.families))
    var t2 = perf_counter_ns()
    print("proof bytes (tail):", len(proof), " fixed:", shape.fixed_bytes[big, 32](0),
          " prove", (t1 - t0) // 1000000, "ms  verify", (t2 - t1) // 1000000, "ms (host, tensor form)")
    # a flipped byte in the first level's sumcheck messages: after its root and the three level-1 multiproofs
    var pos = 8 + 3 * 32 + shape.accumulators() * big.h2() * big.e + 2 * big.h2() * big.e + shape.points * shape.columns() * big.e + 32
    if big.grind_bits > 0:
        pos += 8                                  # the level-1 query seed's nonce
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
    # The first sumcheck is at pos; walk all later roots/frontiers to locate every round and the clear vector.
    var reader = ProofReader(proof.copy())
    reader.pos = pos
    var regions = List[Tuple[Int, Int]]()
    for i in range(len(shape.tail)):
        regions.append((reader.pos, 9 * big.e))
        _ = reader.take(9 * big.e)
        if i + 1 < len(shape.tail):
            _ = reader.take(32)
            if big.grind_bits > 0:
                _ = reader.take(8)
            _ = reader.prefixed()
    regions.append((reader.pos, shape.clear_length * big.e))
    _reject_noncanonical[big](proof, shape, c.families, regions)



def test_horner_descriptor_needs_its_transition_entries() raises:
    """Shape pins the HORNER_TRANSITIONS entries before a Horner ingest range, the ones merge_tables drops:
    a table whose entry there reads another column is rejected at construction."""
    var c = horner_statement().compile[p]()
    var first = Int(c.shape.accs[2]) | Int(c.shape.accs[3]) << 8
    var base = (first - HORNER_TRANSITIONS) * ENTRY
    # (entry offset, byte, new value): col_a of entry 0; coef of entry 2 (the kernel reads only entry 0's weight
    # for every coordinate); chal of entry 3; shift of entry 4; basis2 of entry 5
    for m in [(0, ENT_A, 1), (2, ENT_COEF, 2), (3, ENT_CHAL, 0), (4, ENT_A + 2, 4), (5, ENT_BASIS + 1, 0)]:
        var fam = c.families.copy()
        fam[base + m[0] * ENTRY + m[1]] = UInt8(m[2])
        with assert_raises(contains="transition entries"):
            _ = Shape.__init__[p](c.layout.columns_w(), fam, c.shape.accs, chals=c.shape.chals, ends=c.shape.ends,
                                  pubf=c.shape.pubf, wires=c.shape.wires, sigma=c.shape.sigma)


def test_prove_and_verify_with_horner_accumulators() raises:
    """The second Z kind: three Horner accumulators and a chain-end family, no Z2 in the clear; one wrong
    coefficient fails the small grid."""
    var ctx = DeviceContext()
    var w = SyntheticHorner(1)
    var c = horner_statement().compile[p]()
    assert_equal(c.shape.accumulators(), 3)
    assert_equal(c.shape.products(), 0)
    var proof = prove_workload[p, Blake3](ctx, w)
    assert_true(len(proof) > c.shape.fixed_bytes[p, 32](0), "proof shorter than its fixed part")
    print("horner proof bytes:", len(proof))
    assert_true(verify_workload[p, Blake3](proof^, w, List[UInt8]()))
    var shape = horner_statement().compile[p]().take_shape()
    var prover = Prover[p, Blake3](ctx, horner_statement().compile[p]().take_shape(), c.families.copy())
    var trace = horner_trace[p](1)
    trace[2 * p.N() + p.h1() - 5] = UInt8((Int(trace[2 * p.N() + p.h1() - 5]) + 1) % 127)
    load_trace[p, Blake3](ctx, prover, trace)
    var bad = prover.prove(ctx, List[UInt8]())
    var stopped = String("")
    try:
        _ = verify[p, Blake3](bad^, shape, List[UInt8](), c.families)
    except e:
        stopped = String(e)
    assert_equal(stopped, "small grid identity fails at z2")


def test_prove_and_verify_with_wiring() raises:
    """The copy constraint: one wiring product over two Horner slots and one public factor; then the full
    instance with a (P) product before the wiring lines, a one-slot second product, and cycles across the
    two. A bit moved on one chain keeps every family and fails the joint boundary."""
    var ctx = DeviceContext()
    var w = SyntheticWiring(3, p.h2(), False)
    var c = wiring_statement(p.h2()).compile[p]()
    assert_equal(c.shape.wiring_products(), 1)
    assert_equal(c.shape.products(), 1)
    assert_equal(len(c.shape.sigma), 2 * 2 * p.h2())
    var proof = prove_workload[p, Blake3](ctx, w)
    print("wiring proof bytes:", len(proof))
    assert_true(verify_workload[p, Blake3](proof^, w, w.public_inputs[p]()))
    var wf = SyntheticWiring(3, p.h2(), True)
    var cf = wiring_statement(p.h2(), True).compile[p]()
    assert_equal(cf.shape.wiring_products(), 2)
    assert_equal(cf.shape.products(), 3)
    assert_equal(len(cf.shape.sigma), 2 * 3 * p.h2())
    var pf = prove_workload[p, Blake3](ctx, wf)
    assert_true(verify_workload[p, Blake3](pf^, wf, wf.public_inputs[p]()))
    var shape = wiring_statement(p.h2(), True).compile[p]().take_shape()
    var prover = Prover[p, Blake3](ctx, wiring_statement(p.h2(), True).compile[p]().take_shape(), cf.families.copy())
    var trace = wiring_trace[p](3, True)
    trace[2 * p.N() + 3 * p.h1() + p.h1() - 2] ^= 1             # c on chain 3: no family reads it but its fingerprint
    load_trace[p, Blake3](ctx, prover, trace)
    var bad = prover.prove(ctx, wf.public_inputs[p]())
    var stopped = String("")
    try:
        _ = verify[p, Blake3](bad^, shape, wf.public_inputs[p](), cf.families, SyntheticWiring.public_data[p](cf.layout, wf.public_inputs[p]()))
    except e:
        stopped = String(e)
    assert_equal(stopped, "wiring grand product is not the public factor")


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
