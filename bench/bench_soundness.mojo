"""Conditional interactive soundness ledger for the shipped CSP workloads; see docs/soundness.md.

Run with `uv run mojo run --Werror -I src bench/bench_soundness.mojo`.
No GPU work, proofs, or cached shape counts. Exit zero means the calculation ran, not certification.
"""

from std.math import log2, max, abs, sqrt, ceil
from std.testing import assert_equal, assert_true, assert_raises

from core.field import E_BYTES
from core.params import CLIENT, Params, Profile, REGIME_UNIQUE, REGIME_CAPACITY, REGIME_JOHNSON, query_count, miss_probability
from core.bytes import get_u16
from relations.ir import ENTRY, ACC, END, WIRE, PUBF, RES, ZERO, NONE, KIND_HORNER, CHAL, CHAL_ADD, CHAL_MUL, CHAL_ONE, entry, acc_kind, acc_z_col
from relations.statement import Compiled, Statement, Term, BIT
from workload import Workload
from proof import Shape
from workloads.sha256 import Sha256
from workloads.keccak import Keccak
from workloads.poseidon import Poseidon
from workloads.ecdsa import Ecdsa, Point
from workloads.bigint import Big


def challenge_degrees(table: List[UInt8]) raises -> List[Int]:
    """Total degree in the three independent stage-1 challenges. Addition can only lower this bound."""
    var degrees: List[Int] = [1, 1, 1]
    if len(table) % CHAL != 0:
        raise Error("incomplete challenge descriptor")
    for i in range(0, len(table), CHAL):
        var a = Int(table[i + 1])
        var b = Int(table[i + 2])
        if (a != CHAL_ONE and a >= len(degrees)) or (b != CHAL_ONE and b >= len(degrees)):
            raise Error("challenge degree needs an earlier element")
        var da = 0 if a == CHAL_ONE else degrees[a]
        var db = 0 if b == CHAL_ONE else degrees[b]
        if table[i] == CHAL_ADD:
            degrees.append(max(da, db))
        elif table[i] == CHAL_MUL:
            degrees.append(da + db)
        else:
            raise Error("unsupported challenge operation")
    return degrees^


def weight_degree(degrees: List[Int], encoded: Int) -> Int:
    return 0 if encoded == 0 else degrees[encoded - 1]


def horner_degree(rows: Int, scale: Int, ingest: Int, start: Int) -> Int:
    # rows - 1 transitions; the last row is never ingested. Includes a nonzero initial value.
    return max((rows - 2) * scale + ingest, (rows - 1) * scale if start != 0 else 0)


def query_error(length: Int, dimension: Int, queries: Int, regime: Int = REGIME_UNIQUE, eta_inv: Int = 16) raises -> Float64:
    """The miss probability of `queries` draws at the regime's per-query miss (params.miss_probability)."""
    if dimension <= 0 or dimension >= length or queries <= 0:
        raise Error("query bound needs 0 < dimension < length and positive queries")
    var miss = miss_probability(Float64(dimension) / Float64(length), regime, eta_inv)
    var error = 1.0
    for _ in range(queries):
        error *= miss
    return error


def gap_numerator(length: Int, dimension: Int, regime: Int, eta_inv: Int, section4: Bool = False) raises -> Int:
    """The correlated-agreement error numerator of one code (error = numerator / |E|): `length` at the
    unique-decoding radius (BCIKS20 1.2 / 1.7); at the Johnson radius BCHKS25 Theorem 1.5, radius
    1 - sqrt(rho) - eta with m = max(ceil(sqrt(rho) / (2 eta)), 3):
    (2 (m + 1/2)^5 + 3 (m + 1/2) gamma rho) / (3 rho^1.5) * n + (m + 1/2) / sqrt(rho), rounded up.
    `section4` uses the m of Theorems 4.2 (curves) and 4.6 (mutual), max(ceil(sqrt(rho) / eta), 3);
    those theorems also carry a factor M (words minus one), which the caller multiplies (soundness.md J1, J2).
    The capacity conjecture has no proven numerator; the ledger keeps `length` there as a placeholder."""
    if regime != REGIME_JOHNSON:
        return length
    var rho = Float64(dimension) / Float64(length)
    var eta = 1.0 / Float64(eta_inv)
    var gamma = 1.0 - sqrt(rho) - eta
    if gamma <= 0.0:
        raise Error("Johnson radius needs sqrt(rate) + eta < 1")
    var m = max(ceil(sqrt(rho) / (2.0 * eta)), 3.0) + 0.5
    if section4:
        m = max(ceil(sqrt(rho) / eta), 3.0) + 0.5
    var a = (2.0 * m * m * m * m * m + 3.0 * m * gamma * rho) / (3.0 * rho * sqrt(rho)) * Float64(length) + m / sqrt(rho)
    return Int(ceil(a))


def field_order(e: Int) -> Float64:
    var order = 1.0
    for _ in range(e):
        order *= 127.0
    return order


def bits(error: Float64) -> Float64:
    # Display rounds down, so rounding cannot turn a failing budget into a passing one.
    return Float64(Int(-log2(error) * 100.0)) / 100.0


def ledger[p: Params](c: Compiled) raises -> Tuple[List[Tuple[String, Int]], Float64]:
    """Each integer is a numerator over |E|; the Float64 is the sum of query errors.
    This is conditional on the proof obligations in the note, not a generic IR soundness theorem.
    """
    p.check()
    if p.n_cw() != 1 or p.tail_digits != 3:
        raise Error("ledger covers one codeword per column and three-digit tail folds")
    ref s = c.shape
    var degrees = challenge_degrees(s.chals)
    var horner = Dict[Int, Int]()       # first coordinate column -> endpoint challenge degree
    var horner_families = List[Int]()
    for i in range(s.accumulators()):
        if acc_kind(s.accs, i) != KIND_HORNER:
            raise Error("Herder lookup/permutation budgets are not implemented; none of the four CSP workloads uses them")
        var first = get_u16(s.accs, i * ACC + 2)
        var count = get_u16(s.accs, i * ACC + 4)
        var ingest = 0
        for j in range(first, first + count):
            ingest = max(ingest, weight_degree(degrees, entry(c.families, j).chal))
        horner[acc_z_col(s.accs, i)] = horner_degree(p.h1(), weight_degree(degrees, Int(s.accs[i * ACC + 7])), ingest, Int(s.accs[i * ACC + 6]))
        horner_families.append(s.family_of(i))

    var grid_alpha = 0
    for i in range(s.entries):
        var en = entry(c.families, i)
        grid_alpha = max(grid_alpha, en.family)
        if en.chal != 0 and en.family not in horner_families:
            raise Error("challenge-dependent ordinary family needs its own identity budget")
    var small_alpha = 0
    var endpoint_degree = 0
    for i in range(len(s.ends) // END):
        small_alpha = max(small_alpha, get_u16(s.ends, i * END + 4))
        var degree = horner[get_u16(s.ends, i * END)] + weight_degree(degrees, Int(s.ends[i * END + 7]))
        var b = get_u16(s.ends, i * END + 2)
        if b != NONE:
            degree += horner[b]
        endpoint_degree = max(endpoint_degree, degree)
    var wire_degree = 0
    var factors = len(s.pubf) // PUBF
    for i in range(s.wiring_products()):
        small_alpha = max(small_alpha, get_u16(s.wires, i * WIRE + 4))
        for side in range(2):
            var col = get_u16(s.wires, i * WIRE + 2 * side)
            if col != NONE:
                factors += p.h2()
                wire_degree = max(wire_degree, horner[col])
    for i in range(len(s.pubf) // PUBF):
        wire_degree = max(wire_degree, horner[acc_z_col(s.accs, get_u16(s.pubf, i * PUBF))])
    var restrictions = 0
    for i in range(len(s.restrictions) // RES):
        restrictions += max(p.h1(), get_u16(s.restrictions, i * RES + 4)) - 1

    var gap = gap_numerator(p.L(), p.K(), p.regime, p.eta_inv)      # level 1: uniform E^columns fold, block alphabet
    var batch = 0
    var queries = query_error(p.L(), p.K(), p.queries(), p.regime, p.eta_inv)
    var previous_queries = p.queries()
    for i in range(len(s.tail)):
        var level = s.tail[i]
        gap += 3 * gap_numerator(level.L, level.rows, p.regime, p.eta_inv)   # later folds: tensor randomness in three E elements
        batch += (4 if i == 0 else 1) * previous_queries + 1
        queries += query_error(level.L, level.rows, level.queries, p.regime, p.eta_inv)
        previous_queries = level.queries
    var terms: List[Tuple[String, Int]] = [
        ("alpha_batch", grid_alpha + small_alpha),
        ("grid_identity", 2 * p.h1() + 2 * p.h2()),
        ("small_grid", 3 * p.h2() if s.accumulators() > 0 else 0),
        ("starts_and_zero_rows", (s.accumulators() + len(s.zeros) // ZERO) * (p.h2() - 1)),
        ("restrictions", restrictions),
        ("horner_identities", endpoint_degree),
        ("wire_fingerprints", wire_degree),
        ("wire_product", factors),
        ("wire_zero_factors", 2 * factors),
        ("pcs_gap", gap),
        ("pcs_batch", batch),
        ("sumcheck", 6 * len(s.tail)),
        ("opening_batch", 2),
    ]
    return (terms^, queries)


def projection[p: Params](ref s: Shape, name: String, regime: Int, numerator_no_gap: Int, section4: Bool = False) raises:
    """`section4` charges BCHKS25 Theorems 4.2 / 4.6 as written: the doubled m, and the factor M = words - 1
    per code (level 1 batches the columns, `_level1_symbol`; a tail challenge folds a pair, M = 1)."""
    var per_level = p.lambda_bits - p.grind_bits
    var q1 = query_count(per_level, p.rate(), regime, p.eta_inv)
    var queries = String(q1)
    var error = query_error(p.L(), p.K(), q1, regime, p.eta_inv)
    var m1 = (s.columns() - 1) if section4 else 1
    var gap = m1 * gap_numerator(p.L(), p.K(), regime, p.eta_inv, section4)
    for i in range(len(s.tail)):
        var q = query_count(per_level, Float64(s.tail[i].rows) / Float64(s.tail[i].L), regime, p.eta_inv)
        queries += "/" + String(q)
        error += query_error(s.tail[i].L, s.tail[i].rows, q, regime, p.eta_inv)
        gap += 3 * gap_numerator(s.tail[i].L, s.tail[i].rows, regime, p.eta_inv, section4)
    var q_err = error / Float64(1 << p.grind_bits)
    print(name, "eta_inv", p.eta_inv, "queries_per_level", queries, "query_bits", bits(q_err), "pcs_gap", gap,
          "conditional_iop_bits", bits(Float64(numerator_no_gap + gap) / field_order(p.e) + q_err))


def report[p: Params, W: Workload](target: String, size: Int, w: W) raises:
    var c = w.statement[p]().compile[p]()
    ref s = c.shape
    var result = ledger[p](c)
    var numerator = 0
    for term in result[0]:
        numerator += term[1]
    print("\ncase", target, size, "grid", p.h1(), p.h2(), "e", p.e, "lambda_queries", p.lambda_bits, "grind_bits", p.grind_bits, "regime", p.regime)
    print("columns W/Z/Q/public", s.columns_w, s.columns_z, s.columns_q, s.columns_p,
          "points", s.points, "entries", s.entries, "horner", s.accumulators(), "wiring_products", s.wiring_products())
    print("level 1: dimension/length/queries", p.K(), p.L(), p.queries())
    for i in range(len(s.tail)):
        print("level", i + 2, "dimension/length/queries", s.tail[i].rows, s.tail[i].L, s.tail[i].queries)
    var numerator_no_gap = 0
    for term in result[0]:
        print("field_numerator", term[0], term[1])
        if term[0] != "pcs_gap":
            numerator_no_gap += term[1]
    # projections of the other regimes at the same geometry: the queries each level would need at the same
    # per-level target, the query error at those queries, and (Johnson only) the pcs_gap numerator of
    # BCHKS25 1.5 in place of the unique one, so the last number is the ledger the switch would compile
    projection[p](s, "capacity_conjecture", REGIME_CAPACITY, numerator_no_gap)
    projection[p](s, "johnson_bchks25_1.5", REGIME_JOHNSON, numerator_no_gap)
    projection[p](s, "johnson_bchks25_4.2_4.6", REGIME_JOHNSON, numerator_no_gap, section4=True)
    var q_err = result[1] / Float64(1 << p.grind_bits)     # per 2^grind_bits hashes of prover work per level
    print("query_error_per_attempt", result[1], "grind_bits", p.grind_bits, "query_error", q_err, "query_bits", bits(q_err), "field_numerator_total", numerator)
    print("conditional_iop_bits", bits(Float64(numerator) / field_order(p.e) + q_err))
    print("projected_e16_same_geometry_and_queries", bits(Float64(numerator) / field_order(16) + q_err))
    print("projected_e20_same_geometry_and_queries", bits(Float64(numerator) / field_order(20) + q_err))


def self_check() raises:
    assert_true(abs(query_error(16, 4, 3) - 125.0 / 512.0) < 1e-15)
    assert_equal(field_order(2), 16129.0)
    assert_equal(bits(1.0 / 256.0 + 1.0 / 256.0), 7.0)  # add errors, not bit counts
    assert_equal(horner_degree(4, 2, 1, 0), 5)  # I0*s^2 + I1*s + I2
    assert_equal(horner_degree(4, 2, 1, 1), 6)  # plus the nonzero start*s^3
    var d = challenge_degrees([CHAL_ADD, 0, CHAL_ONE, CHAL_MUL, 3, 1, CHAL_MUL, 4, 2])
    assert_equal(d[3], 1)
    assert_equal(d[4], 2)
    assert_equal(d[5], 3)
    with assert_raises():
        _ = challenge_degrees([CHAL_MUL, 3, 0])
    with assert_raises():
        _ = query_error(16, 16, 3)
    assert_true(query_error(16, 4, 4) < query_error(16, 4, 3))
    assert_true(query_error(16, 8, 3) > query_error(16, 4, 3))
    assert_true(field_order(20) > field_order(16))
    # A compiled two-family 4x4 statement, once clear and once with one committed tail.
    # Hand totals catch omitted final queries, the four-coordinate first batch, and gap accounting.
    comptime for i in range(2):
        comptime p = Params(e=E_BYTES, a1=2, m1=1, a2=2, m2=1, L0=48, m_cosets=1, grind_bits=0, regime=REGIME_UNIQUE, eta_inv=16, tail_rate_inv=32,
                            leaf_bytes=1024, tail_digits=3, tail_clear_max=100 if i == 0 else 0, lambda_bits=3, codewords=1)
        var st = Statement()
        st.col("x", BIT)
        st.family("zero", [Term(1, st.read("x"))])
        var c = st.compile[p]()
        var result = ledger[p](c)
        var numerator = 0
        for term in result[0]:
            numerator += term[1]
        # Flat: 1 alpha + 16 grid + 48 gap + 2 openings. Tail: +3*70 gap +17 batch +6 sumcheck.
        assert_equal(numerator, 67 if i == 0 else 300)
        var miss = 13.0 / 24.0
        var expected = miss * miss * miss * miss
        if i == 1:
            assert_equal(len(c.shape.tail), 1)
            assert_equal(c.shape.tail[0].L, 70)
            miss = 18.0 / 35.0
            expected += miss * miss * miss * miss
        assert_true(abs(result[1] - expected) < 1e-15)


def johnson(tail_rate_inv: Int) -> Profile:
    var q = CLIENT
    q.regime = REGIME_JOHNSON
    q.tail_rate_inv = tail_rate_inv
    return q


def main() raises:
    self_check()
    print("CONDITIONAL INTERACTIVE LEDGER -- NOT A SECURITY CERTIFICATION. docs/soundness.md lists open obligations.")
    print("the e16/e20 projections change only the field order; the compiled e is E_BYTES (field.mojo); no Fiat-Shamir/hash/quantum bound included.")
    # ponytail: the case whitelist mirrors cli/ffi.mojo; update both when benchmark routing changes.
    # Only the routing is repeated: Params, tail_schedule, statements, and all counts come from production code.
    comptime for i in range(5):
        comptime n = 128 << i
        report[CLIENT.grid(32, 64 * ((n + 9 + 63) // 64) + 1)]("sha256", n, Sha256(List[UInt8](length=n, fill=0)))
        report[CLIENT.grid(64, 24 * (n // 136 + 1))]("keccak", n, Keccak(List[UInt8](length=n, fill=0)))
    for n in [2, 4, 8]:
        report[CLIENT.grid(64, 368)]("poseidon", n, Poseidon(List[Int](length=n, fill=0)))
    for n in [12, 16]:
        report[CLIENT.grid(64, 720)]("poseidon", n, Poseidon(List[Int](length=n, fill=0)))
    # statement() uses Ecdsa.circuit(), independent of signature values; no live walk or witness is needed.
    report[CLIENT.grid(144, 576)]("ecdsa", 32, Ecdsa(Big(), Big(), Big(), Point.identity()))
    # tail rate sweep in the Johnson regime: the rho^-1.5 constant of BCHKS25 1.5 punishes low-rate tails
    comptime for r in [4, 8, 16, 32]:
        report[johnson(r).grid(144, 576)]("ecdsa_johnson_tail_rate_inv_" + String(r), 32, Ecdsa(Big(), Big(), Big(), Point.identity()))
    print("\nSTATUS: unresolved proof/implementation obligations; no certified security_bits emitted.")
