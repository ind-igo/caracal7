"""Conditional soundness comparisons for covered workload profiles; see docs/soundness.md.

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
from workloads.rsa import rsa_statement
from workloads.sod import sod_statement
from workloads.dsc import dsc_statement
from workloads.passport import passport_statement
from workloads.mrz import WINDOW_OFFSET
import bench_sod as SodFixture
import bench_dsc as DscFixture


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


def gap_numerator(length: Int, dimension: Int, regime: Int, eta_inv: Int, section4: Bool = False, pairs: Bool = False) raises -> Int:
    """Hab25 Theorem 2, p. 4: (m+1/2)^7 n^2 / (3 rho^1.5), rho=(dimension-1)/n.
    The target radius uses the actual rate dimension/n; the theorem uses degree/n.
    section4 and pairs retain the old BCHKS25 4.2/4.6 and 1.5 projections, respectively.
    The capacity conjecture has no proven numerator; length is only a placeholder there.
    """
    if regime != REGIME_JOHNSON:
        return length
    if dimension <= 1 or dimension >= length or eta_inv < 2:
        raise Error("Johnson gap needs 1 < dimension < length and eta_inv >= 2")
    if not section4 and not pairs:
        var rate = Float64(dimension) / Float64(length)
        var rho_h = Float64(dimension - 1) / Float64(length)
        var gamma_h = 1.0 - sqrt(rate) - 1.0 / Float64(eta_inv)
        if gamma_h <= 0.0:
            raise Error("Johnson radius needs sqrt(rate) + eta < 1")
        var eta_h = 1.0 - sqrt(rho_h) - gamma_h
        var mh = max(ceil(sqrt(rho_h) / (2.0 * eta_h)), 3.0) + 0.5
        var ah = mh * mh * mh * mh * mh * mh * mh * Float64(length) * Float64(length) / (3.0 * rho_h * sqrt(rho_h))
        # Diagnostic Float64 sizing: pad upwards before the integer conversion.
        ah = ceil(ah * (1.0 + 1e-12))
        if ah >= 9.0e18:
            raise Error("Johnson numerator exceeds the diagnostic Int budget")
        return Int(ah)
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


def add_numerator(total: Int, term: Int, factor: Int = 1) raises -> Int:
    if total < 0 or term < 0 or factor < 0 or (factor > 0 and term > (9223372036854775807 - total) // factor):
        raise Error("field numerator exceeds the diagnostic Int budget")
    return total + factor * term


def ceil_ratio(numerator: Int, denominator: Int) raises -> Int:
    if numerator < 0 or denominator <= 0:
        raise Error("ceiling needs a nonnegative numerator and positive denominator")
    return numerator // denominator + (1 if numerator % denominator != 0 else 0)


def ceil_sqrt(value: Int) raises -> Int:
    if value <= 0:
        raise Error("square root needs a positive integer")
    var low = 1
    var high = value
    while low < high:
        var mid = low + (high - low) // 2
        if mid >= ceil_ratio(value, mid):
            high = mid
        else:
            low = mid + 1
    return low


def agreement_threshold(length: Int, dimension: Int, eta_inv: Int) raises -> Int:
    """Exact ceil(sqrt(N*K) + N/eta_inv), shared by scalar and list bounds."""
    if dimension <= 0 or dimension >= length or eta_inv < 2:
        raise Error("agreement needs 0 < dimension < length and eta_inv >= 2")
    var nk = add_numerator(0, length, dimension)
    var scaled = add_numerator(0, add_numerator(0, nk, eta_inv), eta_inv)
    var agreement = ceil_ratio(add_numerator(length, ceil_sqrt(scaled)), eta_inv)
    if agreement > length:
        raise Error("agreement exceeds the code length")
    return agreement


def joint_list_bound(length: Int, dimension: Int, eta_inv: Int) raises -> Int:
    """Checked Johnson block-list ratio in caracal7-fv/rsblock.bend; own proof, needs review.
    A sufficient integer cap, not a claim that this many close codewords exist.
    """
    var agreement = agreement_threshold(length, dimension, eta_inv)
    var degree = dimension - 1
    var margin = add_numerator(0, agreement, agreement) - add_numerator(0, length, degree)
    if agreement <= degree or margin <= 0:
        raise Error("joint list bound needs a positive Johnson margin")
    return ceil_ratio(add_numerator(0, length, agreement - degree), margin)


def dkt26_numerator(length: Int, dimension: Int, eta_inv: Int) raises -> Int:
    """DKT26 Theorem 5.12, p. 53; Lemma 5.3, pp. 40-43, ell=1, L=D+1.
    Exact integer ceilings. Reject intermediate Int overflow, including parameter sizing.
    """
    if dimension <= 1:
        raise Error("DKT26 needs dimension > 1")
    var degree = dimension - 1
    var agreement = agreement_threshold(length, dimension, eta_inv)
    var dn = add_numerator(0, degree, length)
    var twice_t = 7  # 2*m+1, starting at m=3
    while True:
        # m >= sqrt(D/n)/(2*(A/n-sqrt(D/n))), checked without square roots.
        var lhs = add_numerator(0, agreement, twice_t - 1)
        lhs = add_numerator(0, lhs, lhs)
        var rhs = add_numerator(0, add_numerator(0, dn, twice_t), twice_t)
        if lhs >= rhs:
            break
        twice_t = add_numerator(twice_t, 2)
    var support = add_numerator(0, add_numerator(0, length, twice_t), twice_t)
    var b = ceil_sqrt(ceil_ratio(support, 4 * degree)) - 1
    var h = ceil_ratio(support, 12 * degree) - 1
    var psi = add_numerator(1, 2 * degree - 1, 2 * b - 1)
    psi = add_numerator(psi, max(0, b - 2 * degree - 1), 2)
    var joint = add_numerator(b, h, psi)
    var incidence = ceil_ratio(add_numerator(0, length - degree, joint), agreement - degree)
    var exceptional = add_numerator(incidence, 2 * b - 1, h)
    return add_numerator(exceptional, length - degree - 1, b)


def dkt26_gap[p: Params](ref s: Shape) raises -> Int:
    if p.regime != REGIME_JOHNSON or p.n_cw() != 1 or p.tail_digits != 3:
        raise Error("DKT26 comparison needs the supported Johnson profile")
    var scalar = dkt26_numerator(p.L(), p.K(), p.eta_inv)
    # Corollary 7.2, p. 66: E/(Q-1) <= (ceil(E)+1)/Q when ceil(E)<Q.
    # For e>=10, Q exceeds the entire Int range. Smaller orders fit exactly.
    if p.e < 10:
        var order = 1
        for _ in range(p.e):
            order *= 127
        if scalar >= order:
            raise Error("DKT26 affine numerator must be smaller than the challenge field")
    var gap = add_numerator(4, scalar, 4)  # retain the four conjugate-code allowance
    for level in s.tail:
        if level.codewords != 1:
            raise Error("DKT26 comparison does not cover split tail codewords")
        # Corollary 7.8, pp. 69-70: three shared-challenge binary fold levels.
        gap = add_numerator(gap, dkt26_numerator(level.L, level.rows, p.eta_inv), 3)
    return gap


def list_bound(length: Int, dimension: Int, eta_inv: Int, joint: Bool = False) raises -> Int:
    if joint:
        return joint_list_bound(length, dimension, eta_inv)
    if dimension <= 0 or dimension >= length or eta_inv < 2:
        raise Error("list bound needs 0 < dimension < length and eta_inv >= 2")
    return Int(ceil(Float64(eta_inv) / (2.0 * sqrt(Float64(dimension) / Float64(length)))))


def tail_list_bound(ref s: Shape, i: Int, regime: Int, eta_inv: Int, joint: Bool = False) raises -> Int:
    # The next root fixes this list before the current queries. The final clear message is one candidate.
    if regime != REGIME_JOHNSON or i == len(s.tail):
        return 1
    return list_bound(s.tail[i].L, s.tail[i].rows, eta_inv, joint)


def bind_numerator(columns: Int, h1: Int, h2: Int, size: Int) raises -> Int:
    # Unordered candidate pairs; each polynomial difference has total degree <= h1+h2-2.
    var pairs = add_numerator(0, size, size - 1) // 2
    return add_numerator(0, add_numerator(0, columns, pairs), h1 + h2 - 2)


def field_order(e: Int) -> Float64:
    var order = 1.0
    for _ in range(e):
        order *= 127.0
    return order


def bits(error: Float64) -> Float64:
    # Display rounds down, so rounding cannot turn a failing budget into a passing one.
    return Float64(Int(-log2(error) * 100.0)) / 100.0


def ledger[p: Params](c: Compiled, joint_lists: Bool = False) raises -> Tuple[List[Tuple[String, Int]], Float64]:
    """Each integer is a numerator over |E|; the Float64 is the sum of query errors.
    This is conditional on the proof obligations in the note, not a generic IR soundness theorem.
    """
    p.check()
    if joint_lists and p.regime != REGIME_JOHNSON:
        raise Error("joint-list comparison requires the Johnson regime")
    if p.n_cw() != 1 or p.tail_digits != 3:
        raise Error("ledger covers one codeword per column and three-digit tail folds")
    ref s = c.shape
    for level in s.tail:
        if level.codewords != 1:
            raise Error("ledger does not cover split tail codewords")
    var degrees = challenge_degrees(s.chals)
    var horner = Dict[Int, Int]()       # first coordinate column -> endpoint challenge degree
    var horner_families = List[Int]()
    for i in range(s.accumulators()):
        if acc_kind(s.accs, i) != KIND_HORNER:
            raise Error("Herder lookup/permutation budgets are not implemented; this ledger requires Horner accumulators")
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
    var size = 1
    if p.regime == REGIME_JOHNSON:
        size = list_bound(p.L(), p.K(), p.eta_inv, joint_lists)
        gap = add_numerator(4, gap, 4)    # four conjugate codes, affine MCA <= (ceil(a)+1)/Q each
    var batch = 0
    var sumcheck = 0
    var queries = Float64(tail_list_bound(s, 0, p.regime, p.eta_inv, joint_lists)) * query_error(p.L(), p.K(), p.queries(), p.regime, p.eta_inv)
    var previous_queries = p.queries()
    for i in range(len(s.tail)):
        var level = s.tail[i]
        gap = add_numerator(gap, gap_numerator(level.L, level.rows, p.regime, p.eta_inv), 3)   # later folds: tensor randomness in three E elements
        var candidates = tail_list_bound(s, i, p.regime, p.eta_inv, joint_lists)
        batch = add_numerator(batch, (4 if i == 0 else 1) * previous_queries + 1, candidates)
        sumcheck = add_numerator(sumcheck, 6, candidates)
        queries += Float64(tail_list_bound(s, i + 1, p.regime, p.eta_inv, joint_lists)) * query_error(level.L, level.rows, level.queries, p.regime, p.eta_inv)
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
    ]
    var relations = 0
    for term in terms:
        relations = add_numerator(relations, term[1])
    terms.extend([
        ("list_relations", add_numerator(0, relations, size - 1)),
        ("pcs_gap", gap),
        ("pcs_batch", batch),
        ("sumcheck", sumcheck),
        ("opening_batch", 2 * size),
        ("list_bind", bind_numerator(s.opened(), p.h1(), p.h2(), size)),
    ])
    return (terms^, queries)


def projection[p: Params](ref s: Shape, name: String, regime: Int, numerator_no_gap: Int, section4: Bool = False, pairs: Bool = False) raises:
    """`pairs` retains the old 1.5 charge. `section4` retains the old 4.2/4.6 projection: the doubled m, and the factor M = opened - 1
    per code (level 1 batches the columns, `_level1_symbol`; a tail challenge folds a pair, M = 1)."""
    var per_level = p.lambda_bits - p.grind_bits
    var q1 = query_count(per_level, p.rate(), regime, p.eta_inv)
    var queries = String(q1)
    var error = Float64(tail_list_bound(s, 0, regime, p.eta_inv)) * query_error(p.L(), p.K(), q1, regime, p.eta_inv)
    var m1 = (s.opened() - 1) if section4 else 1
    var gap = add_numerator(0, gap_numerator(p.L(), p.K(), regime, p.eta_inv, section4, pairs), m1)
    for i in range(len(s.tail)):
        var q = query_count(per_level, Float64(s.tail[i].rows) / Float64(s.tail[i].L), regime, p.eta_inv)
        queries += "/" + String(q)
        error += Float64(tail_list_bound(s, i + 1, regime, p.eta_inv)) * query_error(s.tail[i].L, s.tail[i].rows, q, regime, p.eta_inv)
        gap = add_numerator(gap, gap_numerator(s.tail[i].L, s.tail[i].rows, regime, p.eta_inv, section4, pairs), 3)
    var q_err = error / Float64(1 << p.grind_bits)
    print(name, "eta_inv", p.eta_inv, "queries_per_level", queries, "query_bits", bits(q_err), "pcs_gap", gap,
          "conditional_work_bits", bits(Float64(add_numerator(numerator_no_gap, gap)) / field_order(p.e) + q_err))


def report[p: Params, W: Workload](target: String, size: Int, w: W) raises:
    var c = w.statement[p]().compile[p]()
    report_compiled[p](target, size, c)


def numeric_code(length: Int, dimension: Int, queries: Int, eta_inv: Int) raises:
    print("numeric_code D/N/T/C/oldL/jointL/queries", dimension - 1, length,
          agreement_threshold(length, dimension, eta_inv), dkt26_numerator(length, dimension, eta_inv),
          list_bound(length, dimension, eta_inv), joint_list_bound(length, dimension, eta_inv), queries)


def report_compiled[p: Params](target: String, size: Int, c: Compiled) raises:
    ref s = c.shape
    var result = ledger[p](c)
    var numerator = 0
    for term in result[0]:
        numerator = add_numerator(numerator, term[1])
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
            numerator_no_gap = add_numerator(numerator_no_gap, term[1])
    # Historical gap projections retain their old scalar charges; the new non-gap list terms remain included.
    projection[p](s, "capacity_conjecture", REGIME_CAPACITY, numerator_no_gap)
    projection[p](s, "johnson_bchks25_1.5_projection", REGIME_JOHNSON, numerator_no_gap, pairs=True)
    projection[p](s, "johnson_bchks25_4.2_4.6_projection", REGIME_JOHNSON, numerator_no_gap, section4=True)
    # This division is a work-normalized diagnostic, not an interactive probability bound.
    var q_err = result[1] / Float64(1 << p.grind_bits)
    if p.regime == REGIME_JOHNSON:
        var dkt_gap = dkt26_gap[p](s)
        print("johnson_dkt26_5.12_conditional", "eta_inv", p.eta_inv, "pcs_gap", dkt_gap,
              "field_numerator_total", add_numerator(numerator_no_gap, dkt_gap),
              "query_bits", bits(q_err), "conditional_work_bits",
              bits(Float64(add_numerator(numerator_no_gap, dkt_gap)) / field_order(p.e) + q_err),
              "conditional_interactive_bits", bits(Float64(add_numerator(numerator_no_gap, dkt_gap)) / field_order(p.e) + result[1]))
        var joint = ledger[p](c, joint_lists=True)
        var joint_total = dkt_gap
        for term in joint[0]:
            if term[0] != "pcs_gap":
                joint_total = add_numerator(joint_total, term[1])
        var joint_field = Float64(joint_total) / field_order(p.e)
        print("johnson_dkt26_joint_list_conditional", "pcs_gap", dkt_gap,
              "field_numerator_total", joint_total, "query_error_per_attempt", joint[1],
              "conditional_work_bits", bits(joint_field + joint[1] / Float64(1 << p.grind_bits)),
              "conditional_interactive_bits", bits(joint_field + joint[1]))
        numeric_code(p.L(), p.K(), p.queries(), p.eta_inv)
        for level in s.tail:
            numeric_code(level.L, level.rows, level.queries, p.eta_inv)
    print("query_error_per_attempt", result[1], "grind_bits", p.grind_bits, "query_error", q_err, "query_bits", bits(q_err), "field_numerator_total", numerator)
    print("conditional_work_bits", bits(Float64(numerator) / field_order(p.e) + q_err),
          "conditional_interactive_bits", bits(Float64(numerator) / field_order(p.e) + result[1]))
    print("projected_e16_same_geometry_and_queries", bits(Float64(numerator) / field_order(16) + q_err))
    print("projected_e20_same_geometry_and_queries", bits(Float64(numerator) / field_order(20) + q_err))


def self_check() raises:
    assert_equal(agreement_threshold(64, 16, 16), 36)
    assert_equal(joint_list_bound(64, 16, 16), 4)
    for pair in [(161280, 20736), (92160, 10368), (10752, 1296), (1344, 162)]:
        assert_equal(joint_list_bound(pair[0], pair[1], 16), 7)
    with assert_raises():
        _ = joint_list_bound(16, 16, 16)
    with assert_raises():
        _ = joint_list_bound(16, 1, 1)
    with assert_raises():
        _ = joint_list_bound(1 << 40, 1 << 30, 16)
    # Independent rational calculation of all four ECDSA scalar bounds.
    assert_equal(dkt26_numerator(161280, 20736, 16), 66374040)
    assert_equal(dkt26_numerator(92160, 10368, 16), 44914431)
    assert_equal(dkt26_numerator(10752, 1296, 16), 5031516)
    assert_equal(dkt26_numerator(1344, 162, 16), 641610)
    assert_equal(dkt26_numerator(64, 17, 16), 28478)  # m=4, integral scalar bound
    assert_equal(ceil_sqrt(9223372036854775807), 3037000500)
    assert_equal(ceil_sqrt(64), 8)
    assert_equal(ceil_sqrt(65), 9)
    with assert_raises():
        _ = dkt26_numerator(64, 1, 16)
    with assert_raises():
        _ = dkt26_numerator(64, 63, 2)
    with assert_raises():
        _ = dkt26_numerator(9223372036854775807, 2, 16)
    assert_true(abs(query_error(16, 4, 3) - 125.0 / 512.0) < 1e-15)
    # rho=16/64=1/4, m=4: exact Hab25 numerator 408146688; +1 from the upward pad.
    assert_equal(gap_numerator(64, 17, REGIME_JOHNSON, 16), 408146689)
    assert_true(gap_numerator(64, 17, REGIME_JOHNSON, 16) > gap_numerator(64, 17, REGIME_JOHNSON, 16, section4=True))
    with assert_raises():
        _ = gap_numerator(64, 1, REGIME_JOHNSON, 16)
    var large_level1 = gap_numerator(161280, 20736, REGIME_JOHNSON, 66)
    var large_tail = gap_numerator(92160, 10368, REGIME_JOHNSON, 66)
    with assert_raises():
        _ = add_numerator(large_level1, large_tail, 3)
    with assert_raises():
        _ = add_numerator(9223372036854775807, 1)
    assert_equal(add_numerator(2, 3, 4), 14)
    assert_equal(list_bound(64, 16, 16), 16)
    assert_equal(bind_numerator(2, 4, 4, 16), 1440)
    assert_equal(bind_numerator(2, 4, 4, 1), 0)
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
    comptime for i in range(3):
        comptime p = Params(e=E_BYTES, a1=2, m1=1, a2=2, m2=1, L0=48, m_cosets=1, grind_bits=0, regime=REGIME_JOHNSON if i == 2 else REGIME_UNIQUE, eta_inv=16, tail_rate_inv=32,
                            leaf_bytes=1024, tail_digits=3, tail_clear_max=100 if i == 0 else 0, lambda_bits=3, codewords=1)
        var st = Statement()
        st.col("x", BIT)
        st.family("zero", [Term(1, st.read("x"))])
        var c = st.compile[p]()
        var result = ledger[p](c)
        var numerator = 0
        for term in result[0]:
            numerator = add_numerator(numerator, term[1])
        if i == 2:
            # Johnson fixture: two queries per level, list sizes 28 (level 1) and 48 (tail).
            assert_equal(p.queries(), 2)
            assert_equal(tail_list_bound(c.shape, 0, REGIME_JOHNSON, 16), 48)
            assert_equal(tail_list_bound(c.shape, 1, REGIME_JOHNSON, 16), 1)
            for term in result[0]:
                if term[0] == "pcs_batch":
                    assert_equal(term[1], 432)   # 48 * (4*2+1)
                elif term[0] == "sumcheck":
                    assert_equal(term[1], 288)   # 48 * 6
                elif term[0] == "list_relations":
                    assert_equal(term[1], 459)   # (28-1) * (1+16)
            var first = sqrt(1.0 / 12.0) + 1.0 / 16.0
            var last = sqrt(1.0 / 35.0) + 1.0 / 16.0
            assert_true(abs(result[1] - (48.0 * first * first + last * last)) < 1e-12)
            var joint = ledger[p](c, joint_lists=True)
            assert_equal(joint_list_bound(48, 4, 16), 5)
            assert_equal(joint_list_bound(70, 2, 16), 6)
            for term in joint[0]:
                if term[0] == "pcs_batch":
                    assert_equal(term[1], 54)
                elif term[0] == "sumcheck":
                    assert_equal(term[1], 36)
                elif term[0] == "list_relations":
                    assert_equal(term[1], 68)
                elif term[0] == "opening_batch":
                    assert_equal(term[1], 10)
            assert_true(abs(joint[1] - (6.0 * first * first + last * last)) < 1e-12)
            c.shape.tail[0].codewords = 2
            with assert_raises():
                _ = ledger[p](c)
        else:
            with assert_raises():
                _ = ledger[p](c, joint_lists=True)
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
    print("CONDITIONAL LEDGER -- NOT A SECURITY CERTIFICATION. docs/security-assurance.md lists coverage and open obligations.")
    print("interactive bits use undiscounted queries; work bits divide query error by 2^grind_bits under an unproved work model.")
    print("the e16/e20 projections change only the field order; the compiled e is E_BYTES (field.mojo); no Fiat-Shamir/hash/quantum bound included.")
    # ponytail: explicit cases cover CLI grids plus four primary fixtures; compare routes when profiles change.
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
    # Public benchmark metadata fixes these statements; no witnesses or GPU work are needed.
    var lengths: List[Int] = [String(SodFixture.DG1_HEX).byte_length() // 2, String(SodFixture.ECONTENT_HEX).byte_length() // 2, String(SodFixture.ATTRS_HEX).byte_length() // 2]
    var embeds: List[Int] = [SodFixture.EMBED_1, SodFixture.EMBED_2]
    var cert_length = String(DscFixture.TBS_HEX).byte_length() // 2
    report_compiled[CLIENT.grid(144, 2016)]("rsa2048", 1, rsa_statement(8, 17).compile[CLIENT.grid(144, 2016)]())
    report_compiled[CLIENT.grid(144, 2688)]("sod", 1, sod_statement(CLIENT.grid(144, 2688).h2(), 8, 17, lengths, embeds, WINDOW_OFFSET).compile[CLIENT.grid(144, 2688)]())
    report_compiled[CLIENT.grid(144, 2688)]("dsc", 1, dsc_statement(CLIENT.grid(144, 2688).h2(), 8, 17, cert_length, DscFixture.N_OFFSET).compile[CLIENT.grid(144, 2688)]())
    report_compiled[CLIENT.grid(144, 4032)]("passport", 1, passport_statement(CLIENT.grid(144, 4032).h2(), 8, 17, lengths, embeds, WINDOW_OFFSET, cert_length, DscFixture.N_OFFSET).compile[CLIENT.grid(144, 4032)]())
    # tail rate sweep in the Johnson regime: the rho^-1.5 constant of BCHKS25 1.5 punishes low-rate tails
    comptime for r in [4, 8, 16, 32]:
        report[johnson(r).grid(144, 576)]("ecdsa_johnson_tail_rate_inv_" + String(r), 32, Ecdsa(Big(), Big(), Big(), Point.identity()))
    print("\nSTATUS: unresolved proof/implementation obligations; no certified security_bits emitted.")
