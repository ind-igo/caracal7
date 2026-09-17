"""The statement builder: the synthetic instance rebuilt by name compiles to the same bytes, every check has
a failing case, and a padded trace with a bit, a limb, and a restriction proves and verifies."""

from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceContext

from caracal7.core.field import E_BYTES
from caracal7.core.params import CLIENT
from caracal7.core.hash import Blake3
from caracal7.proof import Shape
from caracal7.prover import Prover, load_trace, load_advice
from caracal7.verifier import verify
from caracal7.relations import KIND_PERM, KIND_LOOKUP, FIX_ONE, FIX_E, CHAL_MUL, CHAL_ADD, CHAL_ONE, RES
from caracal7.relations.statement import Statement, Term, Read, BIT, LIMB6, BYTE, GATE_1, GATE_2, pad_trace, advice, restriction_line, chain_values
from caracal7.relations.ir import Families
from caracal7.workloads.synthetic import synthetic_statement, synthetic_table, wiring_statement, SYNTHETIC_PUBLIC_M

comptime p = CLIENT.grid(72, 32)


def _hand_written(columns_w: Int = 10, with_accumulator: Bool = True, with_lookup: Bool = False, with_public: Bool = False) raises -> Families:
    """The oracle: synthetic_statement written as Families calls. Eight families over ten columns, satisfied by `synthetic_trace`, plus two permutation
    accumulators (c8, c9 are c0, c1 under one permutation of the grid; records of width 1 and 2)
    whose coordinate columns start the Z tree at global index columns_w. The families cover a linear entry, a quadratic entry, both gates,
    a within-chain shift, a cyclic shift, an axis-2 shift, a challenge and basis coefficient, a
    quadratic axis-1 transition, and the accumulator."""
    if with_lookup and with_public:
        raise Error("the synthetic instance has no lookup + public combination (both use c10)")
    var f = Families()
    f.add(0, 1, 2)                                   # c2 - c0 c1
    f.add(0, 126, 0, col_b=1)
    f.add(1, 1, 3)                                   # c3 - c0 - c1
    f.add(1, 126, 0)
    f.add(1, 126, 1)
    f.add(2, 1, 4, k1_a=1, mult=1)                   # (X1 - e1) (c4(next) - c0)
    f.add(2, 126, 0, mult=1)
    f.add(3, 1, 5, col_b=5)                          # c5^2 - c5
    f.add(3, 126, 5)
    f.add(4, 1, 6)                                   # c6 - c0(omega1^3 x1): cyclic within the chain
    f.add(4, 126, 0, k1_a=3)
    f.add(5, 1, 7, k2_a=1, mult=2)                   # (X2 - e2) (c7(x1, omega2 x2) - c0)
    f.add(5, 126, 0, mult=2)
    f.add(6, 1, 2, chal=3, basis=3)                  # gamma b_3 (c2 - c0 c1): a challenge-expression coefficient
    f.add(6, 126, 0, col_b=1, chal=3, basis=3)
    f.add(7, 1, 4, k1_a=1, col_b=5, mult=1)          # (X1 - e1) c5 (c4(next) - c0): quadratic with the axis-1 gate
    f.add(7, 126, 0, col_b=5, mult=1)
    if with_accumulator:
        f.accumulator(8, columns_w, [0], [8])              # (X1 - e1) (Z(next) (gamma + c8) - Z (gamma + c0))
        f.accumulator(9, columns_w + E_BYTES, [0, 1], [8, 9])   # width-2 records (c0, c1) against (c8, c9): the basis products b_t b_j
    if with_lookup:                                        # (c10, c11) in `synthetic_table`; the prover sorts them into (c12, c13)
        f.lookup(10, columns_w + (2 * E_BYTES if with_accumulator else 0), [10, 11], [12, 13], 0)
    if with_public:                                        # c_{w-1} - c0 pub: the public column is the first past W and Z
        f.add(10, 1, columns_w - 1)
        f.add(10, 126, 0, col_b=columns_w + (2 * E_BYTES if with_accumulator else 0))
    return f^


def test_synthetic_rebuilt_by_name_is_byte_equal() raises:
    for lookup in [False, True]:
        var c = synthetic_statement(with_lookup=lookup).compile[p]()
        var f = _hand_written(14 if lookup else 10, with_lookup=lookup)
        assert_equal(c.families, f.bytes)
        assert_equal(c.shape.accs, f.accs)
        assert_equal(c.shape.columns_w, 14 if lookup else 10)
        assert_equal(c.layout.col("c9"), 9)
    var c = synthetic_statement(with_public=True).compile[p]()
    var f = _hand_written(11, with_public=True)
    assert_equal(c.families, f.bytes)
    assert_equal(c.shape.columns_p, 1)
    assert_equal(len(c.shape.restrictions), RES)
    var f0 = synthetic_statement(53, with_accumulator=False).compile[p]()       # filler columns are bits with their certificate
    assert_equal(f0.shape.columns_w, 53)
    assert_equal(f0.shape.entries, 17 + 2 * 45)                                   # c8, c9 are bits without accumulators


def _fails(var st: Statement) -> String:
    try:
        _ = st.compile[p]()
    except e:
        return String(e)
    return String("")


def test_every_check_has_a_failing_case() raises:
    var st = synthetic_statement()
    st.col("idle")
    assert_equal(_fails(st^), "column is read by nothing and constrained by nothing: idle")
    st = synthetic_statement()
    st.family("bad", [Term(1, st.read("c2"), st.read("c3"))], GATE_2)
    assert_equal(_fails(st^), "axis-2 gated entries must be linear (spec 8 degree bound)")
    st = synthetic_statement()
    st.family("bad", [Term(1, st.read("c2", k1=p.h1())), Term(-1, st.read("c3"))])
    assert_equal(_fails(st^), "read shift: k1 in [0, h1), k2 in [0, h2)")
    st = synthetic_statement()
    st.family("bad", [Term(1, st.read("c2", k2=p.h2())), Term(-1, st.read("c3"))])
    assert_equal(_fails(st^), "read shift: k1 in [0, h1), k2 in [0, h2)")
    st = synthetic_statement()
    st.family("bad", [Term(1, st.read("nope"))])
    assert_equal(_fails(st^), "unknown column nope")
    st = synthetic_statement()
    var err = String("")
    try:
        st.horner("bad", [Term(1, st.read("c2"), st.read("c3"))], scale=2)
    except e:
        err = String(e)
    assert_equal(err, "horner ingest terms are same-chain reads without basis factors, times an unshifted public selector at most: bad")
    try:
        st.chain_end("bad", [Term(1, st.read("z0", k1=1))])
    except e:
        err = String(e)
    assert_equal(err, "chain-end terms read accumulators at (e1, X2) with no shift or basis: bad")
    st = synthetic_statement()
    st.chain_end("bad", [Term(1, st.read("nope"))])
    assert_equal(_fails(st^), "unknown accumulator nope")
    st = synthetic_statement()
    st.restrict("c1", FIX_E, p.h1() + 1)
    assert_equal(_fails(st^), "restriction needs an opened column, a fixed axis-2 coordinate, and a coefficient count in [1, h1]")
    st = synthetic_statement()
    st.family("bad", [Term(1, st.read("c2"), chal=9)])
    assert_equal(_fails(st^), "family entry names a challenge element past the derivation table")
    st = synthetic_statement()
    st.acc("z2", KIND_PERM, ["c0"], ["z0"])
    assert_equal(_fails(st^), "unknown witness column z0")
    st = synthetic_statement()
    st.acc("z2", KIND_LOOKUP, ["c0", "c1"], ["c0", "c9"], table=st.table(synthetic_table(), 2))
    assert_equal(_fails(st^), "a sorted column belongs to one lookup and is read by nothing else: c0")
    st = synthetic_statement(with_public=True)
    st.family("bad", [Term(1, st.read("pub"), st.read("pub"))])
    assert_equal(_fails(st^), "a quadratic entry may read at most one public column")
    st = synthetic_statement(with_lookup=True)
    st.family("bad", [Term(1, st.read("c12")), Term(-1, st.read("c0"))])
    assert_equal(_fails(st^), "a sorted column belongs to one lookup and is read by nothing else: c12")
    st = synthetic_statement(with_lookup=True)
    st.restrict("c13", FIX_ONE)
    assert_equal(_fails(st^), "a sorted column belongs to one lookup and is read by nothing else: c13")
    st = synthetic_statement()
    st.pub("q", 5)
    assert_equal(_fails(st^), "public column period: m divides h2: q")
    var stopped = String("")
    st = synthetic_statement()
    try:
        st.col("c0")
    except e:
        stopped = String(e)
    assert_equal(stopped, "name in use: c0")
    try:
        _ = st.derived(CHAL_MUL, 5, 1)
    except e:
        stopped = String(e)
    assert_equal(stopped, "challenge derivation row must add or multiply earlier elements")
    try:
        _ = st.slot("nope")
    except e:
        stopped = String(e)
    assert_equal(stopped, "unknown accumulator nope")
    var ws = wiring_statement(p.h2())
    try:
        ws.public_factor("v", "ra", 0, 0)
    except e:
        stopped = String(e)
    assert_equal(stopped, "name in use: v")
    ws.wire(0, p.h2(), 1, 0)
    assert_equal(_fails(ws^), "wire chain index is outside the grid")
    ws = wiring_statement(p.h2())
    try:
        ws.zero("a", 3)
    except e:
        stopped = String(e)
    assert_equal(stopped, "zero row coordinate is FIX_ONE or FIX_E")
    ws.zero("a", FIX_E)
    var wz = ws^.compile[p]()
    assert_equal(len(wz.shape.zeros), 4)
    assert_equal(wz.shape.points, wiring_statement(p.h2()).compile[p]().shape.points)   # (e1, z2) is a boundary point already
    ws = Statement()
    ws.col("a", BIT)
    ws.horner("ra", [Term(1, ws.read("a"))], scale=2)
    _ = ws.slot("ra")
    ws.wire(0, 0, 0, 1)
    assert_true(ws.compile[p]().shape.wiring_products() == 1, "a single slot pads to one product")
    var wc = wiring_statement(p.h2()).compile[p]()
    var sigma = wc.shape.sigma.copy()
    sigma[2] = sigma[0]                                # slot 0 on chains 0 and 1 map to one id
    sigma[3] = sigma[1]
    try:
        _ = Shape.__init__[p](wc.layout.columns_w(), wc.families, wc.shape.accs, wc.shape.tables, wc.shape.publics, wc.shape.restrictions,
                              wc.shape.point_list, wc.shape.chals, wc.shape.ends, wc.shape.wires, sigma, wc.shape.pubf)
    except e:
        stopped = String(e)
    assert_equal(stopped, "sigma must map every slot to a distinct id")
    try:
        st.acc("z2", KIND_LOOKUP, ["c0"], ["c9"], table=st.table(synthetic_table(), 2))
    except e:
        stopped = String(e)
    assert_equal(stopped, "lookup record width differs from its table's")
    var el = st.derived(CHAL_ADD, 2, CHAL_ONE)
    assert_equal(el, 5)


def test_padded_trace_proves_and_verifies() raises:
    """A group of a bit, a limb, and a product y = b x, live on all but the last two chains; the builder pads
    the rest, emits the Booleanity family and the range lookup, and the prover accepts the trace. A restriction
    pins x on the first chain to its interpolant."""
    comptime N = p.N()
    comptime h1 = p.h1()
    var st = Statement()
    st.col("b", BIT, group="g")
    st.col("x", BYTE, group="g")
    st.col("y", BYTE, group="g")
    st.col("l", LIMB6, group="g")
    st.family("prod", [Term(1, st.read("y")), Term(-1, st.read("b"), st.read("x"))])
    st.restrict("x", FIX_ONE, h1)
    var c = st.compile[p]()
    assert_equal(c.layout.columns_w(), 5)
    assert_equal(c.layout.col("l.sorted"), 4)
    assert_equal(c.shape.accumulators(), 1)
    assert_equal(c.shape.entries, 2 + 2 + E_BYTES * (2 + 3))
    var live = N - 2 * h1
    var trace = List[UInt8](length=5 * N, fill=0)
    var s = 7
    for i in range(N):                                   # every row filled; padding must overwrite the idle ones
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        trace[i] = UInt8((s >> 20) & 1)
        trace[N + i] = UInt8((s >> 8) % 127)
        trace[2 * N + i] = trace[i] * trace[N + i]
        trace[3 * N + i] = UInt8((s >> 13) % 64)
    pad_trace[p](c.layout, trace, "g", live)
    for i in range(live, N):
        assert_equal(trace[i] + trace[N + i] + trace[2 * N + i], 0)
        assert_equal(trace[3 * N + i], UInt8(i % 64))
        assert_equal(trace[4 * N + i], 0)                 # the sorted column is left to the prover
    var ctx = DeviceContext()
    var prover = Prover[p, Blake3](ctx, st.compile[p]().take_shape(), c.families.copy())
    load_trace[p, Blake3](ctx, prover, trace)
    load_advice[p, Blake3](ctx, prover, advice[p](c.layout, trace))
    var proof = prover.prove(ctx, List[UInt8]())
    var line = restriction_line[p](c.layout, 0, chain_values[p](c.layout, trace, 0))
    assert_true(verify[p, Blake3](proof.copy(), c.shape, List[UInt8](), c.families, line))
    var stopped = String("")
    try:
        pad_trace[p](c.layout, trace, "h", live)
    except e:
        stopped = String(e)
    assert_equal(stopped, "unknown group h")
    try:
        pad_trace[p](c.layout, trace, "g", live + 1)
    except e:
        stopped = String(e)
    assert_equal(stopped, "live_rows is whole chains (a multiple of h1): cyclic reads wrap inside a chain")
    var zeros = trace.copy()                              # a limb column that never shows row 1 fails the dummy rule
    for i in range(N):
        zeros[3 * N + i] = 0
    try:
        _ = advice[p](c.layout, zeros)
    except e:
        stopped = String(e)
    assert_equal(stopped, "lookup table row 1 occurs in no record (pad with the table)")
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks (decisions.md 2026-09-16)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
