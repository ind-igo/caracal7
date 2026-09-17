"""Row groups on one grid (Statement.group): two groups pool their physical columns per kind, their families
hold under the group selectors (a GATE_2 family of A under A's inner selector: its next-chain read on A's
last chain lands in B), one Horner accumulator fingerprints a value of each group, a wire equates them across
the groups, a public factor fingerprints one, and the idle tail is padded by the last group."""

from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceContext

from caracal7.core.params import CLIENT
from caracal7.core.hash import Blake3
from caracal7.relations.statement import Statement, Term, Compiled, BIT, LIMB6, BYTE, GATE_2, KIND_LOOKUP, pad_trace
from caracal7.prover import Prover, load_trace, load_public
from caracal7.verifier import verify

comptime p = CLIENT.grid(72, 32)
comptime N = p.N()
comptime h1 = p.h1()
comptime h2 = p.h2()
comptime CHAINS_A = 8
comptime CHAINS_B = 12


def _statement() raises -> Statement:
    var st = Statement()
    st.pub("gA", 1)
    st.pub("gB", 1)
    st.pub("gAi", 1)
    var base_a = st.group("A", CHAINS_A, "gA", inner="gAi")
    var base_b = st.group("B", CHAINS_B, "gB")
    assert_equal(base_a, 0)
    assert_equal(base_b, CHAINS_A)
    st.col("a0", BIT, group="A")
    st.col("a1", BIT, group="A")
    st.col("a2", BYTE, group="A")
    st.col("a3", BYTE, group="A")
    st.col("b0", BIT, group="B")
    st.col("b1", BYTE, group="B")
    st.col("b2", BYTE, group="B")
    st.family("sumA", [Term(1, st.read("a0")), Term(1, st.read("a1")), Term(-1, st.read("a2"))])
    st.family("nxA", [Term(1, st.read("a2", k2=1)), Term(-1, st.read("a3"))], GATE_2)   # a3 = the next chain's a2; chain 7 reads B
    st.family("eqB", [Term(1, st.read("b0")), Term(-1, st.read("b1"))])
    st.family("sqB", [Term(1, st.read("b2")), Term(-1, st.read("b1"), st.read("b1"))])
    st.horner("fp", [Term(1, st.read("a2")), Term(1, st.read("b2"))], scale=2)
    var s = st.slot("fp")
    st.wire(s, 0, s, base_b)
    st.public_factor("pf", "fp", s, 1)
    return st^


def _trace(layout_a0: Int, layout_a1: Int, layout_a2: Int, layout_a3: Int, layout_b0: Int, layout_b1: Int, layout_b2: Int, columns: Int) -> List[UInt8]:
    var t = List[UInt8](length=columns * N, fill=0)
    var s = 5
    for x2 in range(CHAINS_A):
        for x1 in range(h1):
            s = (s * 1103515245 + 12345) & 0x7FFFFFFF
            var a0 = UInt8((s >> 8) & 1)
            var a1 = UInt8((s >> 9) & 1) if x2 != 0 else UInt8(0)
            t[layout_a0 * N + x2 * h1 + x1] = a0
            t[layout_a1 * N + x2 * h1 + x1] = a1
            t[layout_a2 * N + x2 * h1 + x1] = a0 + a1
            if x2 > 0:
                t[layout_a3 * N + (x2 - 1) * h1 + x1] = a0 + a1
    for x2 in range(CHAINS_A, CHAINS_A + CHAINS_B):
        for x1 in range(h1):
            s = (s * 1103515245 + 12345) & 0x7FFFFFFF
            var b0 = UInt8((s >> 8) & 1) if x2 != CHAINS_A else t[layout_a0 * N + x1]   # chain 8 copies chain 0's a2: the wire
            t[layout_b0 * N + x2 * h1 + x1] = b0
            t[layout_b1 * N + x2 * h1 + x1] = b0
            t[layout_b2 * N + x2 * h1 + x1] = b0
    return t^


def _verdict(proof: List[UInt8], c: Compiled, public: List[UInt8]) raises -> String:
    var fam = c.families.copy()
    try:
        _ = verify[p, Blake3](proof.copy(), c.shape, List[UInt8](), fam, public)
    except e:
        return String(e)
    return String("accepted")


def test_two_groups_pool_columns_and_prove() raises:
    var ctx = DeviceContext()
    var c = _statement().compile[p]()
    var L = c.layout.copy()
    assert_equal(L.columns_w(), 5)                       # b1 exclusive (a quadratic factor); a0|b0, a1; a2|b2, a3
    assert_equal(L.col("a0"), L.col("b0"))
    assert_equal(L.col("a2"), L.col("b2"))
    assert_true(L.col("b1") != L.col("a2"))
    assert_true(L.col("a3") != L.col("b1"))
    var trace = _trace(L.col("a0"), L.col("a1"), L.col("a2"), L.col("a3"), L.col("b0"), L.col("b1"), L.col("b2"), L.columns_w())
    pad_trace[p](L, trace, "A", CHAINS_A * h1)
    pad_trace[p](L, trace, "B", CHAINS_B * h1)
    var block = L.selector[p]("A")
    block.extend(L.selector[p]("B"))
    block.extend(L.selector[p]("A", inner=True))
    assert_equal(Int(block[2 * N + (CHAINS_A - 1) * h1]), 0)
    assert_equal(Int(block[2 * N + (CHAINS_A - 2) * h1]), 1)
    var prover = Prover[p, Blake3](ctx, _statement().compile[p]().take_shape(), c.families.copy())
    load_trace[p, Blake3](ctx, prover, trace)
    load_public[p, Blake3](ctx, prover, block)
    var proof = prover.prove(ctx, List[UInt8]())
    var public = block.copy()
    for _ in range(2):                                   # the factor: h1 bytes per ingest entry of fp (a2 on chain 1; the b2 entry is off there)
        for x1 in range(h1):
            public.append(trace[L.col("a2") * N + h1 + x1])
    assert_equal(_verdict(proof, c, public), "accepted")
    var wrong = public.copy()
    wrong[3 * N + 3] = (wrong[3 * N + 3] + 1) % 127
    assert_equal(_verdict(proof, c, wrong), "wiring grand product is not the public factor")
    var erased = public.copy()                           # a selector that is not the chain indicator is refused, not a different statement
    erased[3 * h1] = 0
    assert_equal(_verdict(proof, c, erased), "a group selector's public data is not its chain indicator")
    erased = public.copy()
    erased[2 * N + (CHAINS_A - 1) * h1] = 1
    assert_equal(_verdict(proof, c, erased), "a group selector's public data is not its chain indicator")
    # a B chain that breaks eqB under its selector
    var bad = trace.copy()
    bad[L.col("b1") * N + (CHAINS_A + 2) * h1 + 5] = 5
    load_trace[p, Blake3](ctx, prover, bad)
    var proof2 = prover.prove(ctx, List[UInt8]())
    assert_equal(_verdict(proof2, c, public), "residual identity fails at z")
    # an A chain whose a3 is not the next chain's a2 (chain 7's a3 is free: the inner selector is 0 there)
    var bad2 = trace.copy()
    bad2[L.col("a3") * N + 3 * h1 + 5] = (bad2[L.col("a3") * N + 3 * h1 + 5] + 1) % 127
    load_trace[p, Blake3](ctx, prover, bad2)
    var proof3 = prover.prove(ctx, List[UInt8]())
    assert_equal(_verdict(proof3, c, public), "residual identity fails at z")
    var free = trace.copy()
    free[L.col("a3") * N + (CHAINS_A - 1) * h1 + 5] = 9
    load_trace[p, Blake3](ctx, prover, free)
    var proof4 = prover.prove(ctx, List[UInt8]())
    assert_equal(_verdict(proof4, c, public), "accepted")
    try:
        pad_trace[p](L, free, "B", (CHAINS_B + 1) * h1)
    except e:
        assert_equal(String(e), "live_rows exceed the group's chains")
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks (decisions.md 2026-09-16)


def test_group_checks() raises:
    var st = Statement()
    st.pub("g", 1)
    _ = st.group("A", 4, "g")
    var err = String("")
    try:
        _ = st.group("B", 4, "nope")
    except e:
        err = String(e)
    assert_equal(err, "group selector is a declared public column with m = 1: B")
    st.pub("h2", 2)
    try:
        _ = st.group("B", 4, "h2")
    except e:
        err = String(e)
    assert_equal(err, "group selector is a declared public column with m = 1: B")
    st.pub("h", 1)
    _ = st.group("B", 4, "h")
    st.col("x", BYTE, group="A")
    st.col("y", BYTE, group="B")
    st.family("mixed", [Term(1, st.read("x")), Term(-1, st.read("y"))])
    try:
        _ = st.compile[p]()
    except e:
        err = String(e)
    assert_equal(err, "family reads columns of two groups: mixed")
    var st2 = Statement()
    st2.pub("g", 1)
    _ = st2.group("A", 4, "g")
    st2.col("x", BYTE, group="A")
    st2.horner("r", [Term(1, st2.read("x"))], scale=2, start=1)
    try:
        _ = st2.compile[p]()
    except e:
        err = String(e)
    assert_equal(err, "a horner accumulator over a group's columns starts at 0 (neutral off the group): r")
    var st3 = Statement()
    st3.pub("g", 1)
    st3.pub("m", 1)
    _ = st3.group("A", 4, "g")
    st3.col("x", BYTE, group="A")
    st3.col("u", BYTE)
    st3.family("leak", [Term(1, st3.read("x")), Term(-1, st3.read("u"))])
    try:
        _ = st3.compile[p]()
    except e:
        err = String(e)
    assert_equal(err, "a grouped family reads an ungrouped witness column (free off the group): leak")
    var st4 = Statement()
    st4.pub("g", 1)
    _ = st4.group("A", 4, "g")
    st4.col("x", BYTE, group="A")
    st4.col("y", BYTE, group="A")
    st4.family("nx", [Term(1, st4.read("x", k2=1)), Term(-1, st4.read("y"))], GATE_2)
    try:
        _ = st4.compile[p]()
    except e:
        err = String(e)
    assert_equal(err, "a grouped GATE_2 family needs the group's inner selector: nx")
    var st5 = Statement()
    st5.pub("g", 1)
    st5.pub("m", 1)
    _ = st5.group("A", 4, "g")
    st5.col("x", BYTE, group="A")
    st5.horner("r", [Term(1, st5.read("x"), st5.read("m"))], scale=2)
    try:
        _ = st5.compile[p]()
    except e:
        err = String(e)
    assert_equal(err, "a grouped horner term takes the group's selector only: r")
    var st6 = Statement()                                # a pooled column nothing reads hides behind its physical column
    st6.pub("g", 1)
    st6.pub("h", 1)
    _ = st6.group("A", 4, "g")
    _ = st6.group("B", 4, "h")
    st6.col("x", BYTE, group="A")
    st6.col("y", BYTE, group="B")
    st6.family("zx", [Term(1, st6.read("x"))])
    try:
        _ = st6.compile[p]()
    except e:
        err = String(e)
    assert_equal(err, "column is read by nothing and constrained by nothing: y")
    var st7 = Statement()
    st7.pub("g", 1)
    st7.pub("gi", 1)
    _ = st7.group("A", 4, "g", inner="gi")
    st7.col("x", BYTE, group="A")
    st7.col("y", BYTE, group="A")
    st7.col("z", BYTE, group="A")
    st7.family("q", [Term(1, st7.read("z")), Term(-1, st7.read("x", k2=1), st7.read("y"))], GATE_2)
    try:
        _ = st7.compile[p]()
    except e:
        err = String(e)
    assert_equal(err, "a grouped GATE_2 family is linear in the group's witness columns (the inner selector is its only factor): q")
    var st11 = Statement()                               # a lookup record is not zero off the group
    st11.pub("g", 1)
    _ = st11.group("A", 4, "g")
    st11.col("a", BYTE, group="A")
    st11.col("l", BYTE, group="A")
    st11.col("ls", BYTE, group="A")
    st11.col("z", BYTE, group="A")
    st11.acc("lk", KIND_LOOKUP, ["l"], ["ls"], table=st11.table([1, 2], 1))
    st11.family("q", [Term(1, st11.read("z")), Term(-1, st11.read("a"), st11.read("l"))])
    try:
        _ = st11.compile[p]()
    except e:
        err = String(e)
    assert_equal(err, "a grouped family's quadratic factor is a plain witness column (zero off the group), not a lookup record: q")
    var st12 = Statement()                               # a LIMB6 is a record too (the builder's range lookup)
    st12.pub("g", 1)
    _ = st12.group("A", 4, "g")
    st12.col("l", LIMB6, group="A")
    st12.col("z", BYTE, group="A")
    st12.family("q", [Term(1, st12.read("z")), Term(-1, st12.read("l"), st12.read("l"))])
    try:
        _ = st12.compile[p]()
    except e:
        err = String(e)
    assert_equal(err, "a grouped family's quadratic factor is a plain witness column (zero off the group), not a lookup record: q")
    var st13 = Statement()                               # a declared group's pad leaves an ungrouped column alone
    st13.pub("g", 1)
    _ = st13.group("A", 4, "g")
    st13.col("x", BYTE, group="A")
    st13.col("u", BYTE)
    st13.family("zx", [Term(1, st13.read("x"))])
    st13.family("zu", [Term(1, st13.read("u"))])
    var L = st13.compile[p]().layout.copy()
    var t = List[UInt8](length=2 * N, fill=1)
    pad_trace[p](L, t, "A", 2 * h1)
    assert_equal(Int(t[L.col("u") * N + N - 1]), 1)
    assert_equal(Int(t[L.col("x") * N + N - 1]), 0)
    pad_trace[p](L, t, "", 0)
    assert_equal(Int(t[L.col("u") * N + N - 1]), 0)
    var st9 = Statement()                                # an ungated next-chain read lands in cells nothing constrains
    st9.pub("g", 1)
    _ = st9.group("A", 4, "g")
    st9.col("x", BYTE, group="A")
    st9.col("y", BYTE, group="A")
    st9.family("nx", [Term(1, st9.read("x", k2=1)), Term(-1, st9.read("y"))])
    try:
        _ = st9.compile[p]()
    except e:
        err = String(e)
    assert_equal(err, "a grouped family reads another chain only in a linear GATE_2 term (k2 = 1, the inner selector masks the group's last chain); other reads land in cells no family constrains: nx")
    var st10 = Statement()
    st10.pub("g", 1)
    _ = st10.group("A", 4, "g")
    try:
        _ = st10.group("B", 4, "g")
    except e:
        err = String(e)
    assert_equal(err, "a group's selectors are its own columns: B")
    var st8 = Statement()                                # a public-only term compiles unselected (the gadget zeroes it off the group)
    st8.pub("g", 1)
    st8.pub("P", 1)
    _ = st8.group("A", 4, "g")
    st8.col("x", BYTE, group="A")
    st8.family("xp", [Term(1, st8.read("x")), Term(-1, st8.read("P"))])
    _ = st8.compile[p]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
