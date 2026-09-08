"""The synthetic instance the tests and benches run before a real frontend exists: eight families
over ten columns, two permutation accumulators, and a trace that satisfies them."""

from caracal7.core.field import F2, f_add, f_mul
from caracal7.core.params import Params
from caracal7.relations.ir import FIX_E, KIND_PERM, KIND_LOOKUP, PUB, CHAL_MUL
from caracal7.relations.statement import Statement, Layout, Term, BIT, BYTE, GATE_1, GATE_2, restriction_line, chain_values
from caracal7.workload import Workload
from caracal7.core.bytes import append_u32, set_u16


@fieldwise_init
struct Synthetic(Workload, Copyable, Movable):
    """The synthetic instance as a Workload: the public inputs are the restricted column's chain values (with
    the public column), the public column is the fixed `synthetic_public_value` and needs no input."""
    var columns_w: Int
    var with_accumulator: Bool
    var with_lookup: Bool
    var with_public: Bool
    var seed: Int

    def statement(self) raises -> Statement:
        return synthetic_statement(self.columns_w, self.with_accumulator, self.with_lookup, self.with_public)

    def trace[p: Params](self, layout: Layout) raises -> List[UInt8]:
        var t = synthetic_trace[p](self.seed, self.with_lookup, self.with_public, self.with_accumulator)
        t.resize(layout.columns_w() * p.N(), 0)              # filler bit columns past the instance are zero
        return t^

    def public_inputs[p: Params](self) raises -> List[UInt8]:
        if not self.with_public:
            return List[UInt8]()
        var c = self.statement().compile[p]()
        return chain_values[p](c.layout, self.trace[p](c.layout), 0)

    @staticmethod
    def public_data[p: Params](layout: Layout, public_inputs: List[UInt8]) raises -> List[UInt8]:
        if len(layout.publics) == 0:
            return List[UInt8]()
        var data = synthetic_public_values[p]()
        data.extend(restriction_line[p](layout, 0, public_inputs))
        return data^


def synthetic_statement(columns_w: Int = 10, with_accumulator: Bool = True, with_lookup: Bool = False, with_public: Bool = False) raises -> Statement:
    """Eight families over ten columns, satisfied by `synthetic_trace`, plus two permutation accumulators
    (c8, c9 are c0, c1 under one permutation of the grid; records of width 1 and 2). The families cover a
    linear entry, a quadratic entry, both gates, a within-chain shift, a cyclic shift, an axis-2 shift, a
    challenge and basis coefficient, a quadratic axis-1 transition, and the accumulator. With the lookup,
    (c10, c11) against `synthetic_table` sorted into (c12, c13); with the public column, c10 = c0 pub and c1
    restricted on the last chain to its own line. Columns past the instance are bits (constrained by their
    certificate) so a wider layout can be planned."""
    if with_lookup and with_public:
        raise Error("the synthetic instance has no lookup + public combination (both use c10)")
    var st = Statement()
    var base = SYNTHETIC_LOOKUP_COLUMNS if with_lookup else (SYNTHETIC_PUBLIC_COLUMNS if with_public else SYNTHETIC_COLUMNS)
    for i in range(base):
        st.col("c" + String(i), BYTE if with_accumulator or i < 8 or i > 9 else BIT)   # c8, c9 exist for the accumulators; bits (zero) without them
    for i in range(base, columns_w):
        st.col("c" + String(i), BIT)
    st.family("mul", [Term(1, st.read("c2")), Term(-1, st.read("c0"), st.read("c1"))])                     # c2 - c0 c1
    st.family("add", [Term(1, st.read("c3")), Term(-1, st.read("c0")), Term(-1, st.read("c1"))])          # c3 - c0 - c1
    st.family("next", [Term(1, st.read("c4", k1=1)), Term(-1, st.read("c0"))], GATE_1)                    # (X1 - e1) (c4(next) - c0)
    st.family("bool", [Term(1, st.read("c5"), st.read("c5")), Term(-1, st.read("c5"))])                    # c5^2 - c5, written out (c5 is BYTE)
    st.family("cyc", [Term(1, st.read("c6")), Term(-1, st.read("c0", k1=3))])                             # c6 - c0(omega1^3 x1)
    st.family("chain", [Term(1, st.read("c7", k2=1)), Term(-1, st.read("c0"))], GATE_2)                   # (X2 - e2) (c7(x1, omega2 x2) - c0)
    st.family("chal", [Term(1, st.read("c2"), chal=2, basis=3), Term(-1, st.read("c0"), st.read("c1"), chal=2, basis=3)])   # gamma b_3 (c2 - c0 c1)
    st.family("quad", [Term(1, st.read("c4", k1=1), st.read("c5")), Term(-1, st.read("c0"), st.read("c5"))], GATE_1)       # (X1 - e1) c5 (c4(next) - c0)
    if with_accumulator:
        st.acc("z0", KIND_PERM, ["c0"], ["c8"])                       # (X1 - e1) (Z(next) (gamma + c8) - Z (gamma + c0))
        st.acc("z1", KIND_PERM, ["c0", "c1"], ["c8", "c9"])           # width-2 records: the basis products b_t b_j
    if with_lookup:
        st.acc("z2", KIND_LOOKUP, ["c10", "c11"], ["c12", "c13"], table=st.table(synthetic_table(), 2))
    if with_public:
        st.pub("pub", SYNTHETIC_PUBLIC_M)                            # d2 = h2 / 4: the column is periodic along axis 2
        st.family("public", [Term(1, st.read("c10")), Term(-1, st.read("c0"), st.read("pub"))])
        st.restrict("c1", FIX_E)
    return st^


comptime SYNTHETIC_PUBLIC_COLUMNS = 11                  # ten plus c10 = c0 pub
comptime SYNTHETIC_PUBLIC_M = 4                         # the public column is periodic along axis 2 with period h2 / 4, so its block has d2 = h2 / 4 rows


def synthetic_public_value[p: Params](x1: Int, x2: Int) -> UInt8:
    """The public column on H: an F value of (x1, x2 mod h2 / m)."""
    return UInt8((x1 * 7 + (x2 % (p.h2() // SYNTHETIC_PUBLIC_M)) * 13 + 3) % 127)


def synthetic_publics[p: Params]() -> List[UInt8]:
    """The PUB record of the one public column: m = 4."""
    var b = List[UInt8](length=PUB, fill=0)
    set_u16(b, 0, SYNTHETIC_PUBLIC_M)
    return b^


def synthetic_public_values[p: Params]() -> List[UInt8]:
    """One period of the public column: (h2 / m, h1) values."""
    comptime h1 = p.h1()
    var vals = List[UInt8](capacity=(p.h2() // SYNTHETIC_PUBLIC_M) * h1)
    for x2 in range(p.h2() // SYNTHETIC_PUBLIC_M):
        for x1 in range(h1):
            vals.append(synthetic_public_value[p](x1, x2))
    return vals^


comptime SYNTHETIC_LOOKUP_COLUMNS = 14
comptime SYNTHETIC_TABLE_ROWS = 64


def synthetic_table() -> List[UInt8]:
    """Table 0 of the lookup instance: row j is (j, 3 j + 1 mod 127)."""
    var t = List[UInt8](capacity=2 * SYNTHETIC_TABLE_ROWS)
    for j in range(SYNTHETIC_TABLE_ROWS):
        t.append(UInt8(j))
        t.append(UInt8((3 * j + 1) % 127))
    return t^


def synthetic_lookup_index(i: Int) -> Int:
    """The table row record i holds: rows 0 .. K - 1 cover the table (the dummy rule), the rest are spread."""
    return i if i < SYNTHETIC_TABLE_ROWS else ((i * 2654435761) >> 7) % SYNTHETIC_TABLE_ROWS


def synthetic_advice[p: Params]() -> List[UInt8]:
    """The advice index list of the lookup instance (sort.mojo), u32 per row."""
    var a = List[UInt8](capacity=4 * p.N())
    for i in range(p.N()):
        append_u32(a, synthetic_lookup_index(i))
    return a^


comptime SYNTHETIC_COLUMNS = 10
comptime SYNTHETIC_PERM = 17                            # c8[i] = c0[(17 i + 5) mod N]: coprime to every grid N


def synthetic_trace[p: Params](seed: Int, with_lookup: Bool = False, with_public: Bool = False, with_accumulator: Bool = True) -> List[UInt8]:
    """Ten columns (column, x2, x1) satisfying `synthetic_statement`; with the lookup, fourteen: the records in
    c10, c11 and the sorted copy's columns c12, c13 left zero for the prover; with the public column (not
    with the lookup), eleven: c10 = c0 pub."""
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    comptime N = p.N()
    var t = List[UInt8](length=(SYNTHETIC_LOOKUP_COLUMNS if with_lookup else (SYNTHETIC_PUBLIC_COLUMNS if with_public else SYNTHETIC_COLUMNS)) * N, fill=0)
    if with_lookup:
        var table = synthetic_table()
        for i in range(N):
            t[10 * N + i] = table[2 * synthetic_lookup_index(i)]
            t[11 * N + i] = table[2 * synthetic_lookup_index(i) + 1]
    var s = seed
    for i in range(N):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        t[i] = UInt8((s >> 8) % 127)
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        t[N + i] = UInt8((s >> 8) % 127)
        t[5 * N + i] = UInt8((s >> 20) & 1)
    for x2 in range(h2):
        for x1 in range(h1):
            var i = x2 * h1 + x1
            var c0 = SIMD[DType.uint8, 1](t[i])
            var c1 = SIMD[DType.uint8, 1](t[N + i])
            t[2 * N + i] = f_mul(c0, c1)[0]
            t[3 * N + i] = f_add(c0, c1)[0]
            t[4 * N + i] = t[x2 * h1 + (x1 + h1 - 1) % h1] if x1 > 0 else UInt8((i * 7) % 127)     # c4(omega1 x1) = c0(x1)
            t[6 * N + i] = t[x2 * h1 + (x1 + 3) % h1]                                              # c6 = c0(omega1^3 x1)
            t[7 * N + i] = t[(x2 - 1) * h1 + x1] if x2 > 0 else UInt8((i * 11) % 127)              # c7(omega2 x2) = c0(x2)
            if with_accumulator:
                t[8 * N + i] = t[(SYNTHETIC_PERM * i + 5) % N]                                         # c8 = c0 permuted
                t[9 * N + i] = t[N + (SYNTHETIC_PERM * i + 5) % N]                                     # c9 = c1 under the same permutation
            if with_public:
                t[10 * N + i] = f_mul(c0, SIMD[DType.uint8, 1](synthetic_public_value[p](x1, x2)))[0]
    return t^


# ---- the second Z kind: polynomial-mulmod 5 in miniature ----

comptime HORNER_DEGREE = 8      # a and b carry weights 0..8, c weights 0..16; weight m lives at row h1 - 2 - m (h1 >= 24)


@fieldwise_init
struct SyntheticHorner(Workload, Copyable, Movable):
    """Per chain, bits a and b and coefficients c = a * b as polynomials, checked as A(gamma) B(gamma) = C(gamma)
    through three Horner accumulators and one chain-end family; no public inputs."""
    var seed: Int

    def statement(self) raises -> Statement:
        return horner_statement()

    def trace[p: Params](self, layout: Layout) raises -> List[UInt8]:
        return horner_trace[p](self.seed)

    def public_inputs[p: Params](self) raises -> List[UInt8]:
        return List[UInt8]()

    @staticmethod
    def public_data[p: Params](layout: Layout, public_inputs: List[UInt8]) raises -> List[UInt8]:
        return List[UInt8]()


def horner_statement() raises -> Statement:
    """R_A = gamma A(gamma) ingesting a one row on (the cyclic read k1 = 1); R_B = 3 delta B(gamma) ingesting b with
    the weight 3 delta; R_C = C(gamma); all with scale gamma, high weight first. The chain-end check
    R_A R_B - 3 delta gamma R_C = 0 on H2 is ungated: an idle chain is all zero."""
    var st = Statement()
    st.col("a", BIT)
    st.col("b", BIT)
    st.col("c", BYTE)
    var dg = st.derived(CHAL_MUL, 1, 2)                          # delta gamma
    st.horner("ra", [Term(1, st.read("a", k1=1))], scale=2)
    st.horner("rb", [Term(3, st.read("b"), chal=1)], scale=2)
    st.horner("rc", [Term(1, st.read("c"))], scale=2)
    st.chain_end("mul", [Term(1, st.read("ra"), st.read("rb")), Term(-3, st.read("rc"), chal=dg)])
    return st^


def horner_trace[p: Params](seed: Int) -> List[UInt8]:
    """Columns a, b, c (column, x2, x1): random bits at weights 0..HORNER_DEGREE, c their product's coefficients."""
    comptime h1 = p.h1()
    comptime N = p.N()
    var t = List[UInt8](length=3 * N, fill=0)
    var s = seed
    for x2 in range(p.h2()):
        var a = List[Int](length=HORNER_DEGREE + 1, fill=0)
        var b = List[Int](length=HORNER_DEGREE + 1, fill=0)
        for m in range(HORNER_DEGREE + 1):
            s = (s * 1103515245 + 12345) & 0x7FFFFFFF
            a[m] = (s >> 8) & 1
            b[m] = (s >> 9) & 1
            t[x2 * h1 + h1 - 2 - m] = UInt8(a[m])
            t[N + x2 * h1 + h1 - 2 - m] = UInt8(b[m])
        for k in range(2 * HORNER_DEGREE + 1):
            var c = 0
            for i in range(HORNER_DEGREE + 1):
                if k - i >= 0 and k - i <= HORNER_DEGREE:
                    c += a[i] * b[k - i]
            t[2 * N + x2 * h1 + h1 - 2 - k] = UInt8(c)
    return t^
