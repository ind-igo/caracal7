"""The synthetic instance the tests and benches run before a real frontend exists: eight families
over ten columns, two permutation accumulators, and a trace that satisfies them."""

from caracal7.core.field import F2, f_add, f_mul
from caracal7.core.params import Params
from caracal7.relations.ir import FIX_E, KIND_PERM, KIND_LOOKUP, PUB
from caracal7.relations.statement import Statement, Term, BIT, BYTE, GATE_1, GATE_2, public_block
from caracal7.core.bytes import append_u32, set_u16

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
    """The PUB record of the one public column: m = 4, d2 = h2 / 4."""
    var b = List[UInt8](length=PUB, fill=0)
    set_u16(b, 0, SYNTHETIC_PUBLIC_M)
    set_u16(b, 2, p.h2() // SYNTHETIC_PUBLIC_M)
    return b^


def synthetic_public_block[p: Params]() raises -> List[UInt8]:
    """The block (d2, h1, 2) of the public column from its values on H."""
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    var vals = List[UInt8](length=p.N(), fill=0)
    for x2 in range(h2):
        for x1 in range(h1):
            vals[x2 * h1 + x1] = synthetic_public_value[p](x1, x2)
    return public_block[p](synthetic_statement(with_public=True).compile[p]().layout, 0, vals)


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
