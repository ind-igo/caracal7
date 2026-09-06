"""The synthetic instance the tests and benches run before a real frontend exists: eight families
over ten columns, two permutation accumulators, and a trace that satisfies them."""

from caracal7.core.field import F2, f_add, f_mul, f_pow, ext_mul, ext_pow
from caracal7.core.params import Params
from caracal7.core.tables import Domains
from caracal7.relations.ir import Families, NONE, NO_BASIS, FIX_ONE, FIX_E, PUB, RES
from caracal7.core.bytes import append_u32, set_u16

def synthetic_families(columns_w: Int = 10, with_accumulator: Bool = True, with_lookup: Bool = False, with_public: Bool = False) raises -> Families:
    """Eight families over ten columns, satisfied by `synthetic_trace`, plus two permutation
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
        f.accumulator(9, columns_w + 16, [0, 1], [8, 9])   # width-2 records (c0, c1) against (c8, c9): the basis products b_t b_j
    if with_lookup:                                        # (c10, c11) in `synthetic_table`; the prover sorts them into (c12, c13)
        f.lookup(10, columns_w + (32 if with_accumulator else 0), [10, 11], [12, 13], 0)
    if with_public:                                        # c_{w-1} - c0 pub: the public column is the first past W and Z
        f.add(11, 1, columns_w - 1)
        f.add(11, 126, 0, col_b=columns_w + (32 if with_accumulator else 0))
    return f^


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
    """The block (d2, h1, 2) of the public column: the interpolant of its values on H, rows k2 = m j
    (the periodic function has no other rows; `interpolate_grid` computes them and they are zero)."""
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    var vals = List[UInt8](length=p.N(), fill=0)
    for x2 in range(h2):
        for x1 in range(h1):
            vals[x2 * h1 + x1] = synthetic_public_value[p](x1, x2)
    var full = interpolate_grid[p](vals)
    var d2 = h2 // SYNTHETIC_PUBLIC_M
    var block = List[UInt8](capacity=d2 * h1 * 2)
    for j in range(d2):
        for t in range(h1 * 2):
            block.append(full[(SYNTHETIC_PUBLIC_M * j * h1) * 2 + t])
    return block^


def synthetic_restriction[p: Params](trace: List[UInt8]) raises -> Tuple[List[UInt8], List[UInt8]]:
    """Restrict c1 on the last chain (X2 = e2) to its own interpolant: the RES record (count h1) and the h1
    F2 coefficients of that line."""
    comptime h1 = p.h1()
    var r = List[UInt8](length=RES, fill=0)
    set_u16(r, 0, 1)
    set_u16(r, 2, FIX_E)
    set_u16(r, 4, h1)
    var line = List[UInt8](capacity=h1)
    for x1 in range(h1):
        line.append(trace[p.N() + (p.h2() - 1) * h1 + x1])
    var d = Domains.__init__[p]()
    return (r^, interpolate_line(line, d.omega1, h1))


def interpolate_line(vals: List[UInt8], omega: F2, h: Int) raises -> List[UInt8]:
    """h F values on <omega> -> h F2 monomial coefficients (2 bytes each): c_k = h^-1 sum_x v(x) omega^(-k x)."""
    var inv_h = f_pow(SIMD[DType.uint8, 1](UInt8(h % 127)), 125)
    var out = List[UInt8](capacity=h * 2)
    for k in range(h):
        var acc = F2(0)
        for x in range(h):
            acc = f_add(acc, ext_mul[1](F2(vals[x], 0), ext_pow[1](omega, (h - (k * x) % h) % h)))
        var c = ext_mul[1](acc, F2(inv_h[0], 0))
        out.append(c[0])
        out.append(c[1])
    return out^


def interpolate_grid[p: Params](vals: List[UInt8]) raises -> List[UInt8]:
    """N F values (x2, x1) on H -> (k2, k1, 2) F2 coefficients, axis 1 then axis 2 (host, O(N (h1 + h2)))."""
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    var d = Domains.__init__[p]()
    var inv_h2 = f_pow(SIMD[DType.uint8, 1](UInt8(h2 % 127)), 125)
    var ctmp = List[UInt8](length=p.N() * 2, fill=0)       # (x2, k1, 2)
    for x2 in range(h2):
        var line = List[UInt8](capacity=h1)
        for x1 in range(h1):
            line.append(vals[x2 * h1 + x1])
        var c = interpolate_line(line, d.omega1, h1)
        for t in range(h1 * 2):
            ctmp[x2 * h1 * 2 + t] = c[t]
    var out = List[UInt8](length=p.N() * 2, fill=0)         # (k2, k1, 2)
    for k1 in range(h1):
        for k2 in range(h2):
            var acc = F2(0)
            for x2 in range(h2):
                acc = f_add(acc, ext_mul[1](F2(ctmp[(x2 * h1 + k1) * 2], ctmp[(x2 * h1 + k1) * 2 + 1]),
                                            ext_pow[1](d.omega2, (h2 - (k2 * x2) % h2) % h2)))
            var c = ext_mul[1](acc, F2(inv_h2[0], 0))
            out[(k2 * h1 + k1) * 2] = c[0]
            out[(k2 * h1 + k1) * 2 + 1] = c[1]
    return out^


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


def synthetic_trace[p: Params](seed: Int, with_lookup: Bool = False, with_public: Bool = False) -> List[UInt8]:
    """Ten columns (column, x2, x1) satisfying `synthetic_families`; with the lookup, fourteen: the records in
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
            t[8 * N + i] = t[(SYNTHETIC_PERM * i + 5) % N]                                             # c8 = c0 permuted
            t[9 * N + i] = t[N + (SYNTHETIC_PERM * i + 5) % N]                                         # c9 = c1 under the same permutation
            if with_public:
                t[10 * N + i] = f_mul(c0, SIMD[DType.uint8, 1](synthetic_public_value[p](x1, x2)))[0]
    return t^
