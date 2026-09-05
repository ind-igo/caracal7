"""The synthetic instance the tests and benches run before a real frontend exists: eight families
over ten columns, two permutation accumulators, and a trace that satisfies them."""

from caracal7.field import f_add, f_mul
from caracal7.params import Params
from caracal7.ir import Families, NONE, NO_BASIS, FIX_ONE, FIX_E

def synthetic_families(columns_w: Int = 10, with_accumulator: Bool = True) raises -> Families:
    """Eight families over ten columns, satisfied by `synthetic_trace`, plus two permutation
    accumulators (c8, c9 are c0, c1 under one permutation of the grid; records of width 1 and 2)
    whose coordinate columns start the Z tree at global index columns_w. The families cover a linear entry, a quadratic entry, both gates,
    a within-chain shift, a cyclic shift, an axis-2 shift, a challenge and basis coefficient, a
    quadratic axis-1 transition, and the accumulator."""
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
    return f^


comptime SYNTHETIC_COLUMNS = 10
comptime SYNTHETIC_PERM = 17                            # c8[i] = c0[(17 i + 5) mod N]: coprime to every grid N


def synthetic_trace[p: Params](seed: Int) -> List[UInt8]:
    """Ten columns (column, x2, x1) satisfying `synthetic_families`."""
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    comptime N = p.N()
    var t = List[UInt8](length=SYNTHETIC_COLUMNS * N, fill=0)
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
    return t^
