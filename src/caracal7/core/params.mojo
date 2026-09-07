"""Params: every knob of the prover, as one comptime value (docs/design.md section 2)."""

from std.math import ceildiv, log2


@fieldwise_init
struct Params(TrivialRegisterPassable, Writable):
    var e: Int          # extension degree; 16 (spec section 1)
    var a1: Int         # h1 = 2^a1 * m1, 2 <= a1 <= 7 (spec 9.1)
    var m1: Int         # odd, m1 | 63
    var a2: Int
    var m2: Int
    var L0: Int         # level-1 subgroup order, L0 | 161280
    var m_cosets: Int   # 1, 2, or 4
    var leaf_bytes: Int # 1,024, one Blake3 chunk
    var tail_digits: Int    # binary digits folded per tail level
    var tail_clear_max: Int # E elements sent in the clear at the last level
    var lambda_bits: Int    # lambda' for the query count, 103 in the spec

    # ---- derived ----
    def h1(self) -> Int:
        return (1 << self.a1) * self.m1

    def h2(self) -> Int:
        return (1 << self.a2) * self.m2

    def N(self) -> Int:
        return self.h1() * self.h2()

    def L(self) -> Int:
        return self.m_cosets * self.L0

    def n_cw(self) -> Int:
        """Codewords per column: smallest power of two with N / (4 n_cw) <= L * rate; rate fixed by L."""
        # ponytail: rate is N/(4 L) at n_cw = 1; the split rule of spec 9.1 only matters
        # once a column exceeds the domain, which no milestone-1 profile does.
        return 1

    def rate(self) -> Float64:
        return Float64(self.N()) / Float64(4 * self.n_cw() * self.L())

    def queries(self) -> Int:
        """|S| = ceil(lambda' / log2(2 / (1 + rate))), spec section 9."""
        return Int(ceildiv(Float64(self.lambda_bits), log2(2.0 / (1.0 + self.rate()))))

    def leaf_columns_max(self) -> Int:
        """Columns one tree can hold under the one-chunk leaf rule: leaf_bytes / (4 n_cw)."""
        return self.leaf_bytes // (4 * self.n_cw())

    def check(self) raises:
        if self.e != 16:
            raise Error("e must be 16: field.mojo fixes E = F_(127^16)")
        if self.a1 < 2 or self.a1 > 7 or self.a2 < 2 or self.a2 > 7:
            raise Error("2 <= a_l <= 7")
        if 63 % self.m1 != 0 or 63 % self.m2 != 0:
            raise Error("m_l | 63")
        if 161280 % self.L0 != 0 or (self.L0 & (self.L0 - 1)) == 0:
            raise Error("L0 | 161280 with an odd part > 1 (the encoder has no scatter path without a radix stage)")
        if self.m_cosets != 1 and self.m_cosets != 2 and self.m_cosets != 4:
            raise Error("m_cosets in {1, 2, 4}")
        if self.N() > 4 * self.L():
            raise Error("message longer than the domain; codeword split not implemented")

    def write_to(self, mut w: Some[Writer]):
        w.write("Params(h1=", self.h1(), ", h2=", self.h2(), ", N=", self.N(),
                ", L=", self.L(), ", rate=", self.rate(), ", queries=", self.queries(), ")")


# Named profiles (docs/design.md section 2). A workload never picks its grid: it is compiled under one of
# these and pads the chains it does not use. New profiles are added here when a statement needs one.

# Reference: 72 x 32, one coset of 80,640. Tests run here; Keccak-256 of 128 B is one permutation on it.
comptime REFERENCE = Params(
    e=16, a1=3, m1=9, a2=5, m2=1, L0=80640, m_cosets=1, leaf_bytes=1024,
    tail_digits=3, tail_clear_max=2500, lambda_bits=103,
)

# Wide: 288 x 128 (Keccak-256 of 2048 B at 16 channels per row). The benches run here.
comptime WIDE = Params(
    e=16, a1=5, m1=9, a2=7, m2=1, L0=161280, m_cosets=1, leaf_bytes=1024,
    tail_digits=3, tail_clear_max=2500, lambda_bits=103,
)

# Client: 2016 x 576, the throughput proxy of the spec (N = 2^20 + 4096). Two ceilings stand between this
# profile and a run on a 16 GB machine: `n_cw` is 1 (the codeword split is not implemented), so level 1 is
# at rate 0.45 with 223 queries instead of the spec's n_cw = 2; and the allocate-once arena is 5.3 GB at 91
# W columns and 11.7 GB at 357 (the LDE and the W encode regions are 1 to 4 GB each). A level folds five
# digits so the first tail level (36,288 rows) fits the F4 domain at rate <= 1/16.
comptime CLIENT = Params(
    e=16, a1=5, m1=63, a2=6, m2=9, L0=161280, m_cosets=4, leaf_bytes=1024,
    tail_digits=5, tail_clear_max=2500, lambda_bits=103,
)
