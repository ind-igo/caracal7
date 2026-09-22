"""Params: every knob of the prover, as one comptime value (docs/design.md section 2).

A `Profile` holds the deployment knobs (field, security, tail, leaf); `Profile.grid(rows_per_chain, chains)`
derives the Params for a statement: each axis rounded up to the smallest legal size, the level-1 code domain
by `domain_for` at the profile's `rate_inv` with the fewest cosets; the tail levels use `tail_rate_inv` (8 since 2026-09-14).
The grid belongs to the statement, the profile to the target.
There is one profile, `CLIENT`; a second one appears with a second target (the VM), not with a second grid."""

from std.math import ceildiv, log2, sqrt

from core.field import E_BYTES

comptime H4_ORDER = 161280          # largest smooth subgroup of F4*; every code domain is m cosets of a divisor
comptime RATE_INV = 32              # rate rule of spec 9.5 for the tail levels: the smallest domain at rate <= 1/32 ...
comptime RATE_MIN_INV = 16          # ... or the largest domain (4 x 161280) if that still gives rate <= 1/16
comptime REGIME_UNIQUE = 0          # proximity radius (1 - rate) / 2: proven (BCIKS20 1.2 / 1.7, the ledger's regime)
comptime REGIME_CAPACITY = 1        # radius 1 - rate - eta: the up-to-capacity conjecture, unproven (docs/soundness.md)
comptime REGIME_JOHNSON = 2         # radius 1 - sqrt(rate) - eta; the ledger and remaining proof conditions are in docs/soundness.md


def miss_probability(rate: Float64, regime: Int, eta_inv: Int) -> Float64:
    """The per-query miss probability of the regime: (1 + rate) / 2 at the unique-decoding radius, rate + eta
    under the capacity conjecture, sqrt(rate) + eta at the Johnson radius (spec section 12)."""
    if regime == REGIME_UNIQUE:
        return (1.0 + rate) / 2.0
    if regime == REGIME_JOHNSON:
        return sqrt(rate) + 1.0 / Float64(eta_inv)
    return rate + 1.0 / Float64(eta_inv)


def query_count(lambda_bits: Int, rate: Float64, regime: Int, eta_inv: Int) -> Int:
    """|S| = ceil(lambda' / -log2(miss)) at the regime's per-query miss probability (miss_probability)."""
    var miss = miss_probability(rate, regime, eta_inv)
    return Int(ceildiv(Float64(lambda_bits), -log2(miss)))


def domain_for(rows: Int, rate_inv: Int = RATE_INV, fewest_cosets: Bool = False) -> Tuple[Int, Int]:
    """(L, cosets) for `rows` symbols per column: the smallest m cosets (m in 1, 2, 4) of a divisor of H4_ORDER
    with an odd part (the encoder scatters from an odd-radix stage) at rate <= 1/rate_inv; else the largest
    domain if it is at rate <= 1/RATE_MIN_INV; else (0, 0), the codeword split. With `fewest_cosets` the
    smallest m that has a domain wins before the size does (level 1: one coset of 161,280 beat four of
    23,040 on prove time, proof size and verify time at the ECDSA grid)."""
    var best = 0
    var cosets = 0
    for m in [1, 2, 4]:
        if fewest_cosets and best != 0:
            break
        for k in range(10):
            for odd in [3, 5, 7, 9, 15, 21, 35, 45, 63, 105, 315]:
                var d = (1 << k) * odd
                if m * d >= rate_inv * rows and (best == 0 or m * d < best):
                    best = m * d
                    cosets = m
    if best == 0 and 4 * H4_ORDER >= RATE_MIN_INV * rows:
        return (4 * H4_ORDER, 4)
    return (best, cosets)


def _axis(target: Int) -> Tuple[Int, Int]:
    """(a, m): the smallest legal 2^a m >= target (2 <= a <= 7, m | 63); (0, 1) if none."""
    var a_best = 0
    var m_best = 1
    var best = 0
    for a in range(2, 8):
        for m in [1, 3, 7, 9, 21, 63]:
            var h = (1 << a) * m
            if h >= target and (best == 0 or h < best):
                best = h
                a_best = a
                m_best = m
    return (a_best, m_best)


@fieldwise_init
struct Profile(TrivialRegisterPassable, Writable):
    var e: Int              # extension degree; E_BYTES = 20 (spec section 1)
    var leaf_bytes: Int     # 1,024, one Blake3 chunk
    var tail_digits: Int    # binary digits folded per tail level
    var tail_clear_max: Int # E elements sent in the clear at the last level
    var lambda_bits: Int    # lambda' for the query count per level: 112 (the spec's 103 until 2026-09-13)
    var grind_bits: Int     # proof-of-work bits on every query seed; the per-level query target is lambda_bits - grind_bits
    var regime: Int         # REGIME_UNIQUE (proven) or REGIME_CAPACITY (conjectured): the per-query miss probability
    var eta_inv: Int        # the capacity regime's slack eta = 1 / eta_inv
    var rate_inv: Int       # level-1 domain rule: the fewest cosets, then the smallest domain, at rate <= 1/rate_inv
    var tail_rate_inv: Int  # tail domain rule: the smallest domain at rate <= 1/tail_rate_inv (spec 9.5's RATE_INV = 32 for the unique regime)

    def grid(self, rows_per_chain: Int, chains: Int) -> Params:
        """The Params of a statement with `chains` chains of `rows_per_chain` rows, padded up to legal sizes.
        Evaluates at compile time: `comptime p = CLIENT.grid(72, 32)`. A grid the level-1 domain cannot hold
        at n_cw = 1 gets L0 = 0 and fails `check()`."""
        var ax1 = _axis(rows_per_chain)
        var ax2 = _axis(chains)
        var n = (1 << ax1[0]) * ax1[1] * (1 << ax2[0]) * ax2[1]
        # the codeword split (spec 9.1): the smallest power of two n_cw whose N / (4 n_cw) symbols a domain holds
        var cw = 1
        var dom = domain_for(n // 4, self.rate_inv, fewest_cosets=True)
        while dom[1] == 0 and cw < (1 << (ax1[0] + ax2[0] - 2)):
            cw *= 2
            dom = domain_for(n // (4 * cw), self.rate_inv, fewest_cosets=True)
        return Params(e=self.e, a1=ax1[0], m1=ax1[1], a2=ax2[0], m2=ax2[1],
                      L0=dom[0] // dom[1] if dom[1] > 0 else 0, m_cosets=dom[1],
                      leaf_bytes=self.leaf_bytes, tail_digits=self.tail_digits,
                      tail_clear_max=self.tail_clear_max, lambda_bits=self.lambda_bits, grind_bits=self.grind_bits,
                      regime=self.regime, eta_inv=self.eta_inv, tail_rate_inv=self.tail_rate_inv, codewords=cw)


# The client-side target: 112-bit queries per level (four levels sum to about 2^-110, next to the field terms
# at e = 20, docs/soundness.md; 103 until 2026-09-13), three-digit tail folds, the level-1 domain at rate <= 1/4
# (the query formula is sound at any rate below the 1/4 distance bound; one coset does a quarter of the encode
# and Merkle work for a 3.6% larger proof at the ECDSA grid), fold while digits remain
# (the tensor verifier's clear check costs units x clear length), 20 bits of grinding on every query seed (the
# prover spends 2^20 hashes per level, milliseconds on the GPU, and samples 92-bit queries; docs/soundness.md).
comptime CLIENT = Profile(e=E_BYTES, leaf_bytes=1024, tail_digits=3, tail_clear_max=0, lambda_bits=112, grind_bits=20,
                          regime=REGIME_JOHNSON, eta_inv=16, rate_inv=4, tail_rate_inv=8)


@fieldwise_init
struct Params(TrivialRegisterPassable, Writable):
    var e: Int          # extension degree; E_BYTES = 20 (spec section 1)
    var a1: Int         # h1 = 2^a1 * m1, 2 <= a1 <= 7 (spec 9.1)
    var m1: Int         # odd, m1 | 63
    var a2: Int
    var m2: Int
    var L0: Int         # level-1 subgroup order, L0 | 161280
    var m_cosets: Int   # 1, 2, or 4
    var leaf_bytes: Int # 1,024, one Blake3 chunk
    var tail_digits: Int    # binary digits folded per tail level
    var tail_clear_max: Int # E elements sent in the clear at the last level
    var lambda_bits: Int    # lambda' for the query count per level: 112 (the spec's 103 until 2026-09-13)
    var grind_bits: Int     # proof-of-work bits on every query seed (transcript.grind); 0 disables the nonce
    var regime: Int         # REGIME_UNIQUE or REGIME_CAPACITY (query_count)
    var eta_inv: Int        # eta = 1 / eta_inv in the capacity regime
    var tail_rate_inv: Int  # tail domain rule (tail_schedule): the smallest domain at rate <= 1/tail_rate_inv
    var codewords: Int      # n_cw: codewords per column, a power of two (spec 9.1); the top log2 binary digits of the packed index

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
        """Codewords per column (`codewords`): the smallest power of two whose N / (4 n_cw) symbols the level-1
        domain holds at the profile's rate (Profile.grid). A codeword is the packed indices with one value of
        the top log2 n_cw binary digits; the odd digit and the low binary digits stay inside it."""
        return self.codewords

    def K(self) -> Int:
        """Message length of one codeword: N / (4 n_cw) F4 symbols."""
        return self.N() // (4 * self.codewords)

    def rate(self) -> Float64:
        return Float64(self.N()) / Float64(4 * self.n_cw() * self.L())

    def queries(self) -> Int:
        """|S| for the level-1 rate at the per-level target lambda' - grind_bits (each level's nonce costs the
        prover 2^grind_bits hashes per attempt) in the profile's regime (query_count)."""
        return query_count(self.lambda_bits - self.grind_bits, self.rate(), self.regime, self.eta_inv)

    def leaf_columns_max(self) -> Int:
        """Columns one tree can hold under the one-chunk leaf rule: leaf_bytes / (4 n_cw)."""
        return self.leaf_bytes // (4 * self.n_cw())

    def check(self) raises:
        if self.L0 == 0:
            raise Error("grid needs a larger codeword split: no level-1 domain holds N / (4 n_cw) symbols at the profile's rate")
        if self.codewords < 1 or (self.codewords & (self.codewords - 1)) != 0 or self.codewords > (1 << (self.a1 + self.a2 - 2)):
            raise Error("codewords is a power of two of at most 2^(a1 + a2 - 2) (the binary digits of the packed index)")
        if self.e != E_BYTES:
            raise Error("e must be E_BYTES: field.mojo fixes E = F_(127^E_BYTES) per build (-D E16)")
        if self.grind_bits < 0 or self.grind_bits > 26 or self.grind_bits >= self.lambda_bits:
            raise Error("grind_bits in [0, 26] and below lambda_bits (u32 nonces: 2^grind_bits tries on average)")
        if self.regime < REGIME_UNIQUE or self.regime > REGIME_JOHNSON or self.eta_inv < 2:
            raise Error("regime is REGIME_UNIQUE, REGIME_CAPACITY or REGIME_JOHNSON; eta_inv >= 2")
        if self.tail_rate_inv < 2:
            raise Error("tail_rate_inv >= 2")
        if miss_probability(self.rate(), self.regime, self.eta_inv) >= 1.0:
            raise Error("the regime's per-query miss probability must be below 1 at the level-1 rate (eta too large)")
        if self.a1 < 2 or self.a1 > 7 or self.a2 < 2 or self.a2 > 7:
            raise Error("2 <= a_l <= 7")
        if 63 % self.m1 != 0 or 63 % self.m2 != 0:
            raise Error("m_l | 63")
        if 161280 % self.L0 != 0 or (self.L0 & (self.L0 - 1)) == 0:
            raise Error("L0 | 161280 with an odd part > 1 (the encoder has no scatter path without a radix stage)")
        if self.m_cosets != 1 and self.m_cosets != 2 and self.m_cosets != 4:
            raise Error("m_cosets in {1, 2, 4}")
        if self.K() > self.L():
            raise Error("message longer than the domain: a larger codeword split")

    def write_to(self, mut w: Some[Writer]):
        w.write("Params(h1=", self.h1(), ", h2=", self.h2(), ", N=", self.N(), ", n_cw=", self.codewords,
                ", L=", self.L(), ", rate=", self.rate(), ", queries=", self.queries(), ")")
