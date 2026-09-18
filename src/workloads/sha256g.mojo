"""SHA-256 as a row group on the 144-row chain (docs/decisions.md "SHA-256 group"): four rounds per chain,
round t = 4 c + q of the group's chain c in slot q, whose 32 rows BASE + 32 q + z hold bit 31 - z of every
word (rows 0..14 and 143 idle). The families of `sha256.mojo` (one round per 32-row chain) carry over with
the same columns; what changes is the geometry:

- A rotation is no longer a cyclic read of the chain: a helper column holds ROTR r of its word, its family
  reading the row r above (k1 = 144 - r) on the slot rows z >= r (the public row mask `ge{r}`) and the row
  32 - r below on the rest (the slot mask `qall` less `ge{r}`). Every rotation the round reads gets a helper;
  the two shifts are the masked upper read alone (`g0m`, `g1m`, as before). The additions ripple down the
  rows as before, the carry into a slot's low bit cut by `nb` (slot rows but the last).
- The state moves to the next round inside the chain (`qlt`: slots 0..2 read 32 rows down) and across the
  chain edge (`q3`: slot 3 reads the next chain's slot 0). The schedule reads w[t + 1], w[t + 14], w[t + 9]
  through helper columns placed the same way (slot masks pick the read for each slot), so the ws sum is
  same-chain. w[t] = ws[t - 16] is a backward read four chains up, off on the message rounds (`s16`, zero on
  the group's first four chains where the mask leaves w free anyway). A block end adds the block's input
  state, read 15 chains up through the helpers `bx*` (`end` on the block's last slot).
- The message is witness: w on the message rounds is free but the padding rows (`pfix`, values `pv`) and the
  IV (`start` on chain 0's slot 0, the eight state bits folded into two nibble columns `iv0`, `iv1`).
- A family reading another chain takes the group's mask for its shifts (statement.mojo), so a helper is free
  on the chains near the group's edge where its read leaves the group: the last chain of the group is idle
  (`chains` > 16 blocks) and the schedule helpers of a block's last rounds feed nothing.

Fingerprints (the wires): the digest is an..hn of slot 3 on the last live chain; a Horner accumulator with
scale zeta ingesting them there (`dsel`) with coefficient zeta^(32 (7 - i) + digest_shift) holds
zeta^digest_shift times sum_k H_k zeta^k (H the digest as a 256-bit integer), the fingerprint mulmod's
`plain_fingerprint` gives a 256-bit value (digest_shift = 0 wires it to RSA's message limb). A digest
embedded in this group's message at byte offset `embed` spans up to three message chains cA, cB, cC (a
block's message words sit on its first four chains, so the next message chain is one or 13 chains on; s0 =
8 embed mod 128 bits into cA); transport columns t1 = w one chain up, t2 = t1 one chain up, u1 = w 13 chains
up, v = u1 one chain up bring the three segments onto chain cC, where the `embedded` terms ingest them with
coefficients zeta^256 (cA's rows >= s0: t2 or v), zeta^128 (cB: t1 or u1) and 1 (w, rows < s0), the public
masks picking the transport: the fingerprint times zeta^(128 - s0), so the producer's group takes
digest_shift = 128 - s0. One accumulator may take the terms of several groups (each group's selector is
zero off it): the caller makes the accumulators and wires their slots."""

from core.params import Params
from relations.ir import CHAL_MUL
from relations.statement import Statement, Layout, Term, Read, BIT, BYTE
from workloads.sha256 import ROUNDS, WORDS, STATE, SHA_COLUMNS, k_const, iv, rotr, blocks, padded_words, chain_words, column_names, live_chains

comptime ROWS = 144
comptime SLOT = 32
comptime SLOTS = 4
comptime BASE = 15              # row of slot 0's top bit: bit k of slot 3 sits k rows above the last row, which no accumulator ingests
comptime STREAM = SLOT * SLOTS  # message bits per chain
comptime ZETA = 2               # the fingerprint point (mulmod.ZETA)
comptime NIBBLES = 2            # the IV as two 4-bit columns


def _rots() -> List[Int]:
    """The rotation and shift amounts (the row masks ge{r})."""
    return [2, 13, 22, 6, 11, 25, 7, 18, 17, 19, 3, 10]


@fieldwise_init
struct _Rot(Copyable, Movable):
    var name: String            # the helper column
    var word: String            # the word it rotates
    var r: Int


@fieldwise_init
struct _Pub(Copyable, Movable):
    var name: String
    var shifts: List[Int]


def _helpers() -> List[_Rot]:
    """The rotations the round reads."""
    var v = List[_Rot]()
    v.append(_Rot("ra2", "a", 2))
    v.append(_Rot("ra13", "a", 13))
    v.append(_Rot("ra22", "a", 22))
    v.append(_Rot("re6", "e", 6))
    v.append(_Rot("re11", "e", 11))
    v.append(_Rot("re25", "e", 25))
    v.append(_Rot("rw7", "w1", 7))
    v.append(_Rot("rw18", "w1", 18))
    v.append(_Rot("rv17", "w14", 17))
    v.append(_Rot("rv19", "w14", 19))
    return v^


struct Zeta(Copyable, Movable):
    """Derived powers of the fingerprint point, memoized (the derivation table holds 255 elements)."""
    var cache: Dict[Int, Int]

    def __init__(out self, zeta: Int = ZETA):
        self.cache = {0: -1, 1: zeta}

    def power(mut self, mut st: Statement, n: Int) raises -> Int:
        if n in self.cache:
            return self.cache[n]
        var v: Int
        if n % 2 == 0:
            var h = self.power(st, n // 2)
            v = st.derived(CHAL_MUL, h, h)
        else:
            v = st.derived(CHAL_MUL, self.power(st, n - 1), self.cache[1])
        self.cache[n] = v
        return v


@fieldwise_init
struct ShaGroup(Copyable, Movable):
    var name: String
    var base: Int               # the group's first chain
    var chains: Int
    var embed: Int              # byte offset of an embedded digest in the message, -1 none
    var digest: List[Term]      # ingest terms (Horner, scale zeta) of the digest fingerprint times zeta^digest_shift, on the last live chain
    var embedded: List[Term]    # ingest terms of the embedded digest's fingerprint times zeta^(128 - s0), on chain `embed_chain`

    def digest_chain(self, message: List[UInt8]) -> Int:
        """The group chain holding the digest (slot 3)."""
        return ROUNDS // SLOTS * blocks(message) - 1

    def segments(self) -> Tuple[Int, Int, Int, Int]:
        """(cA, cB, cC, s0) of the embedded digest: its first chain, the next two message chains, and its
        bit offset into cA's 128-bit window."""
        var w0 = 8 * self.embed // SLOT
        var ca = _chain_of_word(w0)
        var cb = _next_message_chain(ca)
        return (ca, cb, _next_message_chain(cb), 8 * self.embed % STREAM)

    def embed_chain(self) -> Int:
        return self.segments()[2]


def _chain_of_word(w: Int) -> Int:
    """The group chain holding padded-message word w (block w / 16, its message words on the block's first four chains)."""
    return ROUNDS // SLOTS * (w // 16) + w % 16 // SLOTS


def _next_message_chain(c: Int) -> Int:
    return c + 1 if c % (ROUNDS // SLOTS) < SLOTS - 1 else c + ROUNDS // SLOTS - SLOTS + 1


def _stream_index(t: Int, z: Int) -> Int:
    """The padded-message bit at round t's row z, -1 off the message rounds."""
    return 512 * (t // ROUNDS) + SLOT * (t % ROUNDS) + z if t % ROUNDS < 16 else -1


def _pubs(h2: Int, embed: Bool) -> List[_Pub]:
    """The group's public columns after the selector and masks, in declaration order, with their shift sets."""
    var v = List[_Pub]()
    v.append(_Pub("qall", [0]))
    for r in _rots():
        v.append(_Pub("ge" + String(r), [0]))
    v.append(_Pub("nb", [0]))
    v.append(_Pub("qlt", [0, 1]))
    v.append(_Pub("q3a", [0, 1]))
    v.append(_Pub("q01", [0, 3, 4]))
    v.append(_Pub("q23", [0, 3, 4]))
    v.append(_Pub("q012", [0, 2, 3]))
    v.append(_Pub("q3b", [0, 2, 3]))
    v.append(_Pub("lv", [0]))
    v.append(_Pub("end", [0, h2 - 15]))
    v.append(_Pub("K", [0]))
    v.append(_Pub("s16", [0, h2 - 4]))
    v.append(_Pub("pfix", [0]))
    v.append(_Pub("pv", [0]))
    v.append(_Pub("start", [0]))
    v.append(_Pub("iv0", [0]))
    v.append(_Pub("iv1", [0]))
    v.append(_Pub("dsel", [0]))
    if embed:
        for n in ["ea2", "eav", "eb1", "ebu", "ec"]:
            v.append(_Pub(n, [0]))
    return v^


def _masks(h2: Int, embed: Bool) -> List[_Pub]:
    var v = List[_Pub]()
    v.append(_Pub("m01", [0, 1]))
    v.append(_Pub("m034", [0, 3, 4]))
    v.append(_Pub("m023", [0, 2, 3]))
    v.append(_Pub("mw", [0, h2 - 4]))
    v.append(_Pub("mb", [0, h2 - 15]))
    if embed:
        v.append(_Pub("me", [0, h2 - 1]))
        v.append(_Pub("mu", [0, h2 - 13]))
    return v^


def _xor(mut st: Statement, P: String, result: String, u: Read, v: Read) raises:
    """result = u xor v; v is the quadratic factor (the exclusive column)."""
    st.family(P + "=" + result, [Term(1, Read(P + result, 0, 0)), Term(-1, u.copy()), Term(-1, v.copy()), Term(2, u.copy(), v.copy())])


def _sum(mut st: Statement, P: String, result: String, var terms: List[Term], carries: List[String]) raises:
    """addends + carry in (from the row below, cut on a slot's last row by nb) = result + 2 carry out."""
    for k in range(len(carries)):
        terms.append(Term(1 << k, Read(P + "nb", 0, 0), Read(P + carries[k], 1, 0)))
    terms.append(Term(-1, Read(P + result, 0, 0)))
    for k in range(len(carries)):
        terms.append(Term(-(2 << k), Read(P + carries[k], 0, 0)))
    st.family(P + "=" + result, terms^)


def sha256_group(mut st: Statement, g: String, chains: Int, h2: Int, mut zeta: Zeta, digest_shift: Int, embed: Int = -1) raises -> ShaGroup:
    """Declare group `g` of `chains` chains (more than 16 per block of the longest message: the last chain is
    idle) with its columns, public columns and families; the columns are named g.name. Returns the ingest
    terms for the caller's accumulators."""
    if chains < 2 or chains > h2 or h2 < 16:
        raise Error("a SHA-256 group has 2 to h2 chains: " + g)
    var pre = ShaGroup(g, 0, chains, embed, List[Term](), List[Term]())
    if embed >= 0 and pre.embed_chain() >= chains:
        raise Error("the embedded digest's third chain lies outside the group: " + g)
    if st.chains_declared() + chains + 15 > h2:          # masks wrap mod h2: a read 15 chains up from the group's first chains must land off the group
        raise Error("a SHA-256 group ends at least 15 chains before the grid's end: " + g)
    var P = g + "."
    st.pub(P + "sel", 1)
    var base = st.group(g, chains, P + "sel")
    for m in _masks(h2, embed >= 0):
        st.mask(g, P + m.name, m.shifts)
    for u in _pubs(h2, embed >= 0):
        st.pub(P + u.name, 1, group=g, shifts=u.shifts)
    for n in column_names():
        st.col(P + n, BIT, group=g)
    var helpers: List[String] = ["w1", "w14", "w9"]
    for h in _helpers():
        helpers.append(h.name)
    for i in range(WORDS):
        helpers.append("bx" + String(STATE[byte=i]))
    if embed >= 0:
        helpers.extend(["t1", "t2", "u1", "v"])
    for n in helpers:
        st.col(P + n, BYTE, group=g)
    # the schedule's shifted words: one helper per read, placed by slot
    st.family(P + "=w1", [Term(1, Read(P + "w1", 0, 0)), Term(-1, Read(P + "qlt", 0, 0), Read(P + "w", SLOT, 0)),
                          Term(-1, Read(P + "q3a", 0, 0), Read(P + "w", ROWS - 3 * SLOT, 1))])
    st.family(P + "=w14", [Term(1, Read(P + "w14", 0, 0)), Term(-1, Read(P + "q01", 0, 0), Read(P + "w", 2 * SLOT, 3)),
                           Term(-1, Read(P + "q23", 0, 0), Read(P + "w", ROWS - 2 * SLOT, 4))])
    st.family(P + "=w9", [Term(1, Read(P + "w9", 0, 0)), Term(-1, Read(P + "q012", 0, 0), Read(P + "w", SLOT, 2)),
                          Term(-1, Read(P + "q3b", 0, 0), Read(P + "w", ROWS - 3 * SLOT, 3))])
    # rotations: the row r above on the slot rows z >= r, the row 32 - r below on the others
    for h in _helpers():
        var ge = Read(P + "ge" + String(h.r), 0, 0)
        st.family(P + "=" + h.name, [Term(1, Read(P + h.name, 0, 0)), Term(-1, ge.copy(), Read(P + h.word, ROWS - h.r, 0)),
                                     Term(-1, Read(P + "qall", 0, 0), Read(P + h.word, SLOT - h.r, 0)), Term(1, ge.copy(), Read(P + h.word, SLOT - h.r, 0))])
    st.family(P + "=g0m", [Term(1, Read(P + "g0m", 0, 0)), Term(-1, Read(P + "ge3", 0, 0), Read(P + "w1", ROWS - 3, 0))])
    st.family(P + "=g1m", [Term(1, Read(P + "g1m", 0, 0)), Term(-1, Read(P + "ge10", 0, 0), Read(P + "w14", ROWS - 10, 0))])
    _xor(st, P, "s0x", Read(P + "ra2", 0, 0), Read(P + "ra13", 0, 0))
    _xor(st, P, "s0", Read(P + "s0x", 0, 0), Read(P + "ra22", 0, 0))
    _xor(st, P, "s1x", Read(P + "re6", 0, 0), Read(P + "re11", 0, 0))
    _xor(st, P, "s1", Read(P + "s1x", 0, 0), Read(P + "re25", 0, 0))
    _xor(st, P, "ab", Read(P + "a", 0, 0), Read(P + "b", 0, 0))
    st.family(P + "=mj", [Term(1, Read(P + "mj", 0, 0)), Term(-1, Read(P + "a", 0, 0), Read(P + "b", 0, 0)), Term(-1, Read(P + "c", 0, 0), Read(P + "ab", 0, 0))])
    st.family(P + "=ch", [Term(1, Read(P + "ch", 0, 0)), Term(-1, Read(P + "f", 0, 0), Read(P + "e", 0, 0)), Term(-1, Read(P + "g", 0, 0)), Term(1, Read(P + "g", 0, 0), Read(P + "e", 0, 0))])
    _xor(st, P, "g0x", Read(P + "rw7", 0, 0), Read(P + "rw18", 0, 0))
    _xor(st, P, "g1x", Read(P + "rv17", 0, 0), Read(P + "rv19", 0, 0))
    # ws = sigma1(w[t + 14]) + w[t + 9] + sigma0(w[t + 1]) + w[t]
    _sum(st, P, "ws", [Term(1, Read(P + "g1x", 0, 0)), Term(1, Read(P + "g1m", 0, 0)), Term(-2, Read(P + "g1x", 0, 0), Read(P + "g1m", 0, 0)),
                       Term(1, Read(P + "w9", 0, 0)),
                       Term(1, Read(P + "g0x", 0, 0)), Term(1, Read(P + "g0m", 0, 0)), Term(-2, Read(P + "g0x", 0, 0), Read(P + "g0m", 0, 0)),
                       Term(1, Read(P + "w", 0, 0))], ["wc0", "wc1"])
    # w[t] = ws[t - 16] off the message rounds: (1 - s16) (w - ws four chains up) = 0
    st.family(P + "=w", [Term(1, Read(P + "w", 0, 0)), Term(-1, Read(P + "ws", 0, h2 - 4)),
                         Term(-1, Read(P + "s16", 0, 0), Read(P + "w", 0, 0)), Term(1, Read(P + "s16", 0, 0), Read(P + "ws", 0, h2 - 4))])
    # the padding and the IV
    st.family(P + "=pad", [Term(1, Read(P + "pfix", 0, 0), Read(P + "w", 0, 0)), Term(-1, Read(P + "pv", 0, 0))])
    for k in range(NIBBLES):
        var t = List[Term]()
        for i in range(4):
            t.append(Term(1 << i, Read(P + "start", 0, 0), Read(P + String(STATE[byte=4 * k + i]), 0, 0)))
        t.append(Term(-1, Read(P + "iv" + String(k), 0, 0)))
        st.family(P + "=iv" + String(k), t^)
    # the block's input state, 15 chains up (slot 0), on the block's last slot
    for i in range(WORDS):
        var x = String(STATE[byte=i])
        st.family(P + "=bx" + x, [Term(1, Read(P + "bx" + x, 0, 0)), Term(-1, Read(P + "end", 0, 0), Read(P + x, ROWS - 3 * SLOT, h2 - 15))])
    # next state: lv masks the round (idle rounds hold the state), end adds the block's input state
    var an: List[Term] = [Term(1, Read(P + "K", 0, 0)), Term(1, Read(P + "a", 0, 0)), Term(-1, Read(P + "lv", 0, 0), Read(P + "a", 0, 0)),
                          Term(1, Read(P + "end", 0, 0), Read(P + "bxa", 0, 0))]
    for n in ["h", "s1", "ch", "w", "s0", "mj"]:
        an.append(Term(1, Read(P + "lv", 0, 0), Read(P + n, 0, 0)))
    _sum(st, P, "an", an^, ["ac0", "ac1", "ac2"])
    var en: List[Term] = [Term(1, Read(P + "K", 0, 0)), Term(1, Read(P + "e", 0, 0)), Term(-1, Read(P + "lv", 0, 0), Read(P + "e", 0, 0)),
                          Term(1, Read(P + "end", 0, 0), Read(P + "bxe", 0, 0))]
    for n in ["d", "h", "s1", "ch", "w"]:
        en.append(Term(1, Read(P + "lv", 0, 0), Read(P + n, 0, 0)))
    _sum(st, P, "en", en^, ["ec0", "ec1", "ec2"])
    for i in range(1, WORDS):
        if i == 4:
            continue
        var x = String(STATE[byte=i])
        var src = String(STATE[byte=i - 1])
        _sum(st, P, x + "n", [Term(1, Read(P + "lv", 0, 0), Read(P + src, 0, 0)), Term(1, Read(P + x, 0, 0)), Term(-1, Read(P + "lv", 0, 0), Read(P + x, 0, 0)),
                              Term(1, Read(P + "end", 0, 0), Read(P + "bx" + x, 0, 0))], [x + "c"])
    # the state enters the next round: 32 rows down inside the chain, the next chain's slot 0 from slot 3
    for i in range(WORDS):
        var x = String(STATE[byte=i])
        st.family(P + "cp" + x, [Term(1, Read(P + "qlt", 0, 0), Read(P + x, SLOT, 0)), Term(-1, Read(P + "qlt", 0, 0), Read(P + x + "n", 0, 0))])
        st.family(P + "cp3" + x, [Term(1, Read(P + "q3a", 0, 0), Read(P + x, ROWS - 3 * SLOT, 1)), Term(-1, Read(P + "q3a", 0, 0), Read(P + x + "n", 0, 0))])
    var digest = List[Term]()
    for i in range(WORDS):
        digest.append(Term(1, Read(P + String(STATE[byte=i]) + "n", 0, 0), Read(P + "dsel", 0, 0), chal=zeta.power(st, SLOT * (WORDS - 1 - i) + digest_shift)))
    var embedded = List[Term]()
    if embed >= 0:
        st.family(P + "=t1", [Term(1, Read(P + "t1", 0, 0)), Term(-1, Read(P + "w", 0, h2 - 1))])
        st.family(P + "=t2", [Term(1, Read(P + "t2", 0, 0)), Term(-1, Read(P + "t1", 0, h2 - 1))])
        st.family(P + "=u1", [Term(1, Read(P + "u1", 0, 0)), Term(-1, Read(P + "w", 0, h2 - 13))])
        st.family(P + "=v", [Term(1, Read(P + "v", 0, 0)), Term(-1, Read(P + "u1", 0, h2 - 1))])
        embedded.append(Term(1, Read(P + "t2", 0, 0), Read(P + "ea2", 0, 0), chal=zeta.power(st, 2 * STREAM)))
        embedded.append(Term(1, Read(P + "v", 0, 0), Read(P + "eav", 0, 0), chal=zeta.power(st, 2 * STREAM)))
        embedded.append(Term(1, Read(P + "t1", 0, 0), Read(P + "eb1", 0, 0), chal=zeta.power(st, STREAM)))
        embedded.append(Term(1, Read(P + "u1", 0, 0), Read(P + "ebu", 0, 0), chal=zeta.power(st, STREAM)))
        embedded.append(Term(1, Read(P + "w", 0, 0), Read(P + "ec", 0, 0)))
    return ShaGroup(g, base, chains, embed, digest^, embedded^)


# ---- host ----

def _stream_bit(m: List[UInt32], s: Int) -> UInt8:
    """Bit s of the padded message as a big-endian bit stream."""
    return UInt8((m[s // SLOT] >> UInt32(31 - s % SLOT)) & 1)


def _put(mut trace: List[UInt8], N: Int, col: Int, chain: Int, q: Int, w: UInt32):
    for z in range(SLOT):
        trace[col * N + chain * ROWS + BASE + SLOT * q + z] = UInt8((w >> UInt32(31 - z)) & 1)


def sha256_group_trace[p: Params](layout: Layout, g: ShaGroup, message: List[UInt8], mut trace: List[UInt8]) raises:
    """Fill the group's columns of `trace` (columns_w x N bytes) for `message`."""
    comptime N = p.N()
    if p.h1() != ROWS:
        raise Error("a SHA-256 group needs 144 rows per chain")
    var base = layout.group_base[g.name]
    var R = SLOTS * g.chains
    if ROUNDS // SLOTS * blocks(message) >= g.chains:
        raise Error("message needs more chains than the group has: 16 per 64-byte block plus one")
    var n = live_chains(message, R)
    var words = chain_words(message, R)
    var names = column_names()
    var P = g.name + "."
    var cols = List[Int]()
    for i in range(SHA_COLUMNS):
        cols.append(layout.col(P + names[i]))
    for t in range(R):
        var chain = base + t // SLOTS
        var q = t % SLOTS
        for i in range(SHA_COLUMNS):
            _put(trace, N, cols[i], chain, q, words[t * SHA_COLUMNS + i])
        var w1 = words[((t + 1) % R) * SHA_COLUMNS + 16]
        var w14 = words[((t + 14) % R) * SHA_COLUMNS + 16]
        _put(trace, N, layout.col(P + "w1"), chain, q, w1)
        _put(trace, N, layout.col(P + "w14"), chain, q, w14)
        _put(trace, N, layout.col(P + "w9"), chain, q, words[((t + 9) % R) * SHA_COLUMNS + 16])
        for h in _helpers():
            var src = words[t * SHA_COLUMNS] if h.word == "a" else (words[t * SHA_COLUMNS + 4] if h.word == "e" else (w1 if h.word == "w1" else w14))
            _put(trace, N, layout.col(P + h.name), chain, q, rotr(src, h.r))
        if t < n and t % ROUNDS == ROUNDS - 1:
            for i in range(WORDS):
                _put(trace, N, layout.col(P + "bx" + String(STATE[byte=i])), chain, q, words[(t - 63) * SHA_COLUMNS + i])
    if g.embed >= 0:
        var w = layout.col(P + "w")
        var t1 = layout.col(P + "t1")
        var t2 = layout.col(P + "t2")
        var u1 = layout.col(P + "u1")
        var v = layout.col(P + "v")
        for c in range(1, g.chains):
            for x1 in range(ROWS):
                var row = (base + c) * ROWS + x1
                trace[t1 * N + row] = trace[w * N + row - ROWS]
                if c > 1:
                    trace[t2 * N + row] = trace[t1 * N + row - ROWS]
                if c >= 13:
                    trace[u1 * N + row] = trace[w * N + row - 13 * ROWS]
                if c >= 14:
                    trace[v * N + row] = trace[u1 * N + row - ROWS]


def sha256_group_public[p: Params](layout: Layout, g: ShaGroup, message: List[UInt8]) raises -> List[UInt8]:
    """The public data of the group's public columns in declaration order (N bytes each): the selector, the
    masks, then `_pubs`. The message's length and padding are public; its bytes are not."""
    comptime N = p.N()
    comptime h2 = p.h2()
    var base = layout.group_base[g.name]
    var bk = blocks(message)
    if ROUNDS // SLOTS * bk >= g.chains:
        raise Error("message needs more chains than the group has: 16 per 64-byte block plus one")
    var n = ROUNDS * bk
    var m = padded_words(message)
    var kc = k_const()
    var ivw = iv()
    var out = layout.selector[p](g.name)
    for mk in _masks(h2, g.embed >= 0):
        out.extend(layout.mask[p](g.name, mk.shifts))
    var seg = (0, 0, -1, 0)
    if g.embed >= 0:
        if g.embed + 32 > len(message):
            raise Error("the embedded digest lies outside the message")
        seg = g.segments()
    for u in _pubs(h2, g.embed >= 0):
        var name = u.name
        var r = 0
        for rot in _rots():
            if name == "ge" + String(rot):
                r = rot
        var v = List[UInt8](length=N, fill=0)
        var mk = layout.mask[p](g.name, u.shifts)          # zero off the mask for the column's shifts (the verifier checks)
        for c in range(g.chains):
            for q in range(SLOTS):
                var t = SLOTS * c + q
                var live = t < n
                for z in range(SLOT):
                    var row = (base + c) * ROWS + BASE + SLOT * q + z
                    var s = _stream_index(t, z)                # the padded-message bit, -1 off the message rounds
                    var j = SLOT * q + z                       # the chain's window bit
                    var ecc = c == seg[2]
                    var b: UInt8 = 0
                    if name == "qall":
                        b = 1
                    elif name.startswith("ge"):
                        b = 1 if z >= r else 0
                    elif name == "nb":
                        b = 1 if z < SLOT - 1 else 0
                    elif name == "qlt" or name == "q012":
                        b = 1 if q < 3 else 0
                    elif name == "q3a" or name == "q3b":
                        b = 1 if q == 3 else 0
                    elif name == "q01":
                        b = 1 if q < 2 else 0
                    elif name == "q23":
                        b = 1 if q >= 2 else 0
                    elif name == "lv":
                        b = 1 if live else 0
                    elif name == "end":
                        b = 1 if live and t % ROUNDS == ROUNDS - 1 else 0
                    elif name == "K":
                        b = UInt8((kc[t % ROUNDS] >> UInt32(31 - z)) & 1) if live else 0
                    elif name == "s16":
                        b = 1 if live and t >= 16 and t % ROUNDS < 16 else 0
                    elif name == "pfix":
                        b = 1 if s >= 8 * len(message) and s < 512 * bk else 0
                    elif name == "pv":
                        b = _stream_bit(m, s) if s >= 8 * len(message) and s < 512 * bk else 0
                    elif name == "start":
                        b = 1 if t == 0 else 0
                    elif name == "iv0" or name == "iv1":
                        if t == 0:
                            var k = 4 if name == "iv1" else 0
                            for i in range(4):
                                b += UInt8((ivw[k + i] >> UInt32(31 - z)) & 1) << UInt8(i)
                    elif name == "dsel":
                        b = 1 if t == n - 1 else 0
                    elif name == "ea2":
                        b = 1 if ecc and seg[2] - 2 == seg[0] and j >= seg[3] else 0
                    elif name == "eav":
                        b = 1 if ecc and seg[2] - 14 == seg[0] and j >= seg[3] else 0
                    elif name == "eb1":
                        b = 1 if ecc and seg[2] - 1 == seg[1] else 0
                    elif name == "ebu":
                        b = 1 if ecc and seg[2] - 13 == seg[1] else 0
                    elif name == "ec":
                        b = 1 if ecc and j < seg[3] else 0
                    v[row] = b * mk[row]
        out.extend(v^)
    return out^
