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
`plain_fingerprint` gives a 256-bit value (digest_shift = 0 wires it to RSA's message limb). A 32-byte window
of this group's message at byte offset `embed` spans up to three message chains cA, cB, cC (a block's
message words sit on its first four chains, so the next message chain is one or 13 chains on; s0 = 8 embed
mod 128 bits into cA); transport columns t1 = w one chain up, t2 = t1 one chain up, u1 = w 13 chains up, v =
u1 one chain up bring the first two segments onto chain cC, and wd = w read 128 - s0 rows up (cyclic in the
chain, so w's rows move down) puts cC's own s0 bits on the chain's last rows, where the `embedded` terms ingest them with
coefficients zeta^(128 + s0) (cA's rows >= s0: t2 or v), zeta^s0 (cB: t1 or u1) and 1 (wd on the rows
143 - s0 .. 142), the public masks picking the transport: the plain fingerprint of the window's value, wired
to a digest (digest_shift = 0) or to a limb. A group's windows share s0 (one wd family). One accumulator may
take the terms of several groups (each group's selector is zero off it): the caller makes the accumulators
and wires their slots."""

from core.params import Params
from relations.ir import CHAL_MUL, PubTerm, pack_terms
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
    var embeds: List[Int]       # byte offsets of the embedded 32-byte windows in the message
    var digest: List[Term]      # ingest terms (Horner, scale zeta) of the digest fingerprint times zeta^digest_shift, on the last live chain
    var embedded: List[Term]    # ingest terms of every window's plain fingerprint, each on its `embed_chain`

    def digest_chain(self, message: List[UInt8]) -> Int:
        """The group chain holding the digest (slot 3)."""
        return ROUNDS // SLOTS * blocks(message) - 1

    def segments(self, k: Int = 0) -> Tuple[Int, Int, Int, Int]:
        """(cA, cB, cC, s0) of window k: its first chain, the next two message chains, and its bit offset
        into cA's 128-bit window."""
        var w0 = 8 * self.embeds[k] // SLOT
        var ca = _chain_of_word(w0)
        var cb = _next_message_chain(ca)
        return (ca, cb, _next_message_chain(cb), 8 * self.embeds[k] % STREAM)

    def embed_chain(self, k: Int = 0) -> Int:
        """The chain whose `embedded` terms ingest window k."""
        return self.segments(k)[2]


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
        for n in ["ea2", "eav", "eb1", "ebu", "ed"]:
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


def sha256_group(mut st: Statement, g: String, chains: Int, h2: Int, mut zeta: Zeta, digest_shift: Int, embeds: List[Int] = List[Int]()) raises -> ShaGroup:
    """Declare group `g` of `chains` chains (more than 16 per block of the longest message: the last chain is
    idle) with its columns, public columns and families; the columns are named g.name. Returns the ingest
    terms for the caller's accumulators. `embeds` are the byte offsets of 32-byte windows of the message
    whose plain fingerprints the `embedded` terms ingest, each on its own chain: the windows are congruent
    mod 16 (one bit offset s0 for the group) and ingest on distinct chains (32 bytes apart is enough)."""
    if chains < 2 or chains > h2 or h2 < 16:
        raise Error("a SHA-256 group has 2 to h2 chains: " + g)
    var pre = ShaGroup(g, 0, chains, embeds.copy(), List[Term](), List[Term]())
    var embed = len(embeds) > 0
    var s0 = pre.segments(0)[3] if embed else 0
    for k in range(len(embeds)):
        if embeds[k] < 0 or pre.embed_chain(k) >= chains:
            raise Error("an embedded window's third chain lies outside the group: " + g)
        var seg = pre.segments(k)
        if (seg[2] - seg[0] != 2 and seg[2] - seg[0] != 14) or (seg[2] - seg[1] != 1 and seg[2] - seg[1] != 13):
            raise Error("a window's segments are not on the transports' chains: " + g)      # the public data picks t2 or v, t1 or u1
        if seg[3] != s0:
            raise Error("a group's embedded windows share their bit offset: " + g)
        for l in range(k):
            if pre.embed_chain(l) == pre.embed_chain(k):
                raise Error("two embedded windows ingest on one chain: " + g)
    if st.chains_declared() + chains + 15 > h2:          # masks wrap mod h2: a read 15 chains up from the group's first chains must land off the group
        raise Error("a SHA-256 group ends at least 15 chains before the grid's end: " + g)
    var P = g + "."
    st.pub(P + "sel", 1)
    var base = st.group(g, chains, P + "sel")
    for m in _masks(h2, embed):
        st.mask(g, P + m.name, m.shifts)
    for u in _pubs(h2, embed):
        st.pub(P + u.name, 1, group=g, shifts=u.shifts)
    for n in column_names():
        st.col(P + n, BIT, group=g)
    var helpers: List[String] = ["w1", "w14", "w9"]
    for h in _helpers():
        helpers.append(h.name)
    for i in range(WORDS):
        helpers.append("bx" + String(STATE[byte=i]))
    if embed:
        helpers.extend(["t1", "t2", "u1", "v", "wd"])
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
    if embed:
        st.family(P + "=t1", [Term(1, Read(P + "t1", 0, 0)), Term(-1, Read(P + "w", 0, h2 - 1))])
        st.family(P + "=t2", [Term(1, Read(P + "t2", 0, 0)), Term(-1, Read(P + "t1", 0, h2 - 1))])
        st.family(P + "=u1", [Term(1, Read(P + "u1", 0, 0)), Term(-1, Read(P + "w", 0, h2 - 13))])
        st.family(P + "=v", [Term(1, Read(P + "v", 0, 0)), Term(-1, Read(P + "u1", 0, h2 - 1))])
        st.family(P + "=wd", [Term(1, Read(P + "wd", 0, 0)), Term(-1, Read(P + "w", (ROWS - (STREAM - s0)) % ROWS, 0))])
        embedded.append(Term(1, Read(P + "t2", 0, 0), Read(P + "ea2", 0, 0), chal=zeta.power(st, STREAM + s0)))
        embedded.append(Term(1, Read(P + "v", 0, 0), Read(P + "eav", 0, 0), chal=zeta.power(st, STREAM + s0)))
        embedded.append(Term(1, Read(P + "t1", 0, 0), Read(P + "eb1", 0, 0), chal=zeta.power(st, s0)))
        embedded.append(Term(1, Read(P + "u1", 0, 0), Read(P + "ebu", 0, 0), chal=zeta.power(st, s0)))
        embedded.append(Term(1, Read(P + "wd", 0, 0), Read(P + "ed", 0, 0)))
    return ShaGroup(g, base, chains, embeds.copy(), digest^, embedded^)


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
    if len(g.embeds) > 0:
        var w = layout.col(P + "w")
        var t1 = layout.col(P + "t1")
        var t2 = layout.col(P + "t2")
        var u1 = layout.col(P + "u1")
        var v = layout.col(P + "v")
        var wd = layout.col(P + "wd")
        var d = STREAM - g.segments(0)[3]
        for c in range(g.chains):
            for x1 in range(ROWS):
                var row = (base + c) * ROWS + x1
                trace[wd * N + row] = trace[w * N + (base + c) * ROWS + (x1 + ROWS - d) % ROWS]
                if c == 0:
                    continue
                trace[t1 * N + row] = trace[w * N + row - ROWS]
                if c > 1:
                    trace[t2 * N + row] = trace[t1 * N + row - ROWS]
                if c >= 13:
                    trace[u1 * N + row] = trace[w * N + row - 13 * ROWS]
                if c >= 14:
                    trace[v * N + row] = trace[u1 * N + row - ROWS]


def _slot_rows(qs: Int, z_lo: Int, z_hi: Int) -> List[UInt8]:
    """A row vector: 1 on the rows of slots q with bit q of `qs` set and z in [z_lo, z_hi), 0 elsewhere."""
    var v = List[UInt8](length=ROWS, fill=0)
    for q in range(SLOTS):
        if (qs >> q) & 1 == 1:
            for z in range(z_lo, z_hi):
                v[BASE + SLOT * q + z] = 1
    return v^


def _chains(lo: Int, hi: Int, step: Int = 1) -> List[Int]:
    var v = List[Int]()
    for c in range(lo, hi, step):
        v.append(c)
    return v^


def _nonzero(v: List[UInt8]) -> Bool:
    for x in v:
        if x != 0:
            return True
    return False


def sha256_group_public[p: Params](layout: Layout, g: ShaGroup, message: List[UInt8]) raises -> List[UInt8]:
    """The public data of the group's public columns in declaration order, each in term form (`pack_terms`):
    the selector, the masks, then `_pubs`. Every column is a few (row vector, chain list) terms: a row pattern
    over the four slots on all, the live, or one chain, and the round constants on the sixteen chain classes of
    a block. The message's length and padding are public; its bytes are not."""
    comptime h2 = p.h2()
    comptime PER = ROUNDS // SLOTS                      # chains per block
    var base = layout.group_base[g.name]
    var bk = blocks(message)
    if p.h1() != ROWS:
        raise Error("the group needs " + String(ROWS) + " rows per chain")
    if PER * bk >= g.chains:
        raise Error("message needs more chains than the group has: 16 per 64-byte block plus one")
    var n = ROUNDS * bk
    var live = n // SLOTS                               # live chains: t < n on c < live
    var m = padded_words(message)
    var kc = k_const()
    var ivw = iv()
    var all = _slot_rows(15, 0, SLOT)
    var out = layout.selector[p](g.name)
    for mk in _masks(h2, len(g.embeds) > 0):
        out.extend(layout.mask[p](g.name, mk.shifts))
    for k in range(len(g.embeds)):
        if g.embeds[k] + 32 > len(message):
            raise Error("an embedded window lies outside the message")
    for u in _pubs(h2, len(g.embeds) > 0):
        var name = u.name
        var terms = List[PubTerm]()
        if name == "qall":
            terms.append(PubTerm(all.copy(), _chains(0, g.chains)))
        elif name.startswith("ge"):
            var r = 0
            for rot in _rots():
                if name == "ge" + String(rot):
                    r = rot
            terms.append(PubTerm(_slot_rows(15, r, SLOT), _chains(0, g.chains)))
        elif name == "nb":
            terms.append(PubTerm(_slot_rows(15, 0, SLOT - 1), _chains(0, g.chains)))
        elif name == "qlt" or name == "q012":
            terms.append(PubTerm(_slot_rows(7, 0, SLOT), _chains(0, g.chains)))
        elif name == "q3a" or name == "q3b":
            terms.append(PubTerm(_slot_rows(8, 0, SLOT), _chains(0, g.chains)))
        elif name == "q01":
            terms.append(PubTerm(_slot_rows(3, 0, SLOT), _chains(0, g.chains)))
        elif name == "q23":
            terms.append(PubTerm(_slot_rows(12, 0, SLOT), _chains(0, g.chains)))
        elif name == "lv":
            terms.append(PubTerm(all.copy(), _chains(0, live)))
        elif name == "end":                             # t % 64 == 63: slot 3 of a block's last chain
            terms.append(PubTerm(_slot_rows(8, 0, SLOT), _chains(PER - 1, live, PER)))
        elif name == "K":                               # t % 64 = 4 (c % 16) + q: one row pattern per chain class
            for i in range(PER):
                var row = List[UInt8](length=ROWS, fill=0)
                for q in range(SLOTS):
                    for z in range(SLOT):
                        row[BASE + SLOT * q + z] = UInt8((kc[SLOTS * i + q] >> UInt32(31 - z)) & 1)
                terms.append(PubTerm(row^, _chains(i, live, PER)))
        elif name == "s16":                             # message rounds of every block but the first
            for c in range(PER, live):
                if c % PER < 16 // SLOTS:
                    terms.append(PubTerm(all.copy(), [c]))
        elif name == "pfix" or name == "pv":           # the padding bits: on the message rounds' chains only
            for c in range(live):
                if c % PER >= 16 // SLOTS:
                    continue
                var row = List[UInt8](length=ROWS, fill=0)
                for q in range(SLOTS):
                    for z in range(SLOT):
                        var s = _stream_index(SLOTS * c + q, z)
                        if s >= 8 * len(message) and s < 512 * bk:
                            row[BASE + SLOT * q + z] = 1 if name == "pfix" else _stream_bit(m, s)
                if _nonzero(row):
                    terms.append(PubTerm(row^, [c]))
        elif name == "start":
            terms.append(PubTerm(_slot_rows(1, 0, SLOT), [0]))
        elif name == "iv0" or name == "iv1":
            var k = 4 if name == "iv1" else 0
            var row = List[UInt8](length=ROWS, fill=0)
            for z in range(SLOT):
                for i in range(4):
                    row[BASE + z] += UInt8((ivw[k + i] >> UInt32(31 - z)) & 1) << UInt8(i)
            terms.append(PubTerm(row^, [0]))
        elif name == "dsel":
            terms.append(PubTerm(_slot_rows(8, 0, SLOT), [live - 1]))
        elif name == "ea2" or name == "eav" or name == "ed":     # cA's rows >= s0 through t2 or v; cC's s0 bits on the last rows through wd
            var back = 2 if name == "ea2" else 14
            for k in range(len(g.embeds)):
                var seg = g.segments(k)
                if name == "ed" or seg[2] - back == seg[0]:
                    var row = List[UInt8](length=ROWS, fill=0)
                    for j in range(STREAM):
                        if name == "ed":
                            if j >= STREAM - seg[3]:
                                row[BASE + j] = 1
                        elif j >= seg[3]:
                            row[BASE + j] = 1
                    if _nonzero(row):
                        terms.append(PubTerm(row^, [seg[2]]))
        elif name == "eb1" or name == "ebu":
            for k in range(len(g.embeds)):
                var seg = g.segments(k)
                if seg[2] - (1 if name == "eb1" else 13) == seg[1]:
                    terms.append(PubTerm(all.copy(), [seg[2]]))
        else:
            raise Error("no public data rule for " + name)
        var on = List[Bool](length=h2, fill=False)      # zero off the mask for the column's shifts (the verifier checks)
        for y in layout.mask_chains[p](g.name, u.shifts):
            on[y] = True
        var packed = List[PubTerm]()
        for t in terms:
            var cs = List[Int]()
            for c in t.chains:
                if on[base + c]:
                    cs.append(base + c)
            packed.append(PubTerm(t.row.copy(), cs^))
        out.extend(pack_terms(packed, ROWS))
    return out^


def sha256_digest_factor(entry: Int, entries: Int, digest: List[UInt8]) -> List[UInt8]:
    """The public factor's columns for a digest on an accumulator of `entries` ingest terms whose eight digest
    terms (`ShaGroup.digest`, shift 0) start at `entry`: ROWS bytes per entry, the digest words on slot 3."""
    var v = List[UInt8](length=entries * ROWS, fill=0)
    for i in range(WORDS):
        for z in range(SLOT):
            v[(entry + i) * ROWS + BASE + 3 * SLOT + z] = (digest[4 * i + z // 8] >> UInt8(7 - z % 8)) & 1
    return v^
