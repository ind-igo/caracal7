"""Keccak-256 as a Workload (statement-layer 8): a row is one bit position z of the whole 25-lane state in one
round, a chain is one round (h1 = 64), chains run permutation-major, round-fastest. Every column is a bit. Rho
is a cyclic read at 64 - r, pi a renaming, the round-to-round state copy a next-chain read, absorb and iota one
public column per absorb lane (the constant of round r - 1 rides into chain r, RC[23] into the next
permutation's absorb or the digest), the digest a restriction of four X columns on the last chain. Message
length is the one knob: b blocks take the last 24 b chains of the grid; the chains before them hold the zero
state, which the round with no constant fixes, so idle padding is zero."""

from caracal7.core.params import Params
from caracal7.relations.ir import FIX_ONE, FIX_E
from caracal7.relations.statement import Statement, Layout, Term, Read, BIT, GATE_2, public_block, restriction_line
from caracal7.workload import Workload

comptime ROUNDS = 24
comptime LANES = 25
comptime ABSORB = 17            # rate 1088 bits: lanes 0 .. 16 in x + 5 y order
comptime BLOCK = 136            # rate bytes
comptime DIGEST = 32
comptime ROWS = 64              # rows per chain: one per bit position
comptime KECCAK_COLUMNS = 142


def rc() -> List[UInt64]:
    return [0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
            0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
            0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
            0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
            0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
            0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008]


def rho() -> List[Int]:
    """Rotation of lane l = x + 5 y."""
    return [0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14]


def rotl(w: UInt64, r: Int) -> UInt64:
    return w if r == 0 else (w << UInt64(r)) | (w >> UInt64(64 - r))


def column_names() -> List[String]:
    """The 142 columns in declaration order: a (state in), e (after absorb, absorb lanes), per x the parity chain
    p1 p2 p3 c, d, t (after theta), n (chi helper), x (chi out, before iota)."""
    var names = List[String]()
    for l in range(LANES):
        names.append("a" + String(l))
    for l in range(ABSORB):
        names.append("e" + String(l))
    for x in range(5):
        for j in range(1, 4):
            names.append("p" + String(x) + String(j))
        names.append("c" + String(x))
    for x in range(5):
        names.append("d" + String(x))
    for tag in ["t", "n", "x"]:
        for l in range(LANES):
            names.append(tag + String(l))
    return names^


def round_words(a: List[UInt64], pub: List[UInt64]) -> List[UInt64]:
    """One chain as 142 words in column order from the state in `a` and the 17 public words; the last 25 are
    the next chain's state."""
    var w = List[UInt64](capacity=KECCAK_COLUMNS)
    var rot = rho()
    var eff = a.copy()
    for l in range(LANES):
        w.append(a[l])
    for l in range(ABSORB):
        eff[l] = a[l] ^ pub[l]
        w.append(eff[l])
    var c = List[UInt64](length=5, fill=0)
    for x in range(5):
        var acc = eff[x] ^ eff[x + 5]
        w.append(acc)
        acc ^= eff[x + 10]
        w.append(acc)
        acc ^= eff[x + 15]
        w.append(acc)
        acc ^= eff[x + 20]
        w.append(acc)
        c[x] = acc
    var d = List[UInt64](length=5, fill=0)
    for x in range(5):
        d[x] = c[(x + 4) % 5] ^ rotl(c[(x + 1) % 5], 1)
        w.append(d[x])
    var t = List[UInt64](length=LANES, fill=0)
    for l in range(LANES):
        t[l] = eff[l] ^ d[l % 5]
        w.append(t[l])
    var b = List[UInt64](length=LANES, fill=0)           # pi of rho: b[X + 5 Y] = rotl(t[src], rho(src))
    for l in range(LANES):
        var s = pi_source(l % 5, l // 5)
        b[l] = rotl(t[s], rot[s])
    var n = List[UInt64](length=LANES, fill=0)
    for l in range(LANES):
        var x = l % 5
        var y = l // 5
        n[l] = ~b[(x + 1) % 5 + 5 * y] & b[(x + 2) % 5 + 5 * y]
        w.append(n[l])
    for l in range(LANES):
        w.append(b[l] ^ n[l])
    return w^


def pi_source(x: Int, y: Int) -> Int:
    """The lane pi moves to (x, y): B[y', 2 x' + 3 y'] = A[x', y'], so x' = x + 3 y, y' = x."""
    return (x % 5 + 3 * y) % 5 + 5 * (x % 5)


def absorb_words(message: List[UInt8]) -> List[UInt64]:
    """The padded message (pad10*1, domain byte 0x01) as b x 17 little-endian words."""
    var b = len(message) // BLOCK + 1
    var bytes = message.copy()
    bytes.resize(b * BLOCK, 0)
    bytes[len(message)] ^= 0x01
    bytes[b * BLOCK - 1] ^= 0x80
    var words = List[UInt64](length=b * ABSORB, fill=0)
    for i in range(b * BLOCK):
        words[i // 8] |= UInt64(bytes[i]) << UInt64(8 * (i % 8))
    return words^


def live_start(blocks: List[UInt64], chains: Int) raises -> Int:
    """The first live chain: the last 24 b chains hold the b blocks. Both sides call it, so a message that
    needs more chains than the grid has is rejected by the verifier's public data, not only by the trace."""
    var first = chains - ROUNDS * (len(blocks) // ABSORB)
    if first < 0:
        raise Error("message needs more chains than the grid has: 24 per 136-byte block")
    return first


def public_words(blocks: List[UInt64], first: Int, k: Int) -> List[UInt64]:
    """The 17 public words of chain k: the message block on an absorb chain (with RC[23] of the previous
    permutation on lane 0), else RC of the previous round on lane 0; zero before the live chains."""
    var pub = List[UInt64](length=ABSORB, fill=0)
    if k < first:
        return pub^
    var r = (k - first) % ROUNDS
    var q = (k - first) // ROUNDS
    if r == 0:
        for l in range(ABSORB):
            pub[l] = blocks[q * ABSORB + l]
        if q > 0:
            pub[0] ^= rc()[ROUNDS - 1]
    else:
        pub[0] = rc()[r - 1]
    return pub^


def chain_words(message: List[UInt8], chains: Int) raises -> List[UInt64]:
    """Every chain's 142 words (chain-major) for a grid of `chains` chains."""
    var blocks = absorb_words(message)
    var first = live_start(blocks, chains)
    var state = List[UInt64](length=LANES, fill=0)
    var out = List[UInt64](capacity=chains * KECCAK_COLUMNS)
    for k in range(chains):
        var w = round_words(state, public_words(blocks, first, k))
        if len(w) != KECCAK_COLUMNS:
            raise Error("round_words and column_names disagree on the column count")
        for l in range(LANES):
            state[l] = w[KECCAK_COLUMNS - LANES + l]
        out.extend(w^)
    return out^


def digest_words(message: List[UInt8]) raises -> List[UInt64]:
    """The final state's four output lanes: the last chain's x lanes, lane 0 with RC[23]."""
    var chains = ROUNDS * (len(message) // BLOCK + 1)
    var w = chain_words(message, chains)
    var off = (chains - 1) * KECCAK_COLUMNS + KECCAK_COLUMNS - LANES
    var d: List[UInt64] = [w[off] ^ rc()[ROUNDS - 1], w[off + 1], w[off + 2], w[off + 3]]
    return d^


def keccak256(message: List[UInt8]) raises -> List[UInt8]:
    var d = digest_words(message)
    var out = List[UInt8](capacity=DIGEST)
    for i in range(DIGEST):
        out.append(UInt8((d[i // 8] >> UInt64(8 * (i % 8))) & 0xFF))
    return out^


def _xor(mut st: Statement, name: String, result: String, u: Read, v: Read) raises:
    """out = u xor v on bits: out - u - v + 2 u v."""
    st.family(name, [Term(1, st.read(result)), Term(-1, u.copy()), Term(-1, v.copy()), Term(2, u.copy(), v.copy())])


def _in(l: Int) -> Read:
    """The state a round works on: after absorb on the absorb lanes."""
    return Read(("e" if l < ABSORB else "a") + String(l), 0, 0)


def _b(x: Int, y: Int) -> Read:
    """B[x, y] after rho and pi: the source lane's t column read at z - r, the cyclic read 64 - r."""
    var s = pi_source(x, y)
    return Read("t" + String(s), (ROWS - rho()[s]) % ROWS, 0)


def keccak_statement() raises -> Statement:
    var st = Statement()
    for n in column_names():
        st.col(n, BIT, group="keccak")
    for l in range(ABSORB):
        st.pub("m" + String(l), 1)                        # dense (h2, h1) block: message bits and round constants
    for l in range(LANES):
        st.family("copy" + String(l), [Term(1, st.read("a" + String(l), k2=1)), Term(-1, st.read("x" + String(l)))], GATE_2)
    for l in range(ABSORB):
        var a = st.read("a" + String(l))
        var m = st.read("m" + String(l))
        st.family("absorb" + String(l), [Term(1, st.read("e" + String(l))), Term(-1, a.copy()), Term(-1, m.copy()), Term(2, m.copy(), a.copy())])
    for x in range(5):
        var sx = String(x)
        _xor(st, "par" + sx + "1", "p" + sx + "1", _in(x), _in(x + 5))
        _xor(st, "par" + sx + "2", "p" + sx + "2", st.read("p" + sx + "1"), _in(x + 10))
        _xor(st, "par" + sx + "3", "p" + sx + "3", st.read("p" + sx + "2"), _in(x + 15))
        _xor(st, "par" + sx + "4", "c" + sx, st.read("p" + sx + "3"), _in(x + 20))
    for x in range(5):
        _xor(st, "theta" + String(x), "d" + String(x), st.read("c" + String((x + 4) % 5)), st.read("c" + String((x + 1) % 5), k1=ROWS - 1))
    for l in range(LANES):
        _xor(st, "apply" + String(l), "t" + String(l), _in(l), st.read("d" + String(l % 5)))
    for l in range(LANES):
        var x = l % 5
        var y = l // 5
        var b1 = _b(x + 1, y)
        var b2 = _b(x + 2, y)
        st.family("chi" + String(l), [Term(1, st.read("n" + String(l))), Term(-1, b2.copy()), Term(1, b1.copy(), b2.copy())])   # n = (1 - b1) b2
        _xor(st, "out" + String(l), "x" + String(l), _b(x, y), st.read("n" + String(l)))
    for l in range(LANES):
        st.restrict("a" + String(l), FIX_ONE, 1)          # the first chain starts from the zero state
    for l in range(4):
        st.restrict("x" + String(l), FIX_E)               # the digest: four lanes of the last chain
    return st^


def keccak_trace[p: Params](layout: Layout, message: List[UInt8]) raises -> List[UInt8]:
    """Columns (column, chain, z) of bits: bit z of each chain word."""
    if p.h1() != ROWS:
        raise Error("Keccak needs 64 rows per chain")
    comptime N = p.N()
    var words = chain_words(message, p.h2())
    var names = column_names()
    var t = List[UInt8](length=layout.columns_w() * N, fill=0)
    for i in range(KECCAK_COLUMNS):
        var col = layout.col(names[i])
        for k in range(p.h2()):
            var w = words[k * KECCAK_COLUMNS + i]
            for z in range(ROWS):
                t[col * N + k * ROWS + z] = UInt8((w >> UInt64(z)) & 1)
    return t^


def _bits(w: UInt64) -> List[UInt8]:
    var v = List[UInt8](capacity=ROWS)
    for z in range(ROWS):
        v.append(UInt8((w >> UInt64(z)) & 1))
    return v^


def keccak_public_values[p: Params](message: List[UInt8], l: Int) raises -> List[UInt8]:
    """Public column l on H: (chain, z) bits of its public words."""
    var blocks = absorb_words(message)
    var first = live_start(blocks, p.h2())
    var vals = List[UInt8](capacity=p.N())
    for k in range(p.h2()):
        vals.extend(_bits(public_words(blocks, first, k)[l]))
    return vals^


@fieldwise_init
struct Keccak(Workload, Copyable, Movable):
    """Keccak-256 of `message` on a grid of 64 x h2 with h2 >= 24 (len // 136 + 1). Public inputs: the
    message, then its 32-byte digest."""
    var message: List[UInt8]

    def statement(self) raises -> Statement:
        return keccak_statement()

    def trace[p: Params](self, layout: Layout) raises -> List[UInt8]:
        return keccak_trace[p](layout, self.message)

    def public_inputs[p: Params](self) raises -> List[UInt8]:
        var v = self.message.copy()
        v.extend(keccak256(self.message))
        return v^

    @staticmethod
    def public_data[p: Params](layout: Layout, public_inputs: List[UInt8]) raises -> List[UInt8]:
        if len(public_inputs) < DIGEST:
            raise Error("public inputs are the message then the digest")
        var message = List[UInt8](capacity=len(public_inputs) - DIGEST)
        for i in range(len(public_inputs) - DIGEST):
            message.append(public_inputs[i])
        var digest = List[UInt64](length=4, fill=0)
        for i in range(DIGEST):
            digest[i // 8] |= UInt64(public_inputs[len(message) + i]) << UInt64(8 * (i % 8))
        digest[0] ^= rc()[ROUNDS - 1]
        var data = List[UInt8]()
        for l in range(ABSORB):
            data.extend(public_block[p](layout, l, keccak_public_values[p](message, l)))
        var zero = List[UInt8](length=ROWS, fill=0)
        for l in range(LANES):
            data.extend(restriction_line[p](layout, l, zero))
        for l in range(4):
            data.extend(restriction_line[p](layout, LANES + l, _bits(digest[l])))
        return data^
