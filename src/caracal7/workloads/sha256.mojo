"""SHA-256 as a Workload: a row is one bit position of the 32-bit words (row 0 the top bit), a chain is one
round (h1 = 32), blocks run 64 chains each from chain 0. Every column is a bit. The rotations of the sigma
functions are cyclic reads; the shifts a cyclic read masked by a per-row selector; the 32-bit additions a
ripple along the rows with binary carry columns, the carry into the low bit cut by a per-row selector. The
working state moves to the next chain through same-chain "next" columns (an addition is quadratic, the
transition linear with the axis-2 gate); at a block end the next columns add the block's input state, read
63 chains back. Chains past the last block hold the final state (a live selector masks the round), so the
digest is the state on the last chain and the IV the state on the first, both public restriction lines."""

from caracal7.core.params import Params
from caracal7.relations.ir import FIX_ONE, FIX_E
from caracal7.relations.statement import Statement, Layout, Term, Read, BIT, GATE_2, restriction_line
from caracal7.workload import Workload

comptime ROUNDS = 64
comptime WORDS = 8              # state words a .. h
comptime BLOCK = 64             # bytes per block
comptime DIGEST = 32
comptime ROWS = 32              # rows per chain: one per bit, the top bit first
comptime SHA_COLUMNS = 43
comptime PUBLICS = 8            # msg lv end K s16 (dense), nb ge3 ge10 (one chain)
comptime STATE = "abcdefgh"


def k_const() -> List[UInt32]:
    return [0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]


def iv() -> List[UInt32]:
    return [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]


def rotr(w: UInt32, r: Int) -> UInt32:
    return w if r == 0 else (w >> UInt32(r)) | (w << UInt32(32 - r))


def column_names() -> List[String]:
    """The 43 columns in declaration order: the state a..h entering the round, the next state an..hn, the
    schedule word w, its successor ws with carries wc, the sigma helpers, Maj and Ch, the schedule sigma
    helpers (x: the first two rotations, m: the masked shift), the carries of an, en, and the six copies."""
    var names: List[String] = ["a", "b", "c", "d", "e", "f", "g", "h", "an", "bn", "cn", "dn", "en", "fn", "gn", "hn",
                               "w", "ws", "wc0", "wc1", "s0x", "s0", "s1x", "s1", "ab", "mj", "ch",
                               "g0x", "g0m", "g1x", "g1m", "ac0", "ac1", "ac2", "ec0", "ec1", "ec2",
                               "bc", "cc", "dc", "fc", "gc", "hc"]
    return names^


def blocks(message: List[UInt8]) -> Int:
    return (len(message) + 9 + BLOCK - 1) // BLOCK


def padded_words(message: List[UInt8]) -> List[UInt32]:
    """The padded message (0x80, zeros, the 64-bit big-endian bit length) as 16 b big-endian words."""
    var b = blocks(message)
    var bytes = message.copy()
    bytes.append(0x80)
    bytes.resize(b * BLOCK, 0)
    var bits = UInt64(len(message)) * 8
    for i in range(8):
        bytes[b * BLOCK - 1 - i] = UInt8((bits >> UInt64(8 * i)) & 0xFF)
    var words = List[UInt32](length=b * 16, fill=0)
    for i in range(b * BLOCK):
        words[i // 4] |= UInt32(bytes[i]) << UInt32(8 * (3 - i % 4))
    return words^


def _add(addends: List[UInt32], mut carry: List[UInt32]) raises -> UInt32:
    """The sum mod 2^32 with the carry-out bit k of every bit position in word k of `carry` (as many words as
    the caller provides; the carry out of the top bit lands in bit 31 and is dropped)."""
    var s: UInt32 = 0
    var cin = 0
    for z in range(ROWS):
        var tot = cin
        for a in addends:
            tot += Int((a >> UInt32(z)) & 1)
        s |= UInt32(tot & 1) << UInt32(z)
        var cout = tot >> 1
        if cout >> len(carry) != 0:
            raise Error("carry does not fit its bit columns")
        for k in range(len(carry)):
            carry[k] |= UInt32((cout >> k) & 1) << UInt32(z)
        cin = cout
    return s


def sigma0(x: UInt32) -> UInt32:
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)


def sigma1(x: UInt32) -> UInt32:
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)


def live_chains(message: List[UInt8], chains: Int) raises -> Int:
    """64 b: the blocks take the first 64 b chains; the state settles on at least one chain after them. Both
    sides call it, so a message too long for the grid is rejected by the verifier's public data too."""
    var n = ROUNDS * blocks(message)
    if n >= chains:
        raise Error("message needs more chains than the grid has: 64 per 64-byte block plus one")
    return n


def public_word(m: List[UInt32], kc: List[UInt32], n: Int, l: Int, t: Int) -> UInt32:
    """Public column l on chain t as a word, from the padded words `m`, the round constants `kc` and the live
    chain count `n`: msg (the
    block word on the first 16 chains of a block), lv (live chain), end (block end), K (the round constant),
    s16 (message chain); nb, ge3, ge10 are per-row selectors (the same word on every chain): every row but the
    last, rows 3 and up, rows 10 and up."""
    var live = t < n
    var r = t % ROUNDS
    if l == 0:
        return m[t // ROUNDS * 16 + r] if live and r < 16 else 0
    if l == 1:
        return 0xFFFFFFFF if live else 0
    if l == 2:
        return 0xFFFFFFFF if live and r == ROUNDS - 1 else 0
    if l == 3:
        return kc[r] if live else 0
    if l == 4:
        return 0xFFFFFFFF if live and r < 16 else 0
    if l == 5:
        return 0xFFFFFFFE
    if l == 6:
        return 0x1FFFFFFF
    return 0x003FFFFF


def chain_words(message: List[UInt8], chains: Int) raises -> List[UInt32]:
    """Every chain's 43 words (chain-major) for a grid of `chains` chains: the families evaluated forward."""
    var n = live_chains(message, chains)
    var m = padded_words(message)
    var kc = k_const()
    var w = List[UInt32](length=chains, fill=0)
    for t in range(chains):
        if t < n and t % ROUNDS < 16:
            w[t] = m[t // ROUNDS * 16 + t % ROUNDS]
        else:
            w[t] = sigma1(w[t - 2]) + w[t - 7] + sigma0(w[t - 15]) + w[t - 16]
    var st = iv()
    var hist = List[UInt32](capacity=chains * WORDS)      # the state entering every chain
    var out = List[UInt32](capacity=chains * SHA_COLUMNS)
    for t in range(chains):
        var live = t < n
        var end = live and t % ROUNDS == ROUNDS - 1
        var k = kc[t % ROUNDS] if live else 0
        hist.extend(st.copy())
        var a = st[0]
        var b = st[1]
        var c = st[2]
        var d = st[3]
        var e = st[4]
        var f = st[5]
        var g = st[6]
        var h = st[7]
        var w1 = w[(t + 1) % chains]
        var w14 = w[(t + 14) % chains]
        var s0x = rotr(a, 2) ^ rotr(a, 13)
        var s0 = s0x ^ rotr(a, 22)
        var s1x = rotr(e, 6) ^ rotr(e, 11)
        var s1 = s1x ^ rotr(e, 25)
        var ab = a ^ b
        var mj = (a & b) | (c & ab)
        var ch = (e & f) ^ (~e & g)
        var g0x = rotr(w1, 7) ^ rotr(w1, 18)
        var g0m = w1 >> 3
        var g1x = rotr(w14, 17) ^ rotr(w14, 19)
        var g1m = w14 >> 10
        var wc = List[UInt32](length=2, fill=0)
        var ws = _add([g1x ^ g1m, w[(t + 9) % chains], g0x ^ g0m, w[t]], wc)
        var back = (t - 63) * WORDS
        var ac = List[UInt32](length=3, fill=0)
        var ec = List[UInt32](length=3, fill=0)
        var nxt = List[UInt32](capacity=WORDS)
        var cc = List[UInt32](capacity=6)
        if live:
            nxt.append(_add([h, s1, ch, w[t], s0, mj, k, hist[back] if end else 0], ac))
        else:
            nxt.append(_add([a], ac))
        for i in range(1, WORDS):
            if i == 4:
                if live:
                    nxt.append(_add([d, h, s1, ch, w[t], k, hist[back + 4] if end else 0], ec))
                else:
                    nxt.append(_add([e], ec))
                continue
            var one = List[UInt32](length=1, fill=0)
            nxt.append(_add([st[i - 1] if live else st[i], hist[back + i] if end else 0], one))
            cc.append(one[0])
        out.extend(st.copy())
        out.extend(nxt.copy())
        out.extend([w[t], ws, wc[0], wc[1], s0x, s0, s1x, s1, ab, mj, ch, g0x, g0m, g1x, g1m,
                    ac[0], ac[1], ac[2], ec[0], ec[1], ec[2]])
        out.extend(cc^)
        st = nxt^
    return out^


def sha256(message: List[UInt8]) raises -> List[UInt8]:
    """The digest: the state on the chain after the last block, big-endian words."""
    var chains = ROUNDS * blocks(message) + 1
    var words = chain_words(message, chains)
    var out = List[UInt8](capacity=DIGEST)
    for i in range(WORDS):
        var v = words[(chains - 1) * SHA_COLUMNS + i]
        for j in range(4):
            out.append(UInt8((v >> UInt32(8 * (3 - j))) & 0xFF))
    return out^


def _rot(col: String, r: Int, k2: Int = 0) -> Read:
    """ROTR r of a word: bit z reads bit z + r, row rho reads row rho - r, the cyclic read 32 - r."""
    return Read(col, (ROWS - r) % ROWS, k2)


def _xor(mut st: Statement, result: String, u: Read, v: Read) raises:
    st.family("=" + result, [Term(1, st.read(result)), Term(-1, u.copy()), Term(-1, v.copy()), Term(2, u.copy(), v.copy())])


def _sum(mut st: Statement, result: String, var terms: List[Term], carries: List[String]) raises:
    """addends + carry in (from the row below, cut on the last row by nb) = result + 2 carry out, per row."""
    for k in range(len(carries)):
        terms.append(Term(1 << k, st.read("nb"), st.read(carries[k], k1=1)))
    terms.append(Term(-1, st.read(result)))
    for k in range(len(carries)):
        terms.append(Term(-(2 << k), st.read(carries[k])))
    st.family("=" + result, terms^)


def sha256_statement[p: Params]() raises -> Statement:
    if p.h1() != ROWS or p.h2() < ROUNDS + 1:
        raise Error("SHA-256 needs 32 rows per chain and more than 64 chains")
    var st = Statement()
    for n in column_names():
        st.col(n, BIT, group="sha")
    for n in ["msg", "lv", "end", "K", "s16"]:
        st.pub(n, 1)
    for n in ["nb", "ge3", "ge10"]:
        st.pub(n, p.h2())
    comptime BACK = p.h2() - 63
    for i in range(WORDS):
        var x = String(STATE[byte=i])
        st.family("copy" + x, [Term(1, st.read(x, k2=1)), Term(-1, st.read(x + "n"))], GATE_2)
    _xor(st, "s0x", _rot("a", 2), _rot("a", 13))
    _xor(st, "s0", st.read("s0x"), _rot("a", 22))
    _xor(st, "s1x", _rot("e", 6), _rot("e", 11))
    _xor(st, "s1", st.read("s1x"), _rot("e", 25))
    _xor(st, "ab", st.read("a"), st.read("b"))
    st.family("=mj", [Term(1, st.read("mj")), Term(-1, st.read("a"), st.read("b")), Term(-1, st.read("c"), st.read("ab"))])
    st.family("=ch", [Term(1, st.read("ch")), Term(-1, st.read("e"), st.read("f")), Term(-1, st.read("g")), Term(1, st.read("e"), st.read("g"))])
    _xor(st, "g0x", _rot("w", 7, 1), _rot("w", 18, 1))
    st.family("=g0m", [Term(1, st.read("g0m")), Term(-1, st.read("ge3"), _rot("w", 3, 1))])
    _xor(st, "g1x", _rot("w", 17, 14), _rot("w", 19, 14))
    st.family("=g1m", [Term(1, st.read("g1m")), Term(-1, st.read("ge10"), _rot("w", 10, 14))])
    # ws = sigma1(w[t + 14]) + w[t + 9] + sigma0(w[t + 1]) + w[t]: each sigma the xor of its x and m columns
    _sum(st, "ws", [Term(1, st.read("g1x")), Term(1, st.read("g1m")), Term(-2, st.read("g1x"), st.read("g1m")),
                    Term(1, st.read("w", k2=9)),
                    Term(1, st.read("g0x")), Term(1, st.read("g0m")), Term(-2, st.read("g0x"), st.read("g0m")),
                    Term(1, st.read("w"))], ["wc0", "wc1"])
    # w[t + 16] = msg on a message chain, else ws
    st.family("=w", [Term(1, st.read("w", k2=16)), Term(-1, st.read("ws")), Term(1, st.read("s16", k2=16), st.read("ws")),
                     Term(-1, st.read("msg", k2=16))])
    # next state: lv masks the round (idle chains hold the state), end adds the block's input state
    var an: List[Term] = [Term(1, st.read("K")), Term(1, st.read("a")), Term(-1, st.read("lv"), st.read("a")), Term(1, st.read("end"), st.read("a", k2=BACK))]
    for n in ["h", "s1", "ch", "w", "s0", "mj"]:
        an.append(Term(1, st.read("lv"), st.read(n)))
    _sum(st, "an", an^, ["ac0", "ac1", "ac2"])
    var en: List[Term] = [Term(1, st.read("K")), Term(1, st.read("e")), Term(-1, st.read("lv"), st.read("e")), Term(1, st.read("end"), st.read("e", k2=BACK))]
    for n in ["d", "h", "s1", "ch", "w"]:
        en.append(Term(1, st.read("lv"), st.read(n)))
    _sum(st, "en", en^, ["ec0", "ec1", "ec2"])
    for i in range(1, WORDS):
        if i == 4:
            continue
        var x = String(STATE[byte=i])
        var src = String(STATE[byte=i - 1])
        _sum(st, x + "n", [Term(1, st.read("lv"), st.read(src)), Term(1, st.read(x)), Term(-1, st.read("lv"), st.read(x)),
                           Term(1, st.read("end"), st.read(x, k2=BACK))], [x + "c"])
    for i in range(WORDS):
        st.restrict(String(STATE[byte=i]), FIX_ONE)      # the IV
    for i in range(WORDS):
        st.restrict(String(STATE[byte=i]), FIX_E)        # the digest
    return st^


def _bits(w: UInt32) -> List[UInt8]:
    """Row rho holds bit 31 - rho."""
    var v = List[UInt8](capacity=ROWS)
    for r in range(ROWS):
        v.append(UInt8((w >> UInt32(31 - r)) & 1))
    return v^


def sha256_trace[p: Params](layout: Layout, message: List[UInt8]) raises -> List[UInt8]:
    """Columns (column, chain, row) of bits."""
    if p.h1() != ROWS:
        raise Error("SHA-256 needs 32 rows per chain")
    comptime N = p.N()
    var words = chain_words(message, p.h2())
    var names = column_names()
    var t = List[UInt8](length=layout.columns_w() * N, fill=0)
    for i in range(SHA_COLUMNS):
        var col = layout.col(names[i])
        for k in range(p.h2()):
            var w = words[k * SHA_COLUMNS + i]
            for r in range(ROWS):
                t[col * N + k * ROWS + r] = UInt8((w >> UInt32(31 - r)) & 1)
    return t^


def sha256_public_values[p: Params](message: List[UInt8], l: Int) raises -> List[UInt8]:
    """One period of public column l: (chain, row) bits of its words; the per-row selectors are one chain."""
    var chains = p.h2() if l < 5 else 1
    var n = live_chains(message, p.h2())
    var m = padded_words(message)
    var kc = k_const()
    var vals = List[UInt8](capacity=chains * ROWS)
    for t in range(chains):
        vals.extend(_bits(public_word(m, kc, n, l, t)))
    return vals^


@fieldwise_init
struct Sha256(Workload, Copyable, Movable):
    """SHA-256 of `message` on a grid of 32 x h2 with h2 > 64 blocks. Public inputs: the message, then its
    32-byte digest."""
    var message: List[UInt8]

    def statement[p: Params](self) raises -> Statement:
        return sha256_statement[p]()

    def trace[p: Params](self, layout: Layout) raises -> List[UInt8]:
        return sha256_trace[p](layout, self.message)

    def public_inputs[p: Params](self) raises -> List[UInt8]:
        var v = self.message.copy()
        v.extend(sha256(self.message))
        return v^

    @staticmethod
    def public_data[p: Params](layout: Layout, public_inputs: List[UInt8]) raises -> List[UInt8]:
        if len(public_inputs) < DIGEST:
            raise Error("public inputs are the message then the digest")
        var message = List[UInt8](capacity=len(public_inputs) - DIGEST)
        for i in range(len(public_inputs) - DIGEST):
            message.append(public_inputs[i])
        var data = List[UInt8]()
        for l in range(PUBLICS):
            data.extend(sha256_public_values[p](message, l))
        var init = iv()
        for i in range(WORDS):
            data.extend(restriction_line[p](layout, i, _bits(init[i])))
        for i in range(WORDS):
            var d: UInt32 = 0
            for j in range(4):
                d |= UInt32(public_inputs[len(message) + 4 * i + j]) << UInt32(8 * (3 - j))
            data.extend(restriction_line[p](layout, WORDS + i, _bits(d)))
        return data^
