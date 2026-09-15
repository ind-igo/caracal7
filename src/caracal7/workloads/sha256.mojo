"""SHA-256 as a Workload: a row is one bit position of the 32-bit words (row 0 the top bit), a chain is one
round (h1 = 32), blocks run 64 chains each from chain 0. Every column is a bit. The rotations of the sigma
functions are cyclic reads; the shifts a cyclic read masked by a per-row selector; the 32-bit additions a
ripple along the rows with binary carry columns, the carry into the low bit cut by a per-row selector. The
working state moves to the next chain through same-chain "next" columns (an addition is quadratic, the
transition linear with the axis-2 gate); at a block end the next columns add the block's input state, read
63 chains back. Chains past the last block hold the final state (a live selector masks the round), so the
digest is the state on the last chain and the IV the state on the first, both public restriction lines.

Hash chain (`Sha256Chain`, the functions named chain_*): h1 = 32 m1 rows, and since gcd(32, m1) = 1 the row
group is Z_32 x Z_m1 (CRT): row rho is bit-row rho mod 32 of lane rho mod m1, so a rotation is the cyclic read
crt(32 - r, 0) and stays in its lane, and the next lane is the cyclic read crt(0, 1). A lane runs
hashes_of(h2) = (h2 - 1) / 64 hashes of 32 bytes down axis 2, each of the previous digest; lane j + 1 starts
from the last digest of lane j; the chain has m1 hashes_of(h2) hashes, lane 0 from the public message. At a
block end with a successor (the public `rstin` inside a lane, the witness `rstx` = endlast * notlast across
lanes) the next-state sums route their result into the witness column `dg` on the successor's first eight
chains (word i on chain + i + 1, or lane + 1 chain i) and the next state resets to the block's start state
read 63 chains back (chain 0 holds the IV), so every block starts from the IV and the end addition needs no
change; lanes before the last settle on the IV, the last on the chain's digest (the FIX_E restriction).
The schedule's head words read `dg` where `s8w` (a digest chain that is not lane 0's block 0) is set, the
message where `l0b0` (lane 0, block 0) is set, and the padding from the public `pad`. Lane-dependent selectors
are witness products of a per-row and a per-chain public column, so every family stays quadratic."""

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
comptime CHAIN_PUBLICS = 15     # msgw pad lv end K s16 rstin s8c b0c endlast (dense), nb ge3 ge10 lane0 notlast (one chain)
comptime CHAIN_DENSE = 10
comptime PAD_AT = 8             # the padding words 8 .. 15 of a 32-byte block: 0x80000000, six zeros, the bit length


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


def compress(state: List[UInt32], block: List[UInt32]) -> List[UInt32]:
    """The compression of one block of 16 words: state + the 64 rounds."""
    var w = block.copy()
    for t in range(16, ROUNDS):
        w.append(sigma1(w[t - 2]) + w[t - 7] + sigma0(w[t - 15]) + w[t - 16])
    var kc = k_const()
    var s = state.copy()
    for t in range(ROUNDS):
        var t1 = s[7] + (rotr(s[4], 6) ^ rotr(s[4], 11) ^ rotr(s[4], 25)) + ((s[4] & s[5]) ^ (~s[4] & s[6])) + kc[t] + w[t]
        var t2 = (rotr(s[0], 2) ^ rotr(s[0], 13) ^ rotr(s[0], 22)) + ((s[0] & s[1]) | (s[2] & (s[0] ^ s[1])))
        s = [t1 + t2, s[0], s[1], s[2], s[3] + t1, s[4], s[5], s[6]]
    for i in range(WORDS):
        s[i] += state[i]
    return s^


def _bytes(words: List[UInt32]) -> List[UInt8]:
    var out = List[UInt8](capacity=4 * len(words))
    for v in words:
        for j in range(4):
            out.append(UInt8((v >> UInt32(8 * (3 - j))) & 0xFF))
    return out^


def sha256(message: List[UInt8]) raises -> List[UInt8]:
    """The digest, big-endian words."""
    var m = padded_words(message)
    var st = iv()
    for b in range(blocks(message)):
        var block = List[UInt32](capacity=16)
        for i in range(16):
            block.append(m[16 * b + i])
        st = compress(st, block)
    return _bytes(st)


def sha256_chain(message: List[UInt8], hashes: Int) raises -> List[UInt8]:
    """The digest of `hashes` hashes: sha256(sha256(... sha256(message)))."""
    var d = sha256(message)
    for _ in range(hashes - 1):
        d = sha256(d)
    return d^


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


# ---- hash chain: lanes on the odd digit of axis 1 (Sha256Chain) ----

def pad32() -> List[UInt32]:
    return [0x80000000, 0, 0, 0, 0, 0, 0, 0x100]


def hashes_of(chains: Int) -> Int:
    """Hashes per lane: 64 chains per hash and at least one chain for the state to settle."""
    return (chains - 1) // ROUNDS


def chain_hashes[p: Params]() -> Int:
    """Hashes in the chain of a grid: m1 lanes of hashes_of(h2)."""
    return p.m1 * hashes_of(p.h2())


def crt(z: Int, j: Int, m: Int) -> Int:
    """The row of bit-row z (mod 32) on lane j (mod m): the CRT index in Z_32 x Z_m = Z_(32 m)."""
    var r = z % ROWS
    while r % m != j % m:
        r += ROWS
    return r


def chain_column_names() -> List[String]:
    """The 43 SHA columns, then dg (the digest word on a successor's first eight chains), l0b0 (lane 0, block 0),
    s8w (digest chain: s8c off lane 0's block 0), rstx (the lane's last block end before another lane), and
    witness copies of the per-row selectors lane0 and notlast (an entry reads at most one public column)."""
    var names = column_names()
    names.extend(["dg", "l0b0", "s8w", "rstx", "lane0w", "notlastw"])
    return names^


def _rotl(col: String, r: Int, m: Int, k2: Int = 0) -> Read:
    """ROTR r inside a lane of m: the cyclic read crt(32 - r, 0)."""
    return Read(col, crt((ROWS - r) % ROWS, 0, m), k2)


def _sum_chain(mut st: Statement, result: String, var terms: List[Term], carries: List[String], m: Int,
               dg: Int, lane1: Int, back_dg: Int) raises:
    """`_sum` for the chain: the carry ripple is the in-lane read crt(1, 0); with `dg` >= 0 (state word i) the
    result goes to dg on chain + i + 1 where rstin is set and to dg on lane + 1 chain i (the read (lane1,
    back_dg)) where rstx is set."""
    for k in range(len(carries)):
        terms.append(Term(1 << k, st.read("nb"), st.read(carries[k], k1=crt(1, 0, m))))
    terms.append(Term(-1, st.read(result)))
    if dg >= 0:
        terms.append(Term(1, st.read("rstin"), st.read(result)))
        terms.append(Term(-1, st.read("rstin"), st.read("dg", k2=dg + 1)))
        terms.append(Term(1, st.read("rstx"), st.read(result)))
        terms.append(Term(-1, st.read("rstx"), Read("dg", lane1, back_dg)))
    for k in range(len(carries)):
        terms.append(Term(-(2 << k), st.read(carries[k])))
    st.family("=" + result, terms^)


def chain_statement[p: Params]() raises -> Statement:
    if p.a1 != 5 or p.h2() < ROUNDS + 1:
        raise Error("the hash chain needs h1 = 32 m1 and more than 64 chains")
    comptime M = p.m1
    comptime B = hashes_of(p.h2())
    comptime BACK = p.h2() - 63
    comptime LANE1 = crt(0, 1, M)
    var st = Statement()
    for n in chain_column_names():
        st.col(n, BIT, group="sha")
    for n in ["msgw", "pad", "lv", "end", "K", "s16", "rstin", "s8c", "b0c", "endlast"]:
        st.pub(n, 1)
    for n in ["nb", "ge3", "ge10", "lane0", "notlast"]:
        st.pub(n, p.h2())
    st.family("=lane0w", [Term(1, st.read("lane0w")), Term(-1, st.read("lane0"))])
    st.family("=notlastw", [Term(1, st.read("notlastw")), Term(-1, st.read("notlast"))])
    st.family("=l0b0", [Term(1, st.read("l0b0")), Term(-1, st.read("b0c"), st.read("lane0w"))])
    st.family("=s8w", [Term(1, st.read("s8w")), Term(-1, st.read("s8c")), Term(1, st.read("s8c"), st.read("l0b0"))])
    st.family("=rstx", [Term(1, st.read("rstx")), Term(-1, st.read("endlast"), st.read("notlastw"))])
    for i in range(WORDS):
        var x = String(STATE[byte=i])
        st.family("copy" + x, [Term(1, st.read(x, k2=1)), Term(-1, st.read(x + "n"))], GATE_2)
        # a reset: the next state is the block's start state (chain 0 holds the IV); the sum went to dg
        st.family("rs" + x, [Term(1, st.read("rstin"), st.read(x + "n")), Term(-1, st.read("rstin"), st.read(x, k2=BACK)),
                             Term(1, st.read("rstx"), st.read(x + "n")), Term(-1, st.read("rstx"), st.read(x, k2=BACK))])
    _xor(st, "s0x", _rotl("a", 2, M), _rotl("a", 13, M))
    _xor(st, "s0", st.read("s0x"), _rotl("a", 22, M))
    _xor(st, "s1x", _rotl("e", 6, M), _rotl("e", 11, M))
    _xor(st, "s1", st.read("s1x"), _rotl("e", 25, M))
    _xor(st, "ab", st.read("a"), st.read("b"))
    st.family("=mj", [Term(1, st.read("mj")), Term(-1, st.read("a"), st.read("b")), Term(-1, st.read("c"), st.read("ab"))])
    st.family("=ch", [Term(1, st.read("ch")), Term(-1, st.read("e"), st.read("f")), Term(-1, st.read("g")), Term(1, st.read("e"), st.read("g"))])
    _xor(st, "g0x", _rotl("w", 7, M, 1), _rotl("w", 18, M, 1))
    st.family("=g0m", [Term(1, st.read("g0m")), Term(-1, st.read("ge3"), _rotl("w", 3, M, 1))])
    _xor(st, "g1x", _rotl("w", 17, M, 14), _rotl("w", 19, M, 14))
    st.family("=g1m", [Term(1, st.read("g1m")), Term(-1, st.read("ge10"), _rotl("w", 10, M, 14))])
    _sum_chain(st, "ws", [Term(1, st.read("g1x")), Term(1, st.read("g1m")), Term(-2, st.read("g1x"), st.read("g1m")),
                          Term(1, st.read("w", k2=9)),
                          Term(1, st.read("g0x")), Term(1, st.read("g0m")), Term(-2, st.read("g0x"), st.read("g0m")),
                          Term(1, st.read("w"))], ["wc0", "wc1"], M, -1, 0, 0)
    # w[t + 16]: ws off the head chains; on them the padding, the message on lane 0's block 0, else the digest word
    st.family("=w", [Term(1, st.read("w", k2=16)), Term(-1, st.read("ws")), Term(1, st.read("s16", k2=16), st.read("ws")),
                     Term(-1, st.read("pad", k2=16)), Term(-1, st.read("msgw", k2=16), st.read("l0b0", k2=16)),
                     Term(-1, st.read("s8w", k2=16), st.read("dg", k2=16))])
    var an: List[Term] = [Term(1, st.read("K")), Term(1, st.read("a")), Term(-1, st.read("lv"), st.read("a")), Term(1, st.read("end"), st.read("a", k2=BACK))]
    for n in ["h", "s1", "ch", "w", "s0", "mj"]:
        an.append(Term(1, st.read("lv"), st.read(n)))
    _sum_chain(st, "an", an^, ["ac0", "ac1", "ac2"], M, 0, LANE1, (1 - 64 * B) % p.h2())
    var en: List[Term] = [Term(1, st.read("K")), Term(1, st.read("e")), Term(-1, st.read("lv"), st.read("e")), Term(1, st.read("end"), st.read("e", k2=BACK))]
    for n in ["d", "h", "s1", "ch", "w"]:
        en.append(Term(1, st.read("lv"), st.read(n)))
    _sum_chain(st, "en", en^, ["ec0", "ec1", "ec2"], M, 4, LANE1, (5 - 64 * B) % p.h2())
    for i in range(1, WORDS):
        if i == 4:
            continue
        var x = String(STATE[byte=i])
        var src = String(STATE[byte=i - 1])
        _sum_chain(st, x + "n", [Term(1, st.read("lv"), st.read(src)), Term(1, st.read(x)), Term(-1, st.read("lv"), st.read(x)),
                                 Term(1, st.read("end"), st.read(x, k2=BACK))], [x + "c"], M, i, LANE1, (i + 1 - 64 * B) % p.h2())
    for i in range(WORDS):
        st.restrict(String(STATE[byte=i]), FIX_ONE)      # the IV on every lane
    for i in range(WORDS):
        st.restrict(String(STATE[byte=i]), FIX_E)        # the IV on the lanes before the last, the digest on it
    return st^


def chain_word(m: List[UInt32], kc: List[UInt32], n: Int, B: Int, l: Int, t: Int) -> UInt32:
    """Dense public column l on chain t as a word (the same on every lane): msgw (the message words on the first
    eight chains), pad (the padding on chains 8 .. 15 of every block), lv, end, K, s16 as `public_word`,
    rstin (a block end with a block after it in the lane), s8c (the first eight chains of a block), b0c (block
    0), endlast (the lane's last block end)."""
    var live = t < n
    var b = t // ROUNDS
    var r = t % ROUNDS
    if not live:
        return 0
    if l == 0:
        return m[r] if b == 0 and r < PAD_AT else 0
    if l == 1:
        return pad32()[r - PAD_AT] if r >= PAD_AT and r < 16 else 0
    if l == 2:
        return 0xFFFFFFFF
    if l == 3:
        return 0xFFFFFFFF if r == ROUNDS - 1 else 0
    if l == 4:
        return kc[r]
    if l == 5:
        return 0xFFFFFFFF if r < 16 else 0
    if l == 6:
        return 0xFFFFFFFF if r == ROUNDS - 1 and b < B - 1 else 0
    if l == 7:
        return 0xFFFFFFFF if r < PAD_AT else 0
    if l == 8:
        return 0xFFFFFFFF if b == 0 else 0
    return 0xFFFFFFFF if r == ROUNDS - 1 and b == B - 1 else 0


def chain_row_selector(l: Int, rho: Int, m: Int) -> UInt8:
    """Per-row public column l (10 .. 14) at row rho: nb, ge3, ge10 on the bit-row, lane0 and notlast on the lane."""
    var z = rho % ROWS
    var j = rho % m
    if l == 10:
        return 1 if z != ROWS - 1 else 0
    if l == 11:
        return 1 if z >= 3 else 0
    if l == 12:
        return 1 if z >= 10 else 0
    if l == 13:
        return 1 if j == 0 else 0
    return 1 if j != m - 1 else 0


def lane_words(head: List[UInt32], digest_in: List[UInt32], chains: Int, B: Int, reset_last: Bool) raises -> List[UInt32]:
    """One lane's 44 words per chain (the SHA columns then dg): B blocks, block 0 from the 16 `head` words, block
    b >= 1 from the previous digest and the padding; dg on the first eight chains of block 0 from `digest_in`
    (empty on lane 0). Every block starts from the IV; a block end resets the next state when a block follows
    (`reset_last` for the lane's last block: another lane follows)."""
    var n = ROUNDS * B
    var kc = k_const()
    var w = List[UInt32](length=chains, fill=0)
    var st = iv()
    for t in range(chains):
        var r = t % ROUNDS
        if t < n and r < 16:
            if t >= ROUNDS:
                w[t] = st[r] if r < PAD_AT else pad32()[r - PAD_AT]
            else:
                w[t] = head[r]
        else:
            w[t] = sigma1(w[t - 2]) + w[t - 7] + sigma0(w[t - 15]) + w[t - 16]
        if t < n and r == ROUNDS - 1:
            var block = List[UInt32](capacity=16)
            for i in range(16):
                block.append(w[t - 63 + i])
            st = compress(iv(), block)
    st = iv()
    var hist = List[UInt32](capacity=chains * WORDS)      # the state entering every chain
    var out = List[UInt32](capacity=chains * (SHA_COLUMNS + 1))
    var digest = digest_in.copy()
    for t in range(chains):
        var live = t < n
        var b = t // ROUNDS
        var end = live and t % ROUNDS == ROUNDS - 1
        var k = kc[t % ROUNDS] if live else 0
        hist.extend(st.copy())
        var a = st[0]
        var bb = st[1]
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
        var ab = a ^ bb
        var mj = (a & bb) | (c & ab)
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
        var dg: UInt32 = 0
        if live and t % ROUNDS < PAD_AT and len(digest) == WORDS:
            dg = digest[t % ROUNDS]
        if end and (b < B - 1 or reset_last):              # the sum goes to dg, the next state is the IV
            digest = nxt.copy()
            nxt = iv()
        out.extend(nxt.copy())
        out.extend([w[t], ws, wc[0], wc[1], s0x, s0, s1x, s1, ab, mj, ch, g0x, g0m, g1x, g1m,
                    ac[0], ac[1], ac[2], ec[0], ec[1], ec[2]])
        out.extend(cc^)
        out.append(dg)
        st = nxt^
    return out^


def _check_chain(message: List[UInt8], h2: Int) raises:
    if len(message) != DIGEST or hashes_of(h2) < 1:
        raise Error("the hash chain takes a 32-byte first message and at least 65 chains")


def _digest_words(d: List[UInt8]) -> List[UInt32]:
    var out = List[UInt32](capacity=WORDS)
    for i in range(WORDS):
        var v: UInt32 = 0
        for j in range(4):
            v |= UInt32(d[4 * i + j]) << UInt32(8 * (3 - j))
        out.append(v)
    return out^


def chain_trace[p: Params](layout: Layout, message: List[UInt8]) raises -> List[UInt8]:
    """Columns (column, chain, row) of bits: lane j's words on the rows crt(z, j)."""
    comptime M = p.m1
    comptime B = hashes_of(p.h2())
    comptime N = p.N()
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    _check_chain(message, h2)
    var names = chain_column_names()
    var t = List[UInt8](length=layout.columns_w() * N, fill=0)
    var kc = k_const()
    var m = padded_words(message)
    var prev = List[UInt32]()                                 # the digest entering the lane
    var col_l0b0 = layout.col("l0b0")
    var col_s8w = layout.col("s8w")
    var col_rstx = layout.col("rstx")
    var col_lane0w = layout.col("lane0w")
    var col_notlastw = layout.col("notlastw")
    for j in range(M):
        var head = List[UInt32](capacity=16)
        if j == 0:
            head = m.copy()
        else:
            head.extend(prev.copy())
            head.extend(pad32())
        var words = lane_words(head, prev, h2, B, j < M - 1)
        for i in range(SHA_COLUMNS + 1):
            var col = layout.col(names[i])
            for k in range(h2):
                var w = words[k * (SHA_COLUMNS + 1) + i]
                for z in range(ROWS):
                    t[col * N + k * h1 + crt(z, j, M)] = UInt8((w >> UInt32(31 - z)) & 1)
        prev = _digest_words(sha256_chain(message, (j + 1) * B))   # the lane's last digest: dg of the next lane
    var n = ROUNDS * B
    for k in range(h2):
        var b0c = chain_word(m, kc, n, B, 8, k) != 0
        var s8c = chain_word(m, kc, n, B, 7, k) != 0
        var endlast = chain_word(m, kc, n, B, 9, k) != 0
        for rho in range(h1):
            var lane0 = rho % M == 0
            var notlast = rho % M != M - 1
            var l0b0 = lane0 and b0c
            t[col_l0b0 * N + k * h1 + rho] = 1 if l0b0 else 0
            t[col_s8w * N + k * h1 + rho] = 1 if s8c and not l0b0 else 0
            t[col_rstx * N + k * h1 + rho] = 1 if endlast and notlast else 0
            t[col_lane0w * N + k * h1 + rho] = 1 if lane0 else 0
            t[col_notlastw * N + k * h1 + rho] = 1 if notlast else 0
    return t^


def chain_public_values[p: Params](message: List[UInt8], l: Int) raises -> List[UInt8]:
    """One period of public column l: dense columns (chain, row) with the chain's word on every lane; the
    per-row selectors one chain of h1 values."""
    comptime h1 = p.h1()
    _check_chain(message, p.h2())
    var B = hashes_of(p.h2())
    var n = ROUNDS * B
    var m = padded_words(message)
    var kc = k_const()
    if l >= CHAIN_DENSE:
        var vals = List[UInt8](capacity=h1)
        for rho in range(h1):
            vals.append(chain_row_selector(l, rho, p.m1))
        return vals^
    var vals = List[UInt8](capacity=p.h2() * h1)
    for t in range(p.h2()):
        var w = chain_word(m, kc, n, B, l, t)
        for rho in range(h1):
            vals.append(UInt8((w >> UInt32(31 - rho % ROWS)) & 1))
    return vals^


@fieldwise_init
struct Sha256Chain(Workload, Copyable, Movable):
    """A hash chain of chain_hashes[p]() hashes on a grid of 32 m1 x h2: lane j runs hashes_of(h2) hashes down
    axis 2, the first of the previous lane's last digest (lane 0 of `message`, 32 bytes). Public inputs: the
    message, then the last digest."""
    var message: List[UInt8]

    def statement[p: Params](self) raises -> Statement:
        return chain_statement[p]()

    def trace[p: Params](self, layout: Layout) raises -> List[UInt8]:
        return chain_trace[p](layout, self.message)

    def public_inputs[p: Params](self) raises -> List[UInt8]:
        var v = self.message.copy()
        v.extend(sha256_chain(self.message, chain_hashes[p]()))
        return v^

    @staticmethod
    def public_data[p: Params](layout: Layout, public_inputs: List[UInt8]) raises -> List[UInt8]:
        comptime h1 = p.h1()
        if len(public_inputs) != 2 * DIGEST:
            raise Error("public inputs are the 32-byte message then the digest")
        var message = List[UInt8](capacity=DIGEST)
        for i in range(DIGEST):
            message.append(public_inputs[i])
        var data = List[UInt8]()
        for l in range(CHAIN_PUBLICS):
            data.extend(chain_public_values[p](message, l))
        var init = iv()
        for i in range(WORDS):
            var line = List[UInt8](capacity=h1)
            for rho in range(h1):
                line.append(UInt8((init[i] >> UInt32(31 - rho % ROWS)) & 1))
            data.extend(restriction_line[p](layout, i, line))
        for i in range(WORDS):
            var d: UInt32 = 0
            for j in range(4):
                d |= UInt32(public_inputs[DIGEST + 4 * i + j]) << UInt32(8 * (3 - j))
            var line = List[UInt8](capacity=h1)
            for rho in range(h1):
                var w = d if rho % p.m1 == p.m1 - 1 else init[i]
                line.append(UInt8((w >> UInt32(31 - rho % ROWS)) & 1))
            data.extend(restriction_line[p](layout, WORDS + i, line))
        return data^
