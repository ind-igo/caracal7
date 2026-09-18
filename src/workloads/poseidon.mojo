"""Poseidon over Mersenne-31, the expander's instance (width 16, rate 8, 8 full and 14 partial rounds, x^5, a
circulant MDS of small constants, keccak-derived round constants; a round adds the constants, applies the
matrix, then the S-box), on the bit layout of the hash workloads.

A chain is one lane of one round: 64 rows, row r holding bit 62 - r, row 63 idle (the Horner scan never
ingests the last row). The h2 = 16 R chains are 16 segments of R rounds, segment s round t at chain s R + t,
so the next round is the next chain and the other lanes of a round sit R chains apart, cyclically. The MDS
is anti-circulant, M[i][j] = row[(i + j) % 16], so round t holds lane sigma_t s at segment s with
sigma_t = (-1)^t and computes at segment s the output lane the next round wants there: the read at chain
offset e R then carries the coefficient row[(sigma_t e) % 16], uniform up to the round's parity, which
period-2 public columns supply bit by bit. Round 0 is idle (the state zero), rounds 1..22 the permutation,
the chunk added with the constants of round 1 (and 23 for a second chunk), and the digest is the S-box
output of the last round.

A lane-round: s = x + c + ab w (the round constant, a dense public column; the chunk word on an absorb
lane), folded (2^31 = 1 mod M31) to u < 2^31 + 4, with six shifted copies u << k; the matrix pile in two
halves (even and odd offsets: at most 23 and 17 bits per row) with 5-bit carries; y = hA + hB; two folds,
the second subtracting the modulus under a chain-constant bit so the honest state is canonical; then
p2 = mf mf, p4 = p2 p2, p5 = p4 mf, each a bit convolution certified by the Horner identity
R_mf^2 + rho R_p2^2 + rho^2 R_p4 R_mf = R_C (one coefficient accumulator, the products weighted by rho), rippled with 5-bit carries and folded twice below 2^31 + 5; v = mf or p5 by the S-box selector;
the next round's x reads v, cut at the end of a segment.

Every ripple is an exact integer identity: the carry into weight 0 is cut by the row selector nb, pile plus
carry stays below 127 and result plus twice the carry below 127, and a product's top carry is zero because
both operands are below 2^32. The folds read the high half through the selectors lo (weights 0..30) and hs
(weights 0..31) and are exact, so every value carries the honest bound: operands below 2^31 + 5 keep every
convolution coefficient below 32. The idle row's pile is masked everywhere, so its values are zero or
isolated."""

from core.params import Params
from relations.ir import CHAL_MUL
from relations.statement import Statement, Layout, Term, Read, BIT
from workloads.keccak import keccak256
from workload import Workload

comptime ROWS = 64
comptime T = 16             # width
comptime RATE = 8
comptime RF = 8
comptime RP = 14
comptime ROUNDS = RF + RP
comptime CBITS = 5          # coefficient and carry bits of a product ripple: both below 32
comptime MCARRY = 5         # matrix pile carries: a half pile holds at most 23 bits
comptime SHIFTS = 7         # the matrix constants are below 2^7
comptime M31 = (1 << 31) - 1
comptime ZETA = 2           # gamma: the evaluation point
comptime RHO = 1            # delta: the product weight
comptime DIGEST = 4 * T
comptime WEIGHTS = ROWS - 1         # rows 0..62 hold weights 62..0; row 63 is idle
comptime FOLD_READ = ROWS - 31      # k1 of the read 31 weights up
comptime PRODUCTS = 3


def mds_row() -> List[Int]:
    """M[i][j] = row[(i + j) % 16]."""
    var v: List[Int] = [1, 1, 51, 1, 11, 17, 2, 1, 101, 63, 15, 2, 67, 22, 13, 3]
    return v^


# ---- host reference ----

def m31(x: Int) -> Int:
    """x mod 2^31 - 1 for 0 <= x < 2^62."""
    var r = (x & M31) + (x >> 31)
    r = (r & M31) + (r >> 31)
    return 0 if r == M31 else r


def round_constants() raises -> List[Int]:
    """ROUNDS x T constants: a keccak chain from the seed, the low four bytes of each digest, reduced."""
    var seed = String("poseidon_seed_Mersenne 31_") + String(T)
    var buf = List[UInt8]()
    for b in seed.as_bytes():
        buf.append(b)
    buf = keccak256(buf)
    var out = List[Int](capacity=ROUNDS * T)
    for _ in range(ROUNDS * T):
        buf = keccak256(buf)
        out.append(m31(Int(buf[0]) | Int(buf[1]) << 8 | Int(buf[2]) << 16 | Int(buf[3]) << 24))
    return out^


def mds(state: List[Int]) -> List[Int]:
    var row = mds_row()
    var out = List[Int](capacity=T)
    for i in range(T):
        var acc = 0
        for j in range(T):
            acc += row[(i + j) % T] * state[j]
        out.append(m31(acc))
    return out^


def sbox(x: Int) -> Int:
    var x2 = m31(x * x)
    var x4 = m31(x2 * x2)
    return m31(x4 * x)


def full_round(r: Int) -> Bool:
    return r < RF // 2 or r >= RF // 2 + RP


def permute(mut state: List[Int], rc: List[Int]):
    """A round: the constants, the matrix, the S-box (every lane in a full round, lane 0 in a partial one)."""
    for r in range(ROUNDS):
        for j in range(T):
            state[j] = m31(state[j] + rc[r * T + j])
        state = mds(state)
        for j in range(T):
            if full_round(r) or j == 0:
                state[j] = sbox(state[j])


def chunks(inputs: List[Int]) -> List[Int]:
    """The inputs zero-padded to whole chunks of RATE."""
    var v = inputs.copy()
    while len(v) % RATE != 0:
        v.append(0)
    return v^


def poseidon_m31(inputs: List[Int]) raises -> List[Int]:
    """The sponge: each chunk added into the rate slots (the last RATE), a permutation each; the whole state
    is the digest."""
    if len(inputs) < 1 or len(inputs) > 2 * RATE:
        raise Error("Poseidon takes 1 to 16 inputs")
    for v in inputs:
        if v < 0 or v >= M31:
            raise Error("inputs are canonical M31 elements")
    var rc = round_constants()
    var state = List[Int](length=T, fill=0)
    var padded = chunks(inputs)
    for c in range(len(padded) // RATE):
        for k in range(RATE):
            state[T - RATE + k] = m31(state[T - RATE + k] + padded[c * RATE + k])
        permute(state, rc)
    return state^


# ---- statement ----

def _pre(P: Int) -> String:
    return "p" + String(P)


def _g(e: Int, k: Int) -> String:
    return "g" + String(e) + "b" + String(k)


def _g_needed(e: Int, k: Int) -> Bool:
    """Bit k of the coefficient at offset e on either parity."""
    var row = mds_row()
    return ((row[e] | row[(T - e) % T]) >> k) & 1 == 1


def _uk(k: Int) -> String:
    return "u" if k == 0 else "u" + String(k)


def _carries(prefix: String, n: Int) -> List[String]:
    var v = List[String](capacity=n)
    for k in range(n):
        v.append(prefix + String(k))
    return v^


def column_names() -> List[String]:
    var v: List[String] = ["x", "s", "sc0", "sc1"]
    for k in range(SHIFTS):
        v.append(_uk(k))
    v.append("uc")
    v.append("ha")
    v.extend(_carries("hac", MCARRY))
    v.append("hb")
    v.extend(_carries("hbc", MCARRY))
    for n in ["y", "yc", "mo", "moc", "mf", "mfc", "b"]:
        v.append(n)
    for P in [2, 4, 5]:
        for m in range(CBITS):
            v.append(_pre(P) + "c" + String(m))
        for k in range(CBITS):
            v.append(_pre(P) + "y" + String(k))
        for n in ["r", "o", "oc", "f", "fc"]:
            v.append(_pre(P) + n)
    v.append("v")
    v.append("w")
    return v^


def _sum(mut st: Statement, var terms: List[Term], result: String, carries: List[String], extra: List[Term] = List[Term]()) raises:
    """pile + carry in (from the row below, cut on the last row by nb) = out + 2 carry out (+ the signed extra
    terms), per row."""
    for k in range(len(carries)):
        terms.append(Term(1 << k, st.read("nb"), st.read(carries[k], k1=1)))
    terms.append(Term(-1, st.read(result)))
    for k in range(len(carries)):
        terms.append(Term(-(2 << k), st.read(carries[k])))
    terms.extend(extra.copy())
    st.family("=" + result, terms^)


def _fold(mut st: Statement, src: String, result: String, carry: String, extra: List[Term] = List[Term]()) raises:
    """out = (src mod 2^31) + (src >> 31): the low weights through lo, the high half read 31 weights up through hs."""
    _sum(st, [Term(1, st.read("lo"), st.read(src)), Term(1, st.read("hs"), st.read(src, k1=FOLD_READ))], result, [carry], extra)


def poseidon_statement[p: Params]() raises -> Statement:
    comptime h2 = p.h2()
    if p.h1() != ROWS or h2 % (2 * T) != 0 or h2 // T < ROUNDS + 1:
        raise Error("Poseidon needs 64 rows per chain and 16 segments of an even number of chains, more than 22")
    comptime R = h2 // T
    var st = Statement()
    for n in column_names():
        st.col(n, BIT, group="pos")
    for n in ["c", "sb", "ab", "dg", "ld"]:
        st.pub(n, 1)
    for n in ["z0", "nl"]:
        st.pub(n, T)                # period R: one value per round position
    for e in range(T):
        for k in range(SHIFTS):
            if _g_needed(e, k):
                st.pub(_g(e, k), h2 // 2)   # period 2: the round's parity
    for n in ["nb", "lo", "hs"]:
        st.pub(n, h2)
    var rz: List[Int] = [-1, RHO, st.derived(CHAL_MUL, RHO, RHO)]     # the product weights 1, rho, rho^2
    _sum(st, [Term(1, st.read("x")), Term(1, st.read("c")), Term(1, st.read("ab"), st.read("w"))], "s", ["sc0", "sc1"])
    _fold(st, "s", "u", "uc")
    for k in range(1, SHIFTS):
        st.family("=" + _uk(k), [Term(1, st.read(_uk(k))), Term(-1, st.read("u", k1=k))])
    var ha = List[Term]()
    var hb = List[Term]()
    for e in range(T):
        for k in range(SHIFTS):
            if _g_needed(e, k):
                var t = Term(1, st.read(_g(e, k)), st.read(_uk(k), k2=(e * R) % h2))
                if e % 2 == 0:
                    ha.append(t^)
                else:
                    hb.append(t^)
    _sum(st, ha^, "ha", _carries("hac", MCARRY))
    _sum(st, hb^, "hb", _carries("hbc", MCARRY))
    _sum(st, [Term(1, st.read("ha")), Term(1, st.read("hb"))], "y", ["yc"])
    _fold(st, "y", "mo", "moc")
    _fold(st, "mo", "mf", "mfc", [Term(-1, st.read("b"), st.read("lo"))])
    st.family("=b", [Term(1, st.read("b")), Term(-1, st.read("b", k1=1))])
    var ic = List[Term]()
    var ops: List[Int] = [2, 4, 5]
    for P in range(PRODUCTS):
        var pre = _pre(ops[P])
        var pile = List[Term]()
        for m in range(CBITS):
            pile.append(Term(1 << m, st.read(pre + "c" + String(m))))
            ic.append(Term(1 << m, st.read(pre + "c" + String(m)), chal=rz[P]))
        _sum(st, pile^, pre + "r", _carries(pre + "y", CBITS))
        _fold(st, pre + "r", pre + "o", pre + "oc")
        _fold(st, pre + "o", pre + "f", pre + "fc")
    st.horner("rm", [Term(1, st.read("mf"))], scale=ZETA)
    st.horner("rp2", [Term(1, st.read("p2f"))], scale=ZETA)
    st.horner("rp4", [Term(1, st.read("p4f"))], scale=ZETA)
    st.horner("rc", ic, scale=ZETA)
    st.chain_end("sbox", [Term(1, st.read("rm"), st.read("rm"), chal=rz[0]), Term(1, st.read("rp2"), st.read("rp2"), chal=rz[1]),
                          Term(1, st.read("rp4"), st.read("rm"), chal=rz[2]), Term(-1, st.read("rc"))])
    st.family("=v", [Term(1, st.read("v")), Term(-1, st.read("mf")), Term(-1, st.read("sb"), st.read("p5f")), Term(1, st.read("sb"), st.read("mf"))])
    st.family("=x", [Term(1, st.read("nl"), st.read("x", k2=1)), Term(-1, st.read("nl"), st.read("v"))])
    st.family("z0x", [Term(1, st.read("z0"), st.read("x"))])
    st.family("wbd", [Term(1, st.read("w")), Term(-1, st.read("lo"), st.read("w"))])
    st.family("=dg", [Term(1, st.read("ld"), st.read("v")), Term(-1, st.read("dg"))])
    return st^


# ---- trace ----

def _ripple(pile: List[Int], bits: Int, rhs: Int = 0) raises -> Tuple[Int, List[Int]]:
    """pile[w] + carry in = out_w + rhs_w + 2 carry out per weight: the output and the carry bit columns (bit k of
    the carry out of weight w at weight w). Raises on a negative step, a carry past `bits`, or a top carry."""
    var out = 0
    var carries = List[Int](length=bits, fill=0)
    var cin = 0
    for w in range(WEIGHTS):
        var s = pile[w] + cin - ((rhs >> w) & 1)
        if s < 0:
            raise Error("ripple: the pile is below its subtrahend")
        out |= (s & 1) << w
        var cout = s >> 1
        if cout >= (1 << bits):
            raise Error("ripple: carry overflow")
        for k in range(bits):
            carries[k] |= ((cout >> k) & 1) << w
        cin = cout
    if cin != 0:
        raise Error("ripple: carry out of the top weight")
    return (out, carries^)


def _fold_pile(r: Int) -> List[Int]:
    """(r mod 2^31) + (r >> 31) as a pile: the low weights and the high half 31 weights up, r below 2^63."""
    var pile = List[Int](length=ROWS, fill=0)
    for w in range(ROWS):
        if w <= 30:
            pile[w] += (r >> w) & 1
        if w <= 31:
            pile[w] += (r >> (w + 31)) & 1
    return pile^


def _set(mut vals: Dict[String, Int], prefix: String, carries: List[Int]):
    for k in range(len(carries)):
        vals[prefix + String(k)] = carries[k]


def _fold_into(mut vals: Dict[String, Int], src: Int, result: String, carry: String, rhs: Int = 0) raises -> Int:
    var f = _ripple(_fold_pile(src), 1, rhs)
    vals[result] = f[0]
    vals[carry] = f[1][0]
    return f[0]


def _product(mut vals: Dict[String, Int], pre: String, a: Int, b: Int, cheat: Bool = False) raises -> Int:
    """a b below 2^63 as coefficient bits, the ripple to r, two folds; returns the folded product. `cheat`
    adds one to the constant coefficient: every row family still holds, only the fingerprint fails."""
    var c = List[Int](length=ROWS, fill=0)
    for i in range(32):
        if (a >> i) & 1 == 0:
            continue
        for j in range(32):
            c[i + j] += (b >> j) & 1
    if cheat:
        c[0] += 1
    for m in range(CBITS):
        var col = 0
        for w in range(ROWS):
            col |= ((c[w] >> m) & 1) << w
        vals[pre + "c" + String(m)] = col
    var r = _ripple(c, CBITS)
    if r[0] != a * b and not cheat:
        raise Error("product ripple disagrees with the product")
    vals[pre + "r"] = r[0]
    _set(vals, pre + "y", r[1])
    var o = _fold_into(vals, r[0], pre + "o", pre + "oc")
    return _fold_into(vals, o, pre + "f", pre + "fc")


def _lane(sigma: Int, s: Int) -> Int:
    """The lane stored at segment s: sigma s mod 16."""
    return s if sigma > 0 else (T - s) % T


def _perm_round(t: Int, perms: Int) -> Int:
    """The permutation round of round position t (-1 outside the permutations)."""
    return (t - 1) % ROUNDS if t >= 1 and t <= ROUNDS * perms else -1


def _sbox_on(t: Int, lane: Int, perms: Int) -> Bool:
    var r = _perm_round(t, perms)
    return r >= 0 and (full_round(r) or lane == 0)


def _absorb(t: Int, perms: Int) -> Int:
    """The chunk absorbed at round position t, -1 for none: chunk 0 with round 1, chunk 1 with round 23."""
    if t == 1:
        return 0
    if perms == 2 and t == ROUNDS + 1:
        return 1
    return -1


def _perms(n: Int) raises -> Int:
    if n < 1 or n > 2 * RATE:
        raise Error("Poseidon takes 1 to 16 inputs")
    return 1 if n <= RATE else 2


def poseidon_trace[p: Params](layout: Layout, inputs: List[Int], cheat: Bool = False) raises -> List[UInt8]:
    """Columns (column, chain, row) of bits: the sponge walked round by round, every segment a lane. `cheat`
    corrupts the first product's coefficients on the lanes whose S-box is off (nothing downstream reads
    them), so only the chain-end identity can reject the trace."""
    comptime N = p.N()
    comptime h2 = p.h2()
    comptime R = h2 // T
    var perms = _perms(len(inputs))
    if p.h1() != ROWS or R < ROUNDS * perms + 1:
        raise Error("the grid does not hold the permutations")
    for v in inputs:
        if v < 0 or v >= M31:
            raise Error("inputs are canonical M31 elements")
    var rc = round_constants()
    var padded = chunks(inputs)
    var row = mds_row()
    var names = column_names()
    var cols = List[Int](capacity=len(names))
    for n in names:
        cols.append(layout.col(n))
    var trace = List[UInt8](length=layout.columns_w() * N, fill=0)
    var x = List[Int](length=T, fill=0)
    for t in range(R):
        var sigma = 1 if t % 2 == 0 else -1
        var chain = List[Dict[String, Int]]()
        var u = List[Int](length=T, fill=0)
        for s in range(T):
            var lane = _lane(sigma, s)
            var vals = Dict[String, Int]()
            var r = _perm_round(t, perms)
            var c = rc[r * T + lane] if r >= 0 else 0
            var chunk = _absorb(t, perms)
            var wv = padded[chunk * RATE + lane - RATE] if chunk >= 0 and lane >= RATE else 0
            vals["x"] = x[s]
            vals["w"] = wv
            var pile = List[Int](length=ROWS, fill=0)
            for w in range(WEIGHTS):
                pile[w] = ((x[s] >> w) & 1) + ((c >> w) & 1) + ((wv >> w) & 1)
            var sm = _ripple(pile, 2)
            vals["s"] = sm[0]
            _set(vals, "sc", sm[1])
            u[s] = _fold_into(vals, sm[0], "u", "uc")
            for k in range(1, SHIFTS):
                vals[_uk(k)] = u[s] << k
            chain.append(vals^)
        var next = List[Int](length=T, fill=0)
        for s in range(T):
            var piles: List[List[Int]] = [List[Int](length=ROWS, fill=0), List[Int](length=ROWS, fill=0)]
            for e in range(T):
                var coef = row[_lane(sigma, e)]
                var src = u[(s + e) % T]
                for k in range(SHIFTS):
                    if (coef >> k) & 1 == 0:
                        continue
                    for w in range(k, WEIGHTS):
                        piles[e % 2][w] += (src >> (w - k)) & 1
            var ha = _ripple(piles[0], MCARRY)
            var hb = _ripple(piles[1], MCARRY)
            chain[s]["ha"] = ha[0]
            chain[s]["hb"] = hb[0]
            _set(chain[s], "hac", ha[1])
            _set(chain[s], "hbc", hb[1])
            var ypile = List[Int](length=ROWS, fill=0)
            for w in range(WEIGHTS):
                ypile[w] = ((ha[0] >> w) & 1) + ((hb[0] >> w) & 1)
            var y = _ripple(ypile, 1)
            chain[s]["y"] = y[0]
            chain[s]["yc"] = y[1][0]
            var mo = _fold_into(chain[s], y[0], "mo", "moc")
            var b = 1 if (mo & M31) + (mo >> 31) >= M31 else 0
            var mf = _fold_into(chain[s], mo, "mf", "mfc", b * M31)
            chain[s]["b"] = b
            var out_lane = _lane(-sigma, s)
            var on = _sbox_on(t, out_lane, perms)
            var p2 = _product(chain[s], _pre(2), mf, mf, cheat and not on)
            var p4 = _product(chain[s], _pre(4), p2, p2)
            var p5 = _product(chain[s], _pre(5), p4, mf)
            var v = p5 if on else mf
            chain[s]["v"] = v
            next[s] = v
        for s in range(T):
            var at = (s * R + t) * ROWS
            for i in range(len(names)):
                var val = chain[s][names[i]]
                if names[i] == "b":
                    for r in range(ROWS):
                        trace[cols[i] * N + at + r] = UInt8(val)
                    continue
                if val < 0 or val >> WEIGHTS != 0:
                    raise Error("a column value needs 63 bits")
                for r in range(WEIGHTS):
                    trace[cols[i] * N + at + r] = UInt8((val >> (WEIGHTS - 1 - r)) & 1)
        x = next^
    return trace^


# ---- public data ----

def _bits(v: Int) -> List[UInt8]:
    """Row r holds bit 62 - r; row 63 is zero."""
    var out = List[UInt8](capacity=ROWS)
    for r in range(WEIGHTS):
        out.append(UInt8((v >> (WEIGHTS - 1 - r)) & 1))
    out.append(0)
    return out^


def _fill(v: Int) -> List[UInt8]:
    return List[UInt8](length=ROWS, fill=UInt8(v))


def poseidon_public_columns[p: Params](n: Int, digest: List[Int]) raises -> List[List[UInt8]]:
    """One period per public column, in declaration order. The absorb selector admits only the n input
    slots: the padding slots stay zero by the selector, not by the prover's word."""
    comptime h2 = p.h2()
    comptime R = h2 // T
    var perms = _perms(n)
    var rc = round_constants()
    var row = mds_row()
    var last = ROUNDS * perms
    var out = List[List[UInt8]]()
    for l in range(5):
        var col = List[UInt8](capacity=p.N())
        for s in range(T):
            for t in range(R):
                var sigma = 1 if t % 2 == 0 else -1
                var lane = _lane(sigma, s)
                var r = _perm_round(t, perms)
                if l == 0:
                    col.extend(_bits(rc[r * T + lane] if r >= 0 else 0))
                elif l == 1:
                    col.extend(_fill(1 if _sbox_on(t, _lane(-sigma, s), perms) else 0))
                elif l == 2:
                    var chunk = _absorb(t, perms)
                    col.extend(_fill(1 if chunk >= 0 and lane >= RATE and chunk * RATE + lane - RATE < n else 0))
                elif l == 3:
                    col.extend(_bits(digest[_lane(-sigma, s)] if t == last else 0))
                else:
                    col.extend(_fill(1 if t == last else 0))
        out.append(col^)
    var z0 = List[UInt8](capacity=R * ROWS)
    var nl = List[UInt8](capacity=R * ROWS)
    for t in range(R):
        z0.extend(_fill(1 if t == 0 else 0))
        nl.extend(_fill(0 if t == R - 1 else 1))
    out.append(z0^)
    out.append(nl^)
    for e in range(T):
        for k in range(SHIFTS):
            if _g_needed(e, k):
                var g = List[UInt8](capacity=2 * ROWS)
                g.extend(_fill((row[e] >> k) & 1))
                g.extend(_fill((row[(T - e) % T] >> k) & 1))
                out.append(g^)
    var nb = List[UInt8](capacity=ROWS)
    var lo = List[UInt8](capacity=ROWS)
    var hs = List[UInt8](capacity=ROWS)
    for r in range(ROWS):
        var w = WEIGHTS - 1 - r            # -1 on the idle row
        nb.append(UInt8(0 if w <= 0 else 1))
        lo.append(UInt8(1 if w >= 0 and w <= 30 else 0))
        hs.append(UInt8(1 if w >= 0 and w <= 31 else 0))
    out.append(nb^)
    out.append(lo^)
    out.append(hs^)
    return out^


def digest_bytes(digest: List[Int]) -> List[UInt8]:
    var out = List[UInt8](capacity=DIGEST)
    for d in digest:
        for i in range(4):
            out.append(UInt8((d >> (8 * i)) & 0xFF))
    return out^


@fieldwise_init
struct Poseidon(Workload, Copyable, Movable):
    """Poseidon-M31 of 1 to 16 field elements on a grid of 64 x 16 R, R > 22 (one chunk) or 44 (two).
    Public inputs: the input count, then the 16-word digest, 4 little-endian bytes each."""
    var inputs: List[Int]

    def statement[p: Params](self) raises -> Statement:
        return poseidon_statement[p]()

    def trace[p: Params](self, layout: Layout) raises -> List[UInt8]:
        return poseidon_trace[p](layout, self.inputs)

    def public_inputs[p: Params](self) raises -> List[UInt8]:
        var v: List[UInt8] = [UInt8(len(self.inputs))]
        v.extend(digest_bytes(poseidon_m31(self.inputs)))
        return v^

    @staticmethod
    def public_data[p: Params](layout: Layout, public_inputs: List[UInt8]) raises -> List[UInt8]:
        if len(public_inputs) != 1 + DIGEST:
            raise Error("public inputs are the input count then the 64-byte digest")
        var n = Int(public_inputs[0])
        var perms = _perms(n)
        if p.h2() // T < ROUNDS * perms + 1:
            raise Error("the grid does not hold the permutations")
        var digest = List[Int](capacity=T)
        for i in range(T):
            var d = 0
            for j in range(4):
                d |= Int(public_inputs[1 + 4 * i + j]) << (8 * j)
            if d >= M31:
                raise Error("digest words are canonical M31 elements")
            digest.append(d)
        var data = List[UInt8]()
        for col in poseidon_public_columns[p](n, digest):
            data.extend(col.copy())
        return data^
