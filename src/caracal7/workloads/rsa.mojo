"""RSA verify on the product chain: s^e mod n = m for e = 2^(muls - 1) + 1 (65537 for muls = 17), n, s and m
of `limbs` 256-bit limbs. Every modmul a b = q n + r is an integer identity over limb products, checked
with running sums along chains and no additions in the wiring.

A modmul takes two blocks of limbs (limbs + 1) chains: AB holds the products a_i b_j on rows i < limbs, QN
the products q_i n_j (q the quotient, a hint the prover supplies; n public); row i = limbs holds no product
(its piece and bound selectors are zero, so a = b = 0 there, as on the idle chains, and the chain-end
identity zeta^6 R_A R_B = R_C then forces r = 0: the sum lane's read of lo(r) carries no mask of its own).
Chain (i, j) sits at y = (limbs + 1)(limbs - 1 - j) + (limbs - i) of its block, so every read below shifts
forward on axis 2. Its product r = a_i b_j < 2^512 splits into lo (weights below 256) and hi; the sum lane
value t of chain (i, j) is

    t = cont lo(t of (i + 1, j - 1)) + hi(t of (i - 1, j)) + hi(r of (i - 1, j)) + lo(r of (i, j))

read at k2 = limbs (the next chain of the same anti-diagonal, `cl` masks the tail) and k2 = 1 with the UP
row shift (the chain below, `hm` and `rm` mask row 0). Along the anti-diagonal i + j = p the lo halves of t
carry the running sum of position p, each chain's hi (below 4, since t < 2^258) is passed to position p + 1
on the chain above, and the head of the diagonal ((0, p) for p < limbs, (p - limbs + 1, limbs - 1) after)
holds limb p of the block's total in lo(t). The hi of a row-limbs chain is read by nobody, and it is zero:
that t is hi(t) + hi(r) of the chain below, so hi(t) >= 1 there needs hi(r) = 2^256 - 2 (a and b both
2^256 - 1, by the bounds bd and s2 at 256 bits) and hi(t) = 2 on the chain below, which by the same
descent needs hi(t) = 2 on every chain of the column down to row 0, where hi(t) <= 1. The ripple is an exact integer identity with signed 4-bit carries and a zero row on the
idle row's carry.

The compare lane on the AB heads: u_p = lo(t_ab) + K_p + hi(u_(p-1)) - lo(t_qn) - r_p with K_0 = 2^258 + 4
and K_p = 2^258 after, lo(t_qn) wired from the QN head (slot cy), r_p a hint (slot cz, zero above limb
limbs - 1 by `zm`) and the carry read from the previous head (k2 = limbs + 1 or 1, `m9` and `m1`). The
families force lo(u_p) = 4 on every head and hi(u) = 4 on the last one, which telescopes to
sum_p (lo(t_ab) - lo(t_qn) - r_p) 2^(256 p) = 0, so a b = q n + r as integers. u stays below 2^259 for any
witness (the bias exceeds the two subtracted limbs), so it is a bit vector. r is bound below 2^(256 limbs),
not below n: the statement proves s^e = m (mod n) for the public m, which a verifier supplies canonical.

Wiring: modmul 0 takes s as a and b (public factors on ha and rb); modmul m takes r of m - 1 (the cz hints
of its heads) as a and b, the last one r as a and s as b. q_i's occurrences are wired to their first; n_j
is a public factor on every QN chain; the last modmul's r_p are the public factors m_p. Slots: ha, rb, tl
(lo(t) of a head), cy, cz. Public inputs: limbs, muls, then s, n, m as limbs x 32 bytes little endian
(pinned head: limbs and muls)."""

from std.memory import unsafe_memcpy
from max.algorithm import parallelize
from caracal7.core.params import Params
from caracal7.relations.ir import FIX_E
from caracal7.relations.statement import Statement, Layout, Term, BIT
from caracal7.workloads.bigint import Big
from caracal7.workloads.mulmod import ROWS, Q, PIECES, PIECE, SLOTS, BITS, UP, VALUE, FOLDED, WIDTH, MUL, OpValues, product_columns, product_certificate, plain_fingerprint, _row, _cols, _put, _columns, _mul_chain
from caracal7.workload import Workload

comptime LIMB = BITS
comptime SCARRY = 4         # signed ripple carry bits: c0 + 2 c1 + 4 c2 - 8 c3
comptime BIAS = 258         # K = 2^BIAS (+ 4 on head 0): above the two subtracted limbs
comptime SLOT_HA = 0
comptime SLOT_RB = 1
comptime SLOT_TL = 2
comptime SLOT_CY = 3
comptime SLOT_CZ = 4
comptime SRC_S = 0          # public factor sources
comptime SRC_N = 1
comptime SRC_M = 2


def _idx(limbs: Int, i: Int, j: Int) -> Int:
    """Chain y of (i, j) in its block: i in [0, limbs], j in [0, limbs)."""
    return (limbs + 1) * (limbs - 1 - j) + (limbs - i)


def _ij(limbs: Int, y: Int) -> Tuple[Int, Int]:
    return (limbs - y % (limbs + 1), limbs - 1 - y // (limbs + 1))


def _head_y(limbs: Int, p: Int) -> Int:
    """The chain holding limb p of a block's total."""
    if p < limbs:
        return _idx(limbs, 0, p)
    return _idx(limbs, p - limbs + 1, limbs - 1)


def _is_head(limbs: Int, i: Int, j: Int) -> Bool:
    return i == 0 or j == limbs - 1


def _block(limbs: Int) -> Int:
    return limbs * (limbs + 1)


def chain_count(limbs: Int, muls: Int) -> Int:
    return 2 * muls * _block(limbs)


def _chain(limbs: Int, m: Int, block: Int, y: Int) -> Int:
    return (2 * m + block) * _block(limbs) + y


def _weight_rows(lo: Int, hi: Int, x1: Int) -> Bool:
    """Row x1 holds weights in [lo, hi) (the idle row none)."""
    if x1 > ROWS - 2:
        return False
    var w = (ROWS - 2 - x1) * Q
    return w >= lo and w < hi


def _factors(limbs: Int, muls: Int) -> List[Tuple[Int, Int, Int, Int]]:
    """(chain, slot, source, limb) of every public factor in statement order."""
    var v = List[Tuple[Int, Int, Int, Int]]()
    for m in range(muls):
        for y in range(_block(limbs)):
            var ij = _ij(limbs, y)
            if ij[0] == limbs:
                continue
            var c = _chain(limbs, m, 0, y)
            if m == 0:
                v.append((c, SLOT_HA, SRC_S, ij[0]))
            if m == 0 or m == muls - 1:
                v.append((c, SLOT_RB, SRC_S, ij[1]))
            v.append((_chain(limbs, m, 1, y), SLOT_RB, SRC_N, ij[1]))
        if m == muls - 1:
            for p in range(limbs):
                v.append((_chain(limbs, m, 0, _head_y(limbs, p)), SLOT_CZ, SRC_M, p))
    return v^


def _wires(limbs: Int, muls: Int) -> List[Tuple[Int, Int, Int, Int]]:
    """(slot, chain, slot, chain) of every wire."""
    var v = List[Tuple[Int, Int, Int, Int]]()
    for m in range(muls):
        for y in range(_block(limbs)):
            var ij = _ij(limbs, y)
            if ij[0] == limbs:
                continue
            var c = _chain(limbs, m, 0, y)
            if m >= 1:
                v.append((SLOT_HA, c, SLOT_CZ, _chain(limbs, m - 1, 0, _head_y(limbs, ij[0]))))
                if m < muls - 1:
                    v.append((SLOT_RB, c, SLOT_CZ, _chain(limbs, m - 1, 0, _head_y(limbs, ij[1]))))
            if ij[1] >= 1:
                v.append((SLOT_HA, _chain(limbs, m, 1, y), SLOT_HA, _chain(limbs, m, 1, _idx(limbs, ij[0], 0))))
        for p in range(2 * limbs):
            v.append((SLOT_CY, _chain(limbs, m, 0, _head_y(limbs, p)), SLOT_TL, _chain(limbs, m, 1, _head_y(limbs, p))))
    return v^


def _carry(mut st: Statement, prefix: String, j: Int, sign: Int, k1: Int) -> List[Term]:
    """sign (c0 + 2 c1 + 4 c2 - 8 c3) at position j of the row k1 rows down."""
    var v = List[Term]()
    for k in range(SCARRY):
        v.append(Term(sign * (-8 if k == SCARRY - 1 else 1 << k), st.read(prefix + String(k) + String(j), k1=k1)))
    return v^


def _ripple(mut st: Statement, name: String, dest: String, carry: String, mut terms: List[List[Term]]) raises:
    """Per position: the terms + carry in = out + 2 carry out, signed carries, a zero row on the idle row's
    carry (position Q - 1)."""
    for j in range(Q):
        var t = terms[j].copy()
        if j > 0:
            t.extend(_carry(st, carry, j - 1, 1, 0))
        else:
            t.extend(_carry(st, carry, Q - 1, 1, 1))
        t.append(Term(-1, st.read(dest + String(j))))
        t.extend(_carry(st, carry, j, -2, 0))
        st.family(name + String(j), t)
    for k in range(SCARRY):
        st.zero(carry + String(k) + String(Q - 1), FIX_E)


def _bound(mut st: Statement, name: String, mask: String) raises:
    for j in range(Q):
        st.family(name + "b" + String(j), [Term(1, st.read(name + String(j))), Term(-1, st.read(mask), st.read(name + String(j)))])


def rsa_statement(limbs: Int, muls: Int) raises -> Statement:
    if limbs < 1 or limbs > 255 or muls < 1 or muls > 255:
        raise Error("limbs and muls are bytes, at least 1")
    var st = Statement()
    product_columns(st)
    for name in ["t", "u", "cy", "cz"]:
        for j in range(Q):
            st.col(name + String(j), BIT)
    for name in ["sc", "uc"]:
        for k in range(SCARRY):
            for j in range(Q):
                st.col(name + String(k) + String(j), BIT)
    for name in ["lo", "bd", "s0", "s1", "s2", "bu", "cl", "hm", "rm", "ch", "m9", "m1", "mt", "zm", "ym", "um"]:
        st.pub(name, 1)
    for name in ["kc", "k4", "kt"]:
        for j in range(Q):
            st.pub(name + String(j), 1)
    st.pin([UInt8(limbs), UInt8(muls)])
    var rz = product_certificate(st)
    var tl = List[Term]()
    for j in range(Q):
        tl.append(Term(1, st.read("t" + String(j)), st.read("lo"), chal=rz[j]))
    st.horner("tl", tl, scale=rz[Q])
    plain_fingerprint(st, "cy", "cy", rz)
    plain_fingerprint(st, "cz", "cz", rz)
    var sum_terms = List[List[Term]]()
    for j in range(Q):
        var t: List[Term] = [Term(1, st.read("t" + String(j), k2=limbs), st.read("cl")),
                             Term(1, st.read("t" + String(j), k1=UP, k2=1), st.read("hm")),
                             Term(1, st.read("r" + String(j), k1=UP, k2=1), st.read("rm")),
                             Term(1, st.read("r" + String(j)), st.read("lo"))]
        sum_terms.append(t^)
    _ripple(st, "sum", "t", "sc", sum_terms)
    var cmp_terms = List[List[Term]]()
    for j in range(Q):
        var t: List[Term] = [Term(1, st.read("t" + String(j)), st.read("ch")),
                             Term(1, st.read("kc" + String(j))),
                             Term(1, st.read("u" + String(j), k1=UP, k2=limbs + 1), st.read("m9")),
                             Term(1, st.read("u" + String(j), k1=UP, k2=1), st.read("m1")),
                             Term(-1, st.read("cy" + String(j))),
                             Term(-1, st.read("cz" + String(j)))]
        cmp_terms.append(t^)
    _ripple(st, "cmp", "u", "uc", cmp_terms)
    for j in range(Q):
        st.family("ulo" + String(j), [Term(1, st.read("u" + String(j)), st.read("ch")), Term(-1, st.read("k4" + String(j)))])
        st.family("uhi" + String(j), [Term(1, st.read("u" + String(j), k1=UP), st.read("mt")), Term(-1, st.read("kt" + String(j)))])
    _bound(st, "t", "bu")
    _bound(st, "u", "um")
    _bound(st, "cy", "ym")
    _bound(st, "cz", "zm")
    for name in ["ha", "rb", "tl", "cy", "cz"]:
        _ = st.slot(name)
    var names: List[String] = ["ha", "rb", "tl", "cy", "cz"]
    var fs = _factors(limbs, muls)
    for i in range(len(fs)):
        st.public_factor("f" + String(i), names[fs[i][1]], fs[i][1], fs[i][0])
    for w in _wires(limbs, muls):
        st.wire(w[0], w[1], w[2], w[3])
    return st^


# ---- host ----

def limbs_of(v: Big, limbs: Int) raises -> List[Big]:
    var out = List[Big]()
    for i in range(limbs):
        out.append(v.shr(LIMB * i).low(LIMB))
    return out^


def _lo(v: Big) -> Big:
    return v.low(LIMB)


def _hi(v: Big) -> Big:
    return v.shr(LIMB)


@fieldwise_init
struct _Chains(Copyable, Movable):
    """Per chain of one modmul: the product operands, the product, the sum lane value; on AB heads the
    compare lane's operands and value."""
    var a: List[Big]
    var b: List[Big]
    var r: List[Big]
    var t: List[Big]
    var u: List[Big]
    var cy: List[Big]
    var cz: List[Big]


def _modmul_chains(limbs: Int, a: List[Big], b: List[Big], q: List[Big], n: List[Big], r: List[Big]) raises -> _Chains:
    var B = _block(limbs)
    var v = _Chains(List[Big](length=2 * B, fill=Big()), List[Big](length=2 * B, fill=Big()), List[Big](length=2 * B, fill=Big()),
                    List[Big](length=2 * B, fill=Big()), List[Big](length=2 * B, fill=Big()), List[Big](length=2 * B, fill=Big()),
                    List[Big](length=2 * B, fill=Big()))
    for block in range(2):
        for y in range(B - 1, -1, -1):
            var ij = _ij(limbs, y)
            var i = ij[0]
            var j = ij[1]
            var x = block * B + y
            if i < limbs:
                v.a[x] = a[i].copy() if block == 0 else q[i].copy()
                v.b[x] = b[j].copy() if block == 0 else n[j].copy()
                v.r[x] = v.a[x] * v.b[x]
            var t = _lo(v.r[x])
            if i < limbs and j >= 1:
                t = t + _lo(v.t[x + limbs])
            if i >= 1:
                t = t + _hi(v.t[x + 1]) + _hi(v.r[x + 1])
            if t.bit_length() > BIAS:
                raise Error("sum lane value exceeds 2^" + String(BIAS))
            v.t[x] = t^
    var carry = Big(4)
    for p in range(2 * limbs):
        var x = _head_y(limbs, p)
        v.cy[x] = _lo(v.t[B + x])
        v.cz[x] = r[p].copy() if p < limbs else Big()
        var u = _lo(v.t[x]) + Big(1).shl(BIAS) + carry - v.cy[x] - v.cz[x]
        if u.neg or _lo(u) != Big(4):
            raise Error("compare lane does not close at limb " + String(p))
        carry = _hi(u)
        v.u[x] = u^
    if carry != Big(4):
        raise Error("compare lane does not close at the top")
    return v^


def _lane_carries[p: Params](mut chain: List[UInt8], cols: List[Int], pile: List[Int], dest: List[Int]) raises:
    """The signed carry columns `cols` (SCARRY x Q, k-major) of a ripple whose per-slot pile and output bits
    are given."""
    comptime h1 = p.h1()
    var c = 0
    for w in range(SLOTS):
        var l = c + pile[w]
        var r = dest[w] if w < len(dest) else 0
        if (l - r) % 2 != 0:
            raise Error("ripple carry is not integral")
        c = (l - r) // 2
        if c < -8 or c > 7:
            raise Error("ripple carry out of range")
        var e = c + 16 if c < 0 else c
        for k in range(SCARRY):
            chain[cols[k * Q + w % Q] * h1 + _row(w)] = UInt8((e >> k) & 1)
    if c != 0:
        raise Error("ripple does not close")


def _add_bits(mut pile: List[Int], v: Big, sign: Int, shift: Int, limit: Int) raises:
    """pile[w] += sign bit_(w + shift)(v) for w < limit."""
    var bits = v.bits(v.bit_length())
    for w in range(limit):
        var i = w + shift
        if i < len(bits) and bits[i] != 0:
            pile[w] += sign


def rsa_trace[p: Params](layout: Layout, limbs: Int, muls: Int, s: Big, n: Big, m: Big) raises -> List[UInt8]:
    comptime h1 = p.h1()
    comptime N = p.N()
    if h1 != ROWS or chain_count(limbs, muls) > p.h2():
        raise Error("the instance needs " + String(ROWS) + " rows per chain and " + String(chain_count(limbs, muls)) + " chains")
    if s >= n or m >= n or n.bit_length() > LIMB * limbs:
        raise Error("s and m below n, n of at most " + String(LIMB * limbs) + " bits")
    var B = _block(limbs)
    var per = List[_Chains]()
    var a = s.copy()
    for k in range(muls):
        var b = s.copy() if k == 0 or k == muls - 1 else a.copy()
        var qr = (a * b).divmod(n)
        per.append(_modmul_chains(limbs, limbs_of(a, limbs), limbs_of(b, limbs), limbs_of(qr[0], limbs), limbs_of(n, limbs), limbs_of(qr[1], limbs)))
        a = qr[1].copy()
    if a != m:
        raise Error("the signature does not verify")
    var columns = layout.columns_w()
    var trace = List[UInt8](length=columns * N, fill=0)
    var ca = _cols(layout, "a", True)
    var cb = _cols(layout, "b", False)
    var cr = _cols(layout, "r", False)
    var cc = List[Int]()
    for t in range(PIECES):
        for mm in range(7):
            cc.extend(_cols(layout, "c" + String(t) + String(mm), False))
    var cy = List[Int]()
    for k in range(5):
        cy.extend(_cols(layout, "y" + String(k), False))
    var ct = _cols(layout, "t", False)
    var cu = _cols(layout, "u", False)
    var ccy = _cols(layout, "cy", False)
    var ccz = _cols(layout, "cz", False)
    var csc = List[Int]()
    var cuc = List[Int]()
    for k in range(SCARRY):
        csc.extend(_cols(layout, "sc" + String(k), False))
        cuc.extend(_cols(layout, "uc" + String(k), False))
    var failed = List[Int](length=p.h2(), fill=0)
    var errors = List[String](length=p.h2(), fill=String(""))

    @parameter
    def one_chain(x2: Int):
        if x2 >= chain_count(limbs, muls):
            return
        var k = x2 // (2 * B)
        var x = x2 % (2 * B)
        var block = x // B
        var y = x % B
        var ij = _ij(limbs, y)
        var i = ij[0]
        var j = ij[1]
        var chain = List[UInt8](length=columns * h1, fill=0)
        try:
            ref v = per[k]
            if i < limbs:
                _mul_chain[p](chain, False, OpValues(MUL, v.a[x].copy(), v.b[x].copy(), Big(), Big(), 0), True, ca, cb, cr, cc, cy)
            _put(chain, ct, v.t[x].bits(BIAS), False)
            var pile = List[Int](length=SLOTS, fill=0)
            _add_bits(pile, v.r[x], 1, 0, LIMB)
            if i < limbs and j >= 1:
                _add_bits(pile, v.t[x + limbs], 1, 0, LIMB)
            if i >= 1:
                _add_bits(pile, v.t[x + 1], 1, LIMB, Q)
                _add_bits(pile, v.r[x + 1], 1, LIMB, LIMB)
            _lane_carries[p](chain, csc, pile, v.t[x].bits(BIAS))
            if block == 0 and _is_head(limbs, i, j):
                var pp = i + j
                _put(chain, cu, v.u[x].bits(BIAS + 1), False)
                _put(chain, ccy, v.cy[x].bits(LIMB), False)
                _put(chain, ccz, v.cz[x].bits(LIMB), False)
                var cp = List[Int](length=SLOTS, fill=0)
                _add_bits(cp, v.t[x], 1, 0, LIMB)
                cp[BIAS] += 1
                if pp == 0:
                    cp[2] += 1
                else:
                    _add_bits(cp, v.u[_head_y(limbs, pp - 1)], 1, LIMB, Q)
                _add_bits(cp, v.cy[x], -1, 0, LIMB)
                _add_bits(cp, v.cz[x], -1, 0, LIMB)
                _lane_carries[p](chain, cuc, cp, v.u[x].bits(BIAS + 1))
        except e:
            failed[x2] = 1
            errors[x2] = String(e)
            return
        for c in range(columns):
            unsafe_memcpy(dest=trace.unsafe_ptr().unsafe_offset(c * N + x2 * h1), src=chain.unsafe_ptr().unsafe_offset(c * h1), count=h1)

    parallelize[one_chain](p.h2())
    for x2 in range(p.h2()):
        if failed[x2] != 0:
            raise Error(errors[x2])
    return trace^


def rsa_inputs(limbs: Int, muls: Int, s: Big, n: Big, m: Big) raises -> List[UInt8]:
    var v: List[UInt8] = [UInt8(limbs), UInt8(muls)]
    v.extend(s.bytes(LIMB // 8 * limbs))
    v.extend(n.bytes(LIMB // 8 * limbs))
    v.extend(m.bytes(LIMB // 8 * limbs))
    return v^


def rsa_public_data[p: Params](layout: Layout, inputs: List[UInt8]) raises -> List[UInt8]:
    """The public columns in declaration order (N bytes each, x2 major), then the factors' ingest columns."""
    comptime N = p.N()
    if len(inputs) < 2:
        raise Error("public inputs start with limbs and muls")
    var limbs = Int(inputs[0])
    var muls = Int(inputs[1])
    var lb = LIMB // 8 * limbs
    if p.h1() != ROWS or len(inputs) != 2 + 3 * lb or chain_count(limbs, muls) > p.h2():
        raise Error("public inputs are limbs, muls, then s, n, m of limbs x 32 bytes; the grid holds every chain")
    var vals = List[List[Big]]()
    for k in range(3):
        var bytes = List[UInt8]()
        for i in range(lb):
            bytes.append(inputs[2 + k * lb + i])
        vals.append(limbs_of(Big.from_bytes(bytes), limbs))
    var B = _block(limbs)
    var data = List[UInt8](capacity=(16 + 3 * Q) * N)
    # per chain role: (block, i, j, head, p), idle chains -1
    var role = List[Tuple[Int, Int, Int, Bool, Int]](length=p.h2(), fill=(-1, 0, 0, False, 0))
    for x2 in range(chain_count(limbs, muls)):
        var x = x2 % (2 * B)
        var ij = _ij(limbs, x % B)
        role[x2] = (x // B, ij[0], ij[1], x // B == 0 and _is_head(limbs, ij[0], ij[1]), ij[0] + ij[1])
    var names: List[String] = ["lo", "bd", "s0", "s1", "s2", "bu", "cl", "hm", "rm", "ch", "m9", "m1", "mt", "zm", "ym", "um"]
    for name in names:
        for x2 in range(p.h2()):
            var rl = role[x2]
            var live = rl[0] >= 0
            var head = rl[3]
            var i = rl[1]
            var j = rl[2]
            var pp = rl[4]
            for x1 in range(ROWS):
                var b = False
                var product = live and i < limbs           # a and b are zero on the carry chains and the idle ones
                if name == "lo":
                    b = _weight_rows(0, LIMB, x1)
                elif name == "bd":
                    b = product and _weight_rows(0, LIMB, x1)
                elif name == "s0":
                    b = product and _weight_rows(0, PIECE, x1)
                elif name == "s1":
                    b = product and _weight_rows(PIECE, 2 * PIECE, x1)
                elif name == "s2":
                    b = product and _weight_rows(2 * PIECE, LIMB, x1)
                elif name == "bu":
                    b = _weight_rows(0, WIDTH, x1)
                elif name == "cl":
                    b = live and i < limbs and j >= 1 and _weight_rows(0, LIMB, x1)
                elif name == "hm":
                    b = live and i >= 1 and _weight_rows(0, Q, x1)
                elif name == "rm":
                    b = live and i >= 1 and _weight_rows(0, LIMB, x1)
                elif name == "ch" or name == "ym":
                    b = head and _weight_rows(0, LIMB, x1)
                elif name == "m9":
                    b = head and pp >= 1 and pp < limbs and _weight_rows(0, Q, x1)
                elif name == "m1":
                    b = head and pp >= limbs and _weight_rows(0, Q, x1)
                elif name == "mt":
                    b = head and pp == 2 * limbs - 1 and _weight_rows(0, Q, x1)
                elif name == "zm":
                    b = head and pp < limbs and _weight_rows(0, LIMB, x1)
                elif name == "um":
                    b = head and _weight_rows(0, WIDTH, x1)
                data.append(UInt8(1) if b else UInt8(0))
    for name in ["kc", "k4", "kt"]:
        for j in range(Q):
            for x2 in range(p.h2()):
                var rl = role[x2]
                for x1 in range(ROWS):
                    var b = False
                    if rl[3]:
                        var four = x1 == _row(2) and j == 2 % Q
                        if name == "kc":
                            b = (x1 == _row(BIAS) and j == BIAS % Q) or (rl[4] == 0 and four)
                        elif name == "k4":
                            b = four
                        else:
                            b = rl[4] == 2 * limbs - 1 and four
                    data.append(UInt8(1) if b else UInt8(0))
    for f in _factors(limbs, muls):
        data.extend(_columns(vals[f[2]][f[3]].bits(FOLDED), f[1] == SLOT_HA))
    return data^


struct RSA(Workload, Copyable, Movable):
    """s^(2^(muls - 1) + 1) mod n = m on `limbs` 256-bit limbs (`rsa_statement`)."""
    var limbs: Int
    var muls: Int
    var s: Big
    var n: Big
    var m: Big

    def __init__(out self, limbs: Int, muls: Int, s: Big, n: Big, m: Big):
        self.limbs = limbs
        self.muls = muls
        self.s = s.copy()
        self.n = n.copy()
        self.m = m.copy()

    def statement[p: Params](self) raises -> Statement:
        return rsa_statement(self.limbs, self.muls)

    def trace[p: Params](self, layout: Layout) raises -> List[UInt8]:
        return rsa_trace[p](layout, self.limbs, self.muls, self.s, self.n, self.m)

    def public_inputs[p: Params](self) raises -> List[UInt8]:
        return rsa_inputs(self.limbs, self.muls, self.s, self.n, self.m)

    @staticmethod
    def public_data[p: Params](layout: Layout, public_inputs: List[UInt8]) raises -> List[UInt8]:
        return rsa_public_data[p](layout, public_inputs)
