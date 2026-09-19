"""RSA verify on the product chain: s^e mod n = m for e = 2^(muls - 1) + 1 (65537 for muls = 17), n, s and m
of `limbs` 256-bit limbs. Every modmul a b = q n + r is an integer identity over limb products, checked
with running sums along chains and no additions in the wiring.

A modmul is two blocks of chains, AB with the products a_i b_j and QN with q_i n_j (q the quotient, a hint
the prover supplies; n public). Each block is a run of chains whose roles `_cells` lists (one `_Cell` per
chain: its product, the reads of its sum lane and the compare lane's carry, as axis-2 strides); every read
is forward on axis 2, under a public mask per stride, so a family is one term per stride the geometry uses.
The product r = a_i b_j < 2^512 of a chain splits into lo (weights below 256) and hi; the sum lane value t
of chain (i, j) is

    t = cont lo(t of (i + 1, j - 1)) + hi(t of (i - 1, j)) + hi(r of (i - 1, j)) + w lo(r of (i, j))

Along the anti-diagonal i + j = p the lo halves of t carry the running sum of position p, each chain's hi
(below 8, since t < 2^259) is passed to position p + 1 on the chain (i + 1, j), and the head of the
diagonal (the chain of least i) holds limb p of the block's total in lo(t). Limb 2 limbs - 1 is a tail
chain without a product that holds the hi of the last diagonal (`bu` bounds a tail's t below 2^256, so no
hi is lost there: the triangle's tail is A^2 / 2^(256 (2 limbs - 1)) minus the lower limbs, below 2^256;
a rectangle's interior tail (limbs, j) is the hi of a column sum of at most limbs products plus a carry,
which descends below 2^256 the same way). The ripple is an exact integer identity with signed 4-bit carries and a zero row on the
idle row's carry.

Two geometries. The rectangle (QN, and AB of the last modmul, a b with b = s): (limbs + 1) x limbs chains,
chain (i, j) at y = (limbs + 1)(limbs - 1 - j) + (limbs - i) with the tails at i = limbs, so the reads are
at the strides limbs (the diagonal) and 1 (the chain below). The triangle (AB of a squaring, a = b: every
modmul but the last): the products a_i a_j for i <= j only, w = 2 off the diagonal (and the hi of an
off-diagonal r counts twice too); one tail chain, then the diagonals p = 2 limbs - 2 down to 0, each with
i ascending from its head; the square (i, i) passes its hi to (i, i + 1), the last chain of the next
diagonal. limbs (limbs + 1) / 2 + 1 chains instead of limbs (limbs + 1): 37 for 72 at 2048 bits.

The compare lane on the AB heads: u_p = lo(t_ab) + K_p + hi(u_(p-1)) - lo(t_qn) - r_p with K_0 = 2^258 + 4
and K_p = 2^258 after, lo(t_qn) wired from the QN head (the cy accumulator ingests cy on the AB heads and
lo(t) under `qh` on the QN heads: one slot for both ends), r_p a hint (slot cz, zero above limb
limbs - 1 by `zm`) and the carry read from the previous head (the `mp{k}` masks). The families force
lo(u_p) = 4 on every head and hi(u) = 4 on the last one, which telescopes to
sum_p (lo(t_ab) - lo(t_qn) - r_p) 2^(256 p) = 0, so a b = q n + r as integers. u stays below 2^259 for any
witness (the bias exceeds the two subtracted limbs), so it is a bit vector. r is bound below 2^(256 limbs),
not below n: the statement proves s^e = m (mod n) for the public m, which a verifier supplies canonical.

Wiring: modmul 0 squares s (public factors on ha and rb); modmul m squares r of m - 1 (the cz hints of
its heads), the last one multiplies r by s. q_i's occurrences are wired to their first; n_j is a public
factor on every QN chain; the last modmul's r_p are the public factors m_p. Slots: ha, rb, cy, cz (a slot
costs h2 of the 16128 endpoints F2* holds, factors included). Public inputs: limbs, muls, then s, n, m as
limbs x 32 bytes little endian (pinned head: limbs and muls)."""

from std.memory import unsafe_memcpy
from std.builtin.sort import sort
from max.algorithm import parallelize
from core.params import Params
from relations.ir import FIX_E, PubTerm, pack_terms
from relations.statement import Statement, Layout, Term, BIT
from workloads.bigint import Big
from workloads.mulmod import ROWS, Q, PIECES, PIECE, SLOTS, BITS, UP, VALUE, FOLDED, WIDTH, MUL, OpValues, product_columns, product_certificate, plain_fingerprint, _row, _cols, _put, _columns, _mul_chain
from workload import Workload

comptime LIMB = BITS
comptime TBITS = 259        # a sum lane value: below 2^259 (two doubled halves, a continuation, two hi parts)
comptime SCARRY = 4         # signed ripple carry bits: c0 + 2 c1 + 4 c2 - 8 c3
comptime BIAS = 258         # K = 2^BIAS (+ 4 on head 0): above the two subtracted limbs
comptime SLOT_HA = 0
comptime SLOT_RB = 1
comptime SLOT_CY = 2
comptime SLOT_CZ = 3         # four slots; the SOD adds one and the factors take one coset (2016 chains: eight cosets of F2*)
comptime SRC_S = 0          # public factor sources
comptime SRC_N = 1
comptime SRC_M = 2


@fieldwise_init
struct _Cell(Copyable, Movable):
    """One chain's role in its modmul: the strides are forward axis-2 reads (0: none)."""
    var m: Int
    var block: Int      # 0 AB, 1 QN
    var i: Int
    var j: Int
    var product: Bool   # a_i b_j (a tail has none)
    var lo: Int         # the weight of lo(r) in the sum lane: 2 off a triangle's diagonal, 0 on a tail
    var cont: Int       # lo(t) of the next chain on the diagonal
    var hi: Int         # hi(t) and hi(r) of the chain below, (i - 1, j)
    var rw: Int         # the weight of that hi(r): 2 on a triangle
    var sq: Int         # hi(t) and hi(r) of a square (i, i), into (i, i + 1) or the tail
    var head: Int       # the limb this chain heads, -1 none
    var prev: Int       # the previous head (the compare carry); AB heads past 0


def _rect(limbs: Int) -> List[Tuple[Int, Int]]:
    """(i, j) per chain of a rectangle block, i = limbs the tails."""
    var v = List[Tuple[Int, Int]](length=limbs * (limbs + 1), fill=(0, 0))
    for j in range(limbs):
        for i in range(limbs + 1):
            v[(limbs + 1) * (limbs - 1 - j) + (limbs - i)] = (i, j)
    return v^


def _triangle(limbs: Int) -> List[Tuple[Int, Int]]:
    """(i, j) per chain of a triangle block: the tail, then the diagonals from the top, i ascending."""
    var v: List[Tuple[Int, Int]] = [(limbs, limbs - 1)]
    for p in range(2 * limbs - 2, -1, -1):
        for i in range(max(0, p - limbs + 1), p // 2 + 1):
            v.append((i, p - i))
    return v^


def _stride(at: Dict[Int, Int], limbs: Int, y: Int, i: Int, j: Int) raises -> Int:
    """The forward stride from chain y to chain (i, j)."""
    var k = at[i * (limbs + 1) + j] - y
    if k <= 0:
        raise Error("a sum lane read is forward")
    return k


def _block_cells(m: Int, block: Int, limbs: Int, square: Bool) raises -> List[_Cell]:
    var ij = _triangle(limbs) if square else _rect(limbs)
    var at = Dict[Int, Int]()                 # (i, j) -> y
    for y in range(len(ij)):
        at[ij[y][0] * (limbs + 1) + ij[y][1]] = y

    var v = List[_Cell]()
    var heads = List[Int](length=2 * limbs, fill=-1)
    for y in range(len(ij)):
        var i = ij[y][0]
        var j = ij[y][1]
        var p = i + j
        var tail = i == limbs
        var lo = 0 if tail else (2 if square and i < j else 1)
        var cont = 0
        var hi = 0
        var sq = 0
        if square:
            if not tail and i < p // 2:
                cont = _stride(at, limbs, y, i + 1, j - 1)
            if not tail and i >= 1:
                hi = _stride(at, limbs, y, i - 1, j)
            if tail or (p % 2 == 1 and i == p // 2):
                sq = _stride(at, limbs, y, limbs - 1 if tail else i, limbs - 1 if tail else i)
        else:
            if not tail and j >= 1:
                cont = _stride(at, limbs, y, i + 1, j - 1)
            if i >= 1:
                hi = _stride(at, limbs, y, i - 1, j)
        var head = -1
        if (square and (tail or i == max(0, p - limbs + 1))) or (not square and (i == 0 or j == limbs - 1)):
            head = 2 * limbs - 1 if tail else p
            heads[head] = y
        v.append(_Cell(m, block, i, j, not tail, lo, cont, hi, 2 if square else 1, sq, head, 0))
    for y in range(len(v)):
        var h = v[y].head
        if block == 0 and h >= 1:
            v[y].prev = heads[h - 1] - y
            if v[y].prev <= 0:
                raise Error("the compare carry reads forward")
    return v^


def _cells(limbs: Int, muls: Int) raises -> List[_Cell]:
    """Every chain of the verify in order: per modmul the AB block (a triangle but for the last modmul), then QN."""
    var v = List[_Cell]()
    for m in range(muls):
        v.extend(_block_cells(m, 0, limbs, m < muls - 1))
        v.extend(_block_cells(m, 1, limbs, False))
    return v^


def chain_count(limbs: Int, muls: Int) raises -> Int:
    return (muls - 1) * (limbs * (limbs + 1) // 2 + 1) + (muls + 1) * limbs * (limbs + 1)


def _heads(cells: List[_Cell], m: Int, block: Int) -> List[Int]:
    """The chain (from the base) heading each limb of block `block` of modmul m."""
    var limbs = 0
    for x in range(len(cells)):
        if cells[x].m == m and cells[x].block == block:
            limbs = max(limbs, cells[x].head + 1)
    var v = List[Int](length=limbs, fill=-1)
    for x in range(len(cells)):
        if cells[x].m == m and cells[x].block == block and cells[x].head >= 0:
            v[cells[x].head] = x
    return v^


def m_chain(limbs: Int, muls: Int, p: Int) raises -> Int:
    """The chain (from the base) whose cz accumulator holds limb p of m: the last modmul's AB head p."""
    return _heads(_cells(limbs, muls), muls - 1, 0)[p]


def _add_unique(mut l: List[Int], s: Int):
    if s == 0:
        return
    for t in l:
        if t == s:
            return
    l.append(s)


def _strides(cells: List[_Cell]) -> Tuple[List[Int], List[Int], List[Int], List[Int]]:
    """The distinct strides in use: (cont, hi, sq, prev), each ascending."""
    var cont = List[Int]()
    var hi = List[Int]()
    var sq = List[Int]()
    var prev = List[Int]()
    for c in cells:
        _add_unique(cont, c.cont)
        _add_unique(hi, c.hi)
        _add_unique(sq, c.sq)
        _add_unique(prev, c.prev)
    sort(cont)
    sort(hi)
    sort(sq)
    sort(prev)
    return (cont^, hi^, sq^, prev^)


def _num(name: String) -> Int:
    """The stride a mask name carries after its two-letter prefix."""
    var v = 0
    var b = name.as_bytes()
    for i in range(2, len(b)):
        v = 10 * v + Int(b[i]) - 48
    return v


def _mask_names(cells: List[_Cell]) -> List[String]:
    """The public columns in declaration order; the stride masks carry the stride in the name."""
    var st = _strides(cells)
    var v: List[String] = ["lo", "bd", "s0", "s1", "s2", "bu"]
    for k in st[0]:
        v.append("cl" + String(k))
    for k in st[1]:
        v.append("hm" + String(k))
        v.append("rm" + String(k))
    for k in st[2]:
        v.append("sq" + String(k))
    v.append("ch")
    for k in st[3]:
        v.append("mp" + String(k))
    v.extend(["mt", "zm", "ym", "um", "qh"])
    return v^


def _mask_range(c: _Cell, limbs: Int, name: String) -> Tuple[Int, Int, Int]:
    """Public column `name` on the chain with role c: (value, lo, hi), the value on the rows of weights in
    [lo, hi) and 0 elsewhere."""
    var head = c.block == 0 and c.head >= 0
    var v = 0
    var lo = 0
    var hi = LIMB
    if name == "lo":
        v = c.lo
    elif name == "bd":
        v = 1 if c.product else 0
    elif name == "s0":
        v = 1 if c.product else 0
        hi = PIECE
    elif name == "s1":
        v = 1 if c.product else 0
        lo = PIECE
        hi = 2 * PIECE
    elif name == "s2":
        v = 1 if c.product else 0
        lo = 2 * PIECE
    elif name == "bu":
        v = 1
        hi = WIDTH if c.product else LIMB          # a tail holds a limb: no hi
    elif name.startswith("cl"):
        v = 1 if c.cont == _num(name) else 0
    elif name.startswith("hm"):
        v = 1 if c.hi == _num(name) else 0
        hi = Q
    elif name.startswith("rm"):
        v = c.rw if c.hi == _num(name) else 0
    elif name.startswith("sq"):
        v = 1 if c.sq == _num(name) else 0
    elif name == "ch" or name == "ym":
        v = 1 if head else 0
    elif name.startswith("mp"):
        v = 1 if head and c.prev == _num(name) else 0
        hi = Q
    elif name == "mt":
        v = 1 if head and c.head == 2 * limbs - 1 else 0
        hi = Q
    elif name == "zm":
        v = 1 if head and c.head < limbs else 0
    elif name == "um":
        v = 1 if head else 0
        hi = WIDTH
    elif name == "qh":
        v = 1 if c.block == 1 and c.head >= 0 else 0
    return (v, lo, hi)


def _weight_rows(lo: Int, hi: Int, x1: Int) -> Bool:
    """Row x1 holds weights in [lo, hi) (the idle row none)."""
    if x1 > ROWS - 2:
        return False
    var w = (ROWS - 2 - x1) * Q
    return w >= lo and w < hi


def _factors(limbs: Int, muls: Int, m_wired: Bool = False, s_wired: Bool = False, n_wired: Bool = False) raises -> List[Tuple[Int, Int, Int, Int]]:
    """(chain, slot, source, limb) of every public factor in statement order; `m_wired` leaves limb 0 of m to
    a wire the caller adds (`m_chain`), `s_wired` leaves s a witness (its occurrences wired to each other),
    `n_wired` leaves n's limbs to wires the caller adds on the rb slot (`n_chains`)."""
    var cells = _cells(limbs, muls)
    var v = List[Tuple[Int, Int, Int, Int]]()
    for x in range(len(cells)):
        ref c = cells[x]
        if not c.product:
            continue
        if c.block == 0 and c.m == 0 and not s_wired:
            v.append((x, SLOT_HA, SRC_S, c.i))
        if c.block == 0 and (c.m == 0 or c.m == muls - 1) and not s_wired:
            v.append((x, SLOT_RB, SRC_S, c.j))
        if c.block == 1 and not n_wired:
            v.append((x, SLOT_RB, SRC_N, c.j))
    var heads = _heads(cells, muls - 1, 0)
    for p in range(1 if m_wired else 0, limbs):
        v.append((heads[p], SLOT_CZ, SRC_M, p))
    return v^


def n_chains(limbs: Int, muls: Int, j: Int) raises -> List[Int]:
    """The chains (from the base) whose rb slot holds limb j of n: every product of the QN blocks."""
    var cells = _cells(limbs, muls)
    var v = List[Int]()
    for x in range(len(cells)):
        if cells[x].block == 1 and cells[x].product and cells[x].j == j:
            v.append(x)
    return v^


def _wires(limbs: Int, muls: Int, s_wired: Bool = False) raises -> List[Tuple[Int, Int, Int, Int]]:
    """(slot, chain, slot, chain) of every wire; `s_wired` joins the occurrences of each limb of s (the ha
    and rb slots of the first modmul, the rb slot of the last) instead of the public factors."""
    var cells = _cells(limbs, muls)
    var v = List[Tuple[Int, Int, Int, Int]]()
    if s_wired:
        for i in range(limbs):
            var ends = List[Tuple[Int, Int]]()
            for x in range(len(cells)):
                ref c = cells[x]
                if c.block != 0 or not c.product:
                    continue
                if c.m == 0 and c.i == i:
                    ends.append((SLOT_HA, x))
                if (c.m == 0 or c.m == muls - 1) and c.j == i:
                    ends.append((SLOT_RB, x))
            for k in range(1, len(ends)):
                v.append((ends[k - 1][0], ends[k - 1][1], ends[k][0], ends[k][1]))
    var first = Dict[Int, Int]()              # q_i's first chain per modmul
    var prev = List[Int]()
    for m in range(muls):
        var ab = _heads(cells, m, 0)
        var qn = _heads(cells, m, 1)
        for x in range(len(cells)):
            ref c = cells[x]
            if c.m != m or not c.product:
                continue
            if c.block == 0 and m >= 1:
                v.append((SLOT_HA, x, SLOT_CZ, prev[c.i]))
                if m < muls - 1:
                    v.append((SLOT_RB, x, SLOT_CZ, prev[c.j]))
            if c.block == 1:
                if c.i in first:
                    v.append((SLOT_HA, x, SLOT_HA, first[c.i]))
                else:
                    first[c.i] = x
        first.clear()
        for p in range(2 * limbs):
            v.append((SLOT_CY, ab[p], SLOT_CY, qn[p]))
        prev = ab.copy()
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
    var st = Statement()
    _ = rsa_build(st, limbs, muls)
    return st^


def rsa_build(mut st: Statement, limbs: Int, muls: Int, base: Int = 0, m_wired: Bool = False, s_wired: Bool = False,
              n_wired: Bool = False) raises -> List[Int]:
    """The RSA columns, families and wiring on `st`, the chains from `base` (ungrouped: the columns are zero
    off them, as on the idle chains). `m_wired` leaves limb 0 of m to a wire the caller adds to the cz slot
    on chain base + `m_chain`(0); `s_wired` makes s a witness; `n_wired` leaves limb j of n to wires the
    caller adds to the rb slot on the chains base + `n_chains`(j) (the plain fingerprint of the limb: a
    SHA-256 group's 32-byte window, sha256g). Returns the slots ha, rb, cy, cz. Pins limbs and muls."""
    if limbs < 1 or limbs > 255 or muls < 1 or muls > 255:
        raise Error("limbs and muls are bytes, at least 1")
    product_columns(st)
    for name in ["t", "u", "cy", "cz"]:
        for j in range(Q):
            st.col(name + String(j), BIT)
    for name in ["sc", "uc"]:
        for k in range(SCARRY):
            for j in range(Q):
                st.col(name + String(k) + String(j), BIT)
    var cells = _cells(limbs, muls)
    var strides = _strides(cells)
    for name in _mask_names(cells):
        st.pub(name, 1)
    for name in ["kc", "k4", "kt"]:
        for j in range(Q):
            st.pub(name + String(j), 1)
    st.pin([UInt8(limbs), UInt8(muls)])
    var rz = product_certificate(st)
    var cy = List[Term]()
    for j in range(Q):
        cy.append(Term(1, st.read("cy" + String(j)), chal=rz[j]))                        # the AB heads (ym bounds cy to them)
        cy.append(Term(1, st.read("t" + String(j)), st.read("qh"), chal=rz[j]))          # lo(t) on the QN heads
    st.horner("cy", cy, scale=rz[Q])
    plain_fingerprint(st, "cz", "cz", rz)
    var sum_terms = List[List[Term]]()
    for j in range(Q):
        var t = List[Term]()
        for k in strides[0]:
            t.append(Term(1, st.read("t" + String(j), k2=k), st.read("cl" + String(k))))
        for k in strides[1]:
            t.append(Term(1, st.read("t" + String(j), k1=UP, k2=k), st.read("hm" + String(k))))
            t.append(Term(1, st.read("r" + String(j), k1=UP, k2=k), st.read("rm" + String(k))))
        for k in strides[2]:
            t.append(Term(1, st.read("t" + String(j), k1=UP, k2=k), st.read("sq" + String(k))))
            t.append(Term(1, st.read("r" + String(j), k1=UP, k2=k), st.read("sq" + String(k))))
        t.append(Term(1, st.read("r" + String(j)), st.read("lo")))
        sum_terms.append(t^)
    _ripple(st, "sum", "t", "sc", sum_terms)
    var cmp_terms = List[List[Term]]()
    for j in range(Q):
        var t: List[Term] = [Term(1, st.read("t" + String(j)), st.read("ch")),
                             Term(1, st.read("kc" + String(j)))]
        for k in strides[3]:
            t.append(Term(1, st.read("u" + String(j), k1=UP, k2=k), st.read("mp" + String(k))))
        t.append(Term(-1, st.read("cy" + String(j))))
        t.append(Term(-1, st.read("cz" + String(j))))
        cmp_terms.append(t^)
    _ripple(st, "cmp", "u", "uc", cmp_terms)
    for j in range(Q):
        st.family("ulo" + String(j), [Term(1, st.read("u" + String(j)), st.read("ch")), Term(-1, st.read("k4" + String(j)))])
        st.family("uhi" + String(j), [Term(1, st.read("u" + String(j), k1=UP), st.read("mt")), Term(-1, st.read("kt" + String(j)))])
    _bound(st, "t", "bu")
    _bound(st, "u", "um")
    _bound(st, "cy", "ym")
    _bound(st, "cz", "zm")
    var names: List[String] = ["ha", "rb", "cy", "cz"]
    var slots = List[Int]()
    for name in names:
        slots.append(st.slot(name))
    var fs = _factors(limbs, muls, m_wired, s_wired, n_wired)
    for i in range(len(fs)):
        st.public_factor("f" + String(i), names[fs[i][1]], slots[fs[i][1]], base + fs[i][0])
    for w in _wires(limbs, muls, s_wired):
        st.wire(slots[w[0]], base + w[1], slots[w[2]], base + w[3])
    return slots^


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


def _modmul_chains(cells: List[_Cell], m: Int, limbs: Int, a: List[Big], b: List[Big], q: List[Big], n: List[Big], r: List[Big]) raises -> _Chains:
    """The chain values of modmul m: `cells` of that modmul, from the first (sources are forward, so the
    chains are filled from the last)."""
    var x0 = 0
    while cells[x0].m != m:
        x0 += 1
    var count = 0
    while x0 + count < len(cells) and cells[x0 + count].m == m:
        count += 1
    var v = _Chains(List[Big](length=count, fill=Big()), List[Big](length=count, fill=Big()), List[Big](length=count, fill=Big()),
                    List[Big](length=count, fill=Big()), List[Big](length=count, fill=Big()), List[Big](length=count, fill=Big()),
                    List[Big](length=count, fill=Big()))
    for x in range(count - 1, -1, -1):
        ref c = cells[x0 + x]
        if c.product:
            v.a[x] = a[c.i].copy() if c.block == 0 else q[c.i].copy()
            v.b[x] = b[c.j].copy() if c.block == 0 else n[c.j].copy()
            v.r[x] = v.a[x] * v.b[x]
        var t = _lo(v.r[x]) * Big(c.lo)
        if c.cont > 0:
            t = t + _lo(v.t[x + c.cont])
        if c.hi > 0:
            t = t + _hi(v.t[x + c.hi]) + _hi(v.r[x + c.hi]) * Big(c.rw)
        if c.sq > 0:
            t = t + _hi(v.t[x + c.sq]) + _hi(v.r[x + c.sq])
        if t.bit_length() > (TBITS if c.product else LIMB):
            raise Error("sum lane value exceeds its width")
        v.t[x] = t^
    var ab = _heads(cells, m, 0)
    var qn = _heads(cells, m, 1)
    var carry = Big(4)
    for p in range(2 * limbs):
        var x = ab[p] - x0
        v.cy[x] = _lo(v.t[qn[p] - x0])
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


def rsa_trace[p: Params](layout: Layout, limbs: Int, muls: Int, s: Big, n: Big, m: Big, base: Int = 0) raises -> List[UInt8]:
    """The whole grid's W columns, the RSA chains from `base`; the other columns and chains stay zero."""
    comptime h1 = p.h1()
    comptime N = p.N()
    if h1 != ROWS or base + chain_count(limbs, muls) > p.h2():
        raise Error("the instance needs " + String(ROWS) + " rows per chain and " + String(chain_count(limbs, muls)) + " chains")
    if s >= n or m >= n or n.bit_length() > LIMB * limbs:
        raise Error("s and m below n, n of at most " + String(LIMB * limbs) + " bits")
    var cells = _cells(limbs, muls)
    var per = List[_Chains]()
    var a = s.copy()
    for k in range(muls):
        var b = s.copy() if k == 0 or k == muls - 1 else a.copy()
        var qr = (a * b).divmod(n)
        per.append(_modmul_chains(cells, k, limbs, limbs_of(a, limbs), limbs_of(b, limbs), limbs_of(qr[0], limbs), limbs_of(n, limbs), limbs_of(qr[1], limbs)))
        a = qr[1].copy()
    if a != m:
        raise Error("the signature does not verify")
    var first = List[Int](length=muls, fill=len(cells))    # the first chain of each modmul
    for x in range(len(cells) - 1, -1, -1):
        first[cells[x].m] = x
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
        if x2 < base or x2 >= base + len(cells):
            return
        ref c = cells[x2 - base]
        var x = x2 - base - first[c.m]
        var chain = List[UInt8](length=columns * h1, fill=0)
        try:
            ref v = per[c.m]
            if c.product:
                _mul_chain[p](chain, False, OpValues(MUL, v.a[x].copy(), v.b[x].copy(), Big(), Big(), 0), True, ca, cb, cr, cc, cy)
            _put(chain, ct, v.t[x].bits(TBITS), False)
            var pile = List[Int](length=SLOTS, fill=0)
            _add_bits(pile, v.r[x], c.lo, 0, LIMB)
            if c.cont > 0:
                _add_bits(pile, v.t[x + c.cont], 1, 0, LIMB)
            if c.hi > 0:
                _add_bits(pile, v.t[x + c.hi], 1, LIMB, Q)
                _add_bits(pile, v.r[x + c.hi], c.rw, LIMB, LIMB)
            if c.sq > 0:
                _add_bits(pile, v.t[x + c.sq], 1, LIMB, Q)
                _add_bits(pile, v.r[x + c.sq], 1, LIMB, LIMB)
            _lane_carries[p](chain, csc, pile, v.t[x].bits(TBITS))
            if c.block == 0 and c.head >= 0:
                var pp = c.head
                _put(chain, cu, v.u[x].bits(BIAS + 1), False)
                _put(chain, ccy, v.cy[x].bits(LIMB), False)
                _put(chain, ccz, v.cz[x].bits(LIMB), False)
                var cp = List[Int](length=SLOTS, fill=0)
                _add_bits(cp, v.t[x], 1, 0, LIMB)
                cp[BIAS] += 1
                if pp == 0:
                    cp[2] += 1
                else:
                    _add_bits(cp, v.u[x + c.prev], 1, LIMB, Q)
                _add_bits(cp, v.cy[x], -1, 0, LIMB)
                _add_bits(cp, v.cz[x], -1, 0, LIMB)
                _lane_carries[p](chain, cuc, cp, v.u[x].bits(BIAS + 1))
        except e:
            failed[x2] = 1
            errors[x2] = String(e)
            return
        for col in range(columns):
            unsafe_memcpy(dest=trace.unsafe_ptr().unsafe_offset(col * N + x2 * h1), src=chain.unsafe_ptr().unsafe_offset(col * h1), count=h1)

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


def rsa_public_data[p: Params](layout: Layout, inputs: List[UInt8], base: Int = 0, m_wired: Bool = False, s_wired: Bool = False,
                               n_wired: Bool = False) raises -> List[UInt8]:
    """The public columns in declaration order, each in term form (`pack_terms`: one term per distinct
    (value, weight range) of `_mask_range` over the live chains), then the factors' ingest columns
    (`rsa_build`'s base and wiring flags; a wired value's bytes in `inputs` are ignored)."""
    if len(inputs) < 2:
        raise Error("public inputs start with limbs and muls")
    var limbs = Int(inputs[0])
    var muls = Int(inputs[1])
    var lb = LIMB // 8 * limbs
    if p.h1() != ROWS or len(inputs) != 2 + 3 * lb or base + chain_count(limbs, muls) > p.h2():
        raise Error("public inputs are limbs, muls, then s, n, m of limbs x 32 bytes; the grid holds every chain")
    var vals = List[List[Big]]()
    for k in range(3):
        var bytes = List[UInt8]()
        for i in range(lb):
            bytes.append(inputs[2 + k * lb + i])
        vals.append(limbs_of(Big.from_bytes(bytes), limbs))
    var cells = _cells(limbs, muls)
    var names = _mask_names(cells)
    var data = List[UInt8]()
    for name in names:
        var keys = Dict[Int, Int]()
        var terms = List[PubTerm]()
        for x2 in range(len(cells)):
            var r = _mask_range(cells[x2], limbs, name)
            if r[0] == 0:
                continue
            var key = (r[0] * 1024 + r[1]) * 1024 + r[2]
            if key not in keys:
                keys[key] = len(terms)
                var row = List[UInt8](length=ROWS, fill=0)
                for x1 in range(ROWS):
                    if _weight_rows(r[1], r[2], x1):
                        row[x1] = UInt8(r[0])
                terms.append(PubTerm(row^, List[Int]()))
            terms[keys[key]].chains.append(base + x2)
        data.extend(pack_terms(terms, ROWS))
    for name in ["kc", "k4", "kt"]:
        for j in range(Q):
            var keys = Dict[Int, Int]()
            var terms = List[PubTerm]()
            for x2 in range(len(cells)):
                if cells[x2].block != 0 or cells[x2].head < 0:
                    continue
                var pp = cells[x2].head
                var four = j == 2 % Q
                var bias = name == "kc" and j == BIAS % Q
                if name == "kc":
                    four = four and pp == 0
                elif name == "kt":
                    four = four and pp == 2 * limbs - 1
                var key = (2 if bias else 0) + (1 if four else 0)
                if key == 0:
                    continue
                if key not in keys:
                    keys[key] = len(terms)
                    var row = List[UInt8](length=ROWS, fill=0)
                    if bias:
                        row[_row(BIAS)] = 1
                    if four:
                        row[_row(2)] = 1
                    terms.append(PubTerm(row^, List[Int]()))
                terms[keys[key]].chains.append(base + x2)
            data.extend(pack_terms(terms, ROWS))
    for f in _factors(limbs, muls, m_wired, s_wired, n_wired):
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
