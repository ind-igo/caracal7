"""Arithmetic mod secp256k1's p on the builder: a circuit of chains, each a product (polynomial-mulmod 1 to
8), an addition or subtraction (its bit-level second column group), or the canonical check. No product is
ever committed, only bits.

The product: a chain of ROWS rows holds Q weight slots per row, row ROWS - 2 the lowest weights, position 0
the lowest in a row (so a Horner scan with scale zeta^Q evaluates at zeta); the last row holds no weight and
is never ingested. a is split into PIECES pieces of PIECE bits, each in its own columns at the bit's global
weight, its piece's rows selected by the public column `s{t}` (without that a prover could pile every bit
into one piece and reach a coefficient of 127, which is 0 in F_127); b is one column set; the coefficients of
C_t = A_t B (at most 88 each) are certified as
64 b6 + 32 b5 + v with b6 b5 = 0, bit m at the slot of the coefficient's weight plus m. Then the pile at slot w
(the 21 coefficient bits stored there) ripples into r: pile + carry in = r + 2 carry out, the carry out of a
slot stored at the slot, five bits. The identity zeta^6 R_A R_B = R_C at the chain end binds every
coefficient; the zero rows on the idle row's carry bits keep the carry into weight 0 at zero (without them
the idle row, which no accumulator reads, could carry a one in).

The fold (polynomial-mulmod 7): 2^256 = 2^32 + 977 mod p, so r = lo + 2^256 hi is congruent to
lo + sum_s (hi << s) over the seven shifts SHIFTS. Public selector columns know a row's weight: `lo` is 1 on
the rows of weights below 256, `cp` below 264; h = cp r@80 copies the high half of r (64 rows up, cyclic)
weight-aligned into the low rows, and the selector keeps h zero everywhere else. Then the fold pile at slot w
is lo r_w plus seven h reads at slots w - s (same-row or a row below, k1 in 0..8), rippled into o with a
3-bit carry; o is below 2^298. The second fold does the same from o into f, which is below 2^257 < 2 p and
congruent to a b mod p. Neither fold needs zero rows: the piles above weight 295 are zero, so the carry into
the idle row is zero and the idle row's pile is zero.

The addition chain: x + y = s + q p with q = q1 + 2 q2 in 0..3 chain-constant bits, a ripple with a signed
carry c0 + 2 c1 - 4 cn in [-4, 3] (the two sides are sums of bit vectors, so carries can go negative), and
every value bounded below 2^WIDTH by the selector `bd`. The bits of p are public columns `pb{j}`, 2 p the
same read one slot down. A subtraction x - y is the same chain with the output in the x slot; the canonical
check f < p is the chain with s the public constant p - 1, y the witness p - 1 - f, and q forced to zero
by the mask `cn`, a public column that is 1 on the canonical chains.

Wiring (polynomial-mulmod 8): a circuit is a list of chains (kind, a, b), each operand reference -1 a public
value or k >= 0 the output of an earlier chain k. The plain fingerprints of every chain value (a by H_A, one
accumulator beside the piece-weighted R_A; b, f, x, y, s) are wiring slots; an operand from chain k is an
edge to that chain's output slot, a public operand and every unconsumed output a public factor. Public inputs
carry the circuit (so the static public data can derive the per-chain columns and the factor list) then the
public values, VALUE bytes each, below 2^FOLDED."""

from caracal7.core.params import Params
from caracal7.relations.ir import FIX_E, CHAL_MUL
from caracal7.relations.statement import Statement, Layout, Term, BIT
from caracal7.workload import Workload

comptime ROWS = 144         # rows per chain: 572 weight slots, up to 525 live (C_2's top coefficient at weight 519 with b below 2^260, its bit 6 at slot 525)
comptime Q = 4              # weight slots per row
comptime PIECES = 3
comptime PIECE = 88         # bits per piece of a, 22 rows: coefficients of A_t B stay at most 88 < 95, the certificate's range; the last piece holds 84
comptime CBITS = 7          # coefficient bits: c = 64 b6 + 32 b5 + v, b6 b5 = 0
comptime CARRY = 5          # ripple carry bits: (21 + 31) / 2 < 32
comptime BITS = 256
comptime ZETA = 2           # gamma: the evaluation point
comptime RHO = 1            # delta: the piece weight
comptime SLOTS = Q * (ROWS - 1)
comptime FOLD = 3           # fold ripple carry bits: (8 + 7) / 2 < 8
comptime FOLDED = BITS + 1  # a folded result and every public value are below 2^257
comptime WIDTH = 260        # every chain value is below 2^WIDTH (the bound selector `bd`), a multiple of Q
comptime HI = ROWS - 2 - (BITS - 1) // Q    # first row of the low half: rows HI..ROWS-2 hold weights below 256
comptime UP = (ROWS - BITS // Q) % ROWS     # k1 that reads 256 weights up (64 rows), cyclic
comptime VALUE = FOLDED // 8 + 1            # bytes per public value
comptime ACARRY = 3                         # addition carry bits: c0 + 2 c1 - 4 cn
comptime MUL = 0            # chain kinds
comptime ADD = 1
comptime SUB = 2
comptime CANON = 3
comptime NONE_REF = 65535   # a public operand in the circuit bytes
comptime CHAIN_BYTES = 5    # kind u8, a u16, b u16


def _shifts() -> List[Int]:
    """2^256 = sum_s 2^s mod p: 977 = 2^9 + 2^8 + 2^7 + 2^6 + 2^4 + 1."""
    var v: List[Int] = [0, 4, 6, 7, 8, 9, 32]
    return v^


def _below(j: Int, s: Int) -> Tuple[Int, Int]:
    """(k1, position) of the slot s below position j of a row."""
    var u = Q * (ROWS // 2) + j - s
    return ((ROWS // 2 - u // Q + ROWS) % ROWS, u % Q)


def _row(w: Int) -> Int:
    return ROWS - 2 - w // Q


def _lo_row(x1: Int) -> Bool:
    return x1 >= HI and x1 <= ROWS - 2


def _bd_row(x1: Int) -> Bool:
    """The bound's rows: weights below WIDTH."""
    return x1 >= _row(WIDTH - 1) and x1 <= ROWS - 2


def _cp_row(x1: Int) -> Bool:
    """The copy's rows: weights below 2 WIDTH - 256, the high half of a product of two bounded values."""
    return x1 >= _row(2 * WIDTH - BITS - 1) and x1 <= ROWS - 2


def _c(t: Int, m: Int, j: Int) -> String:
    return "c" + String(t) + String(m) + String(j)


def _y(k: Int, j: Int) -> String:
    return "y" + String(k) + String(j)


def _piece(i: Int) -> Int:
    return min(i // PIECE, PIECES - 1)


def _piece_row(t: Int, x1: Int) -> Bool:
    """Row x1 holds weights of piece t (the last piece takes everything above, up to WIDTH)."""
    var lo = ROWS - 2 - (WIDTH - 1) // Q if t == PIECES - 1 else _row((t + 1) * PIECE - 1)
    return x1 >= lo and x1 <= _row(t * PIECE)


@fieldwise_init
struct Chain(Copyable, Movable, ImplicitlyCopyable):
    """One chain of a circuit: `kind` MUL (a b), ADD (a + b), SUB (a - b), CANON (a < p); `a`, `b` are -1 for
    a public value or the index of an earlier chain whose output they are."""
    var kind: Int
    var a: Int
    var b: Int


def single_chain() -> List[Chain]:
    var v: List[Chain] = [Chain(MUL, -1, -1)]
    return v^


def _in_slots(kind: Int) raises -> Tuple[Int, Int]:
    """The wiring slots of a chain's two operands: ha rb rf fx fy fs are slots 0 to 5."""
    if kind == MUL:
        return (0, 1)
    if kind == ADD:
        return (3, 4)
    if kind == SUB:
        return (5, 4)
    if kind == CANON:
        return (3, -1)
    raise Error("unknown chain kind")


def _out_slot(kind: Int) raises -> Int:
    if kind == MUL:
        return 2
    if kind == ADD:
        return 5
    if kind == SUB:
        return 3
    raise Error("a canonical check has no output")


def _slot_acc(slot: Int) -> String:
    var names: List[String] = ["ha", "rb", "rf", "fx", "fy", "fs"]
    return names[slot]


def mulmod_statement(zeros: Bool = True, circuit: List[Chain] = List[Chain]()) raises -> Statement:
    """`zeros = False` drops the zero rows: the unsound variant the test proves the idle-row carry against.
    `circuit` (default one product of public operands): see `Chain`."""
    var chains = circuit.copy() if len(circuit) > 0 else single_chain()
    for j in range(len(chains)):
        if chains[j].a >= j or chains[j].b >= j or (chains[j].kind == CANON and chains[j].b >= 0):
            raise Error("a chain operand comes from an earlier chain; the canonical check has one")
    var st = Statement()
    for t in range(PIECES):
        for j in range(Q):
            st.col("a" + String(t) + String(j), BIT)
    for j in range(Q):
        st.col("b" + String(j), BIT)
    for t in range(PIECES):
        for m in range(CBITS):
            for j in range(Q):
                st.col(_c(t, m, j), BIT)
    for j in range(Q):
        st.col("r" + String(j), BIT)
    for k in range(CARRY):
        for j in range(Q):
            st.col(_y(k, j), BIT)
    for name in ["h", "o", "g", "f"]:
        for j in range(Q):
            st.col(name + String(j), BIT)
    for name in ["z", "v"]:
        for k in range(FOLD):
            for j in range(Q):
                st.col(name + String(k) + String(j), BIT)
    for name in ["ax", "ay", "as"]:
        for j in range(Q):
            st.col(name + String(j), BIT)
    st.col("aq1", BIT)
    st.col("aq2", BIT)
    for k in range(ACARRY):
        for j in range(Q):
            st.col("ac" + String(k) + String(j), BIT)
    # ponytail: dense (h2, h1) blocks; the constant ones take m = h2 once the statement knows the grid
    for name in ["lo", "cp", "bd", "cn"]:
        st.pub(name, 1)
    for j in range(Q):
        st.pub("pb" + String(j), 1)
    for t in range(PIECES):
        st.pub("s" + String(t), 1)
    st.pin(circuit_bytes(chains))
    # rz[10 t + k] = rho^t zeta^k as element indices, -1 for 1
    var rz = List[Int](length=10 * PIECES, fill=-1)
    rz[1] = ZETA
    for k in range(2, 10):
        rz[k] = st.derived(CHAL_MUL, rz[k - 1], ZETA)
    rz[10] = RHO
    rz[20] = st.derived(CHAL_MUL, RHO, RHO)
    for t in range(1, PIECES):
        for k in range(1, 10):
            rz[10 * t + k] = st.derived(CHAL_MUL, rz[10 * t], rz[k])
    var ia = List[Term]()
    for t in range(PIECES):
        for j in range(Q):
            ia.append(Term(1, st.read("a" + String(t) + String(j)), chal=rz[10 * t + j]))
    st.horner("ra", ia, scale=rz[Q])
    var ih = List[Term]()
    for t in range(PIECES):
        for j in range(Q):
            ih.append(Term(1, st.read("a" + String(t) + String(j)), chal=rz[j]))
    st.horner("ha", ih, scale=rz[Q])
    var plain: List[String] = ["b", "f", "ax", "ay", "as"]
    var accs: List[String] = ["rb", "rf", "fx", "fy", "fs"]
    for i in range(len(plain)):
        var terms = List[Term]()
        for j in range(Q):
            terms.append(Term(1, st.read(plain[i] + String(j)), chal=rz[j]))
        st.horner(accs[i], terms, scale=rz[Q])
    var ic = List[Term]()
    for t in range(PIECES):
        for m in range(CBITS):
            for j in range(Q):
                ic.append(Term(1 << m, st.read(_c(t, m, j)), chal=rz[10 * t + j + 6 - m]))
    st.horner("rc", ic, scale=rz[Q])
    st.chain_end("mul", [Term(1, st.read("ra"), st.read("rb"), chal=rz[6]), Term(-1, st.read("rc"))])
    # ripple per position: pile + carry in - r - 2 carry out; position 0 takes the carry from the next row's position 3
    for j in range(Q):
        var terms = List[Term]()
        for t in range(PIECES):
            for m in range(CBITS):
                terms.append(Term(1, st.read(_c(t, m, j))))
        for k in range(CARRY):
            terms.append(Term(1 << k, st.read(_y(k, j - 1)) if j > 0 else st.read(_y(k, Q - 1), k1=1)))
        terms.append(Term(-1, st.read("r" + String(j))))
        for k in range(CARRY):
            terms.append(Term(-(2 << k), st.read(_y(k, j))))
        st.family("carry" + String(j), terms)
    # alias b6 b5 = 0 for the coefficient at each position: bit 6 sits six slots up, bit 5 five
    for t in range(PIECES):
        for j in range(Q):
            st.family("alias" + String(t) + String(j), [Term(1, st.read(_c(t, 6, (j + 6) % Q), k1=(ROWS - (j + 6) // Q) % ROWS),
                                                              st.read(_c(t, 5, (j + 5) % Q), k1=(ROWS - (j + 5) // Q) % ROWS))])
    for t in range(PIECES):
        for j in range(Q):
            st.family("piece" + String(t) + String(j), [Term(1, st.read("a" + String(t) + String(j))), Term(-1, st.read("s" + String(t)), st.read("a" + String(t) + String(j)))])
    if zeros:
        for k in range(CARRY):
            st.zero(_y(k, Q - 1), FIX_E)
        for k in range(ACARRY):
            st.zero("ac" + String(k) + String(Q - 1), FIX_E)
    _fold_families(st, "r", "h", "o", "z")
    _fold_families(st, "o", "g", "f", "v")
    _add_families(st)
    var slots = List[Int]()
    for s in range(6):
        slots.append(st.slot(_slot_acc(s)))
    var consumed = List[Bool](length=len(chains), fill=False)
    for j in range(len(chains)):
        var ins = _in_slots(chains[j].kind)
        for side in range(2):
            var src = chains[j].a if side == 0 else chains[j].b
            var slot = ins[0] if side == 0 else ins[1]
            if slot < 0:
                continue
            if src < 0:
                st.public_factor("p" + String(j) + _slot_acc(slot), _slot_acc(slot), slot, j)
            else:
                st.wire(slot, j, _out_slot(chains[src].kind), src)
                consumed[src] = True
        if chains[j].kind == CANON:
            st.public_factor("p" + String(j) + "fs", "fs", 5, j)
    for j in range(len(chains)):
        if not consumed[j] and chains[j].kind != CANON:
            st.public_factor("out" + String(j), _slot_acc(_out_slot(chains[j].kind)), _out_slot(chains[j].kind), j)
    return st^


def _fold_families(mut st: Statement, src: String, copy: String, dst: String, carry: String) raises:
    """copy = cp src@UP (the high half, weight-aligned in the low rows); then per position lo src + seven copy
    reads + carry in = out + 2 carry out. No zero rows: the pile is zero above weight 295 and on the idle
    row, so every carry above the live weights is forced to zero."""
    for j in range(Q):
        st.family(copy + "hi" + String(j), [Term(1, st.read(copy + String(j))), Term(-1, st.read("cp"), st.read(src + String(j), k1=UP))])
    for j in range(Q):
        var terms: List[Term] = [Term(1, st.read("lo"), st.read(src + String(j)))]
        for s in _shifts():
            var at = _below(j, s)
            terms.append(Term(1, st.read(copy + String(at[1]), k1=at[0])))
        for k in range(FOLD):
            terms.append(Term(1 << k, st.read(carry + String(k) + String(j - 1)) if j > 0 else st.read(carry + String(k) + String(Q - 1), k1=1)))
        terms.append(Term(-1, st.read(dst + String(j))))
        for k in range(FOLD):
            terms.append(Term(-(2 << k), st.read(carry + String(k) + String(j))))
        st.family(dst + "fold" + String(j), terms)


def _carry_terms(mut st: Statement, j: Int, sign: Int, k1: Int) -> List[Term]:
    """sign (c0 + 2 c1 - 4 cn) at position j of the row k1 rows down."""
    var v = List[Term]()
    v.append(Term(sign, st.read("ac0" + String(j), k1=k1)))
    v.append(Term(2 * sign, st.read("ac1" + String(j), k1=k1)))
    v.append(Term(-4 * sign, st.read("ac2" + String(j), k1=k1)))
    return v^


def _add_families(mut st: Statement) raises:
    """x + y + carry in = s + q1 p + q2 (2 p) + 2 carry out per position, the carry signed; q chain-constant;
    x, y, s zero above WIDTH by `bd`; q zero on the canonical chains by `cn`."""
    for j in range(Q):
        var terms: List[Term] = [Term(1, st.read("ax" + String(j))), Term(1, st.read("ay" + String(j))), Term(-1, st.read("as" + String(j)))]
        terms.append(Term(-1, st.read("aq1"), st.read("pb" + String(j))))
        terms.append(Term(-1, st.read("aq2"), st.read("pb" + String(j - 1)) if j > 0 else st.read("pb" + String(Q - 1), k1=1)))
        if j > 0:
            terms.extend(_carry_terms(st, j - 1, 1, 0))
        else:
            terms.extend(_carry_terms(st, Q - 1, 1, 1))
        terms.extend(_carry_terms(st, j, -2, 0))
        st.family("add" + String(j), terms)
    for name in ["ax", "ay", "as"]:
        for j in range(Q):
            st.family(name + "bd" + String(j), [Term(1, st.read(name + String(j))), Term(-1, st.read("bd"), st.read(name + String(j)))])
    for name in ["aq1", "aq2"]:
        st.family(name + "const", [Term(1, st.read(name)), Term(-1, st.read(name, k1=1))])
        st.family(name + "canon", [Term(1, st.read("cn"), st.read(name))])


# ---- host arithmetic on bit lists (bit 0 first) ----

def bits_of(bytes: List[UInt8], off: Int, n: Int) -> List[Int]:
    """n bits of the little-endian bytes at off, bit 0 first."""
    var v = List[Int](capacity=n)
    for i in range(n):
        v.append(Int(bytes[off + i // 8] >> UInt8(i % 8)) & 1)
    return v^


def bytes_of(bits: List[Int]) -> List[UInt8]:
    var v = List[UInt8](length=(len(bits) + 7) // 8, fill=0)
    for i in range(len(bits)):
        v[i // 8] |= UInt8(bits[i]) << UInt8(i % 8)
    return v^


def product_bits(a: List[Int], b: List[Int]) -> List[Int]:
    """The product of two bit vectors, len(a) + len(b) bits."""
    var r = List[Int](length=len(a) + len(b), fill=0)
    for i in range(len(a)):
        if a[i] == 0:
            continue
        var carry = 0
        for j in range(len(b)):
            var s = r[i + j] + b[j] + carry
            r[i + j] = s & 1
            carry = s >> 1
        var k = i + len(b)
        while carry > 0:
            var s = r[k] + carry
            r[k] = s & 1
            carry = s >> 1
            k += 1
    return r^


def ge_bits(a: List[Int], b: List[Int]) -> Bool:
    """a >= b, both n bits."""
    for i in range(len(a) - 1, -1, -1):
        if a[i] != b[i]:
            return a[i] > b[i]
    return True


def add_bits(a: List[Int], b: List[Int], n: Int) -> List[Int]:
    """a + b on n bits (the inputs any shorter length)."""
    var v = List[Int](length=n, fill=0)
    var carry = 0
    for i in range(n):
        var s = carry + (a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)
        v[i] = s & 1
        carry = s >> 1
    return v^


def sub_bits(a: List[Int], b: List[Int], n: Int) -> List[Int]:
    """a - b on n bits, for a >= b."""
    var v = List[Int](length=n, fill=0)
    var borrow = 0
    for i in range(n):
        var d = (a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0) - borrow
        v[i] = d & 1
        borrow = 1 if d < 0 else 0
    return v^


def p_bits(n: Int) -> List[Int]:
    """secp256k1's p = 2^256 - 2^32 - 977 on n bits."""
    var v = List[Int](length=n, fill=0)
    v[BITS] = 1
    var d = List[Int](length=n, fill=0)
    var c = (1 << 32) + 977
    for i in range(64):
        d[i] = (c >> i) & 1
    return sub_bits(v, d, n)


def _add_shifted(mut acc: List[Int], src: List[Int], lo: Int, hi: Int, shift: Int):
    """acc += sum_{i in [lo, hi)} src[i] 2^(i - lo + shift)."""
    var carry = 0
    for k in range(shift, len(acc)):
        var i = lo + k - shift
        var s = acc[k] + carry + (src[i] if i < hi else 0)
        acc[k] = s & 1
        carry = s >> 1


def fold_bits(bits: List[Int]) -> List[Int]:
    """lo + sum_s (hi << s) for the split of `bits` at BITS: congruent to the input mod p, FOLDED + 48 bits."""
    var acc = List[Int](length=FOLDED + 48, fill=0)
    _add_shifted(acc, bits, 0, min(BITS, len(bits)), 0)
    for s in _shifts():
        _add_shifted(acc, bits, BITS, len(bits), s)
    return acc^


def folded_bits(a: List[Int], b: List[Int]) raises -> List[Int]:
    """The FOLDED bits of a product chain's output for the operand bits."""
    var f = fold_bits(fold_bits(product_bits(a, b)))
    for i in range(FOLDED, len(f)):
        if f[i] != 0:
            raise Error("the folded result exceeds " + String(FOLDED) + " bits")
    f.resize(FOLDED, 0)
    return f^


def _columns(bits: List[Int], grouped: Bool) -> List[UInt8]:
    """Bit i at the slot of weight i, in the column set of its piece when `grouped`: (PIECES or 1) x Q columns of
    ROWS bytes, column-major, in the accumulators' ingest order."""
    var cols = List[UInt8](length=(PIECES if grouped else 1) * Q * ROWS, fill=0)
    for i in range(len(bits)):
        var g = _piece(i) if grouped else 0
        cols[(g * Q + i % Q) * ROWS + _row(i)] = UInt8(bits[i])
    return cols^


# ---- trace ----

@fieldwise_init
struct ChainValues(Copyable, Movable):
    """The values on one chain: a, b, f of a product; x, y, s and q of an addition (x + y = s + q p)."""
    var kind: Int
    var a: List[Int]
    var b: List[Int]
    var f: List[Int]
    var q: Int


def _trim(v: List[Int]) raises -> List[Int]:
    """A chain value on WIDTH bits."""
    var w = v.copy()
    for i in range(WIDTH, len(w)):
        if w[i] != 0:
            raise Error("chain values are below 2^" + String(WIDTH))
    w.resize(WIDTH, 0)
    return w^


def _values(kind: Int, a: List[Int], b: List[Int]) raises -> ChainValues:
    """The chain's values from its two inputs (`b` empty for the canonical check)."""
    var n = WIDTH + 2
    var pb = p_bits(n)
    if kind == MUL:
        return ChainValues(kind, a.copy(), b.copy(), folded_bits(a, b), 0)
    if kind == ADD:
        var s = add_bits(a, b, n)
        var q = 0
        for _ in range(3):
            if ge_bits(s, pb):
                s = sub_bits(s, pb, n)
                q += 1
        return ChainValues(kind, a.copy(), b.copy(), _trim(s), q)
    if kind == SUB:                    # x = s (the minuend), y = b, out = x - y + q p
        var x = a.copy()
        x.resize(n, 0)
        var q = 0
        var b2 = b.copy()
        b2.resize(n, 0)
        for _ in range(4):
            if ge_bits(x, b2):
                break
            x = add_bits(x, pb, n)
            q += 1
        return ChainValues(kind, _trim(sub_bits(x, b2, n)), b.copy(), a.copy(), q)
    var pm1 = sub_bits(pb, [1], n)     # CANON: x = a, s = p - 1, y = p - 1 - a
    var a2 = a.copy()
    a2.resize(n, 0)
    if not ge_bits(pm1, a2):
        raise Error("the value is not canonical")
    return ChainValues(kind, a.copy(), _trim(sub_bits(pm1, a2, n)), _trim(pm1), 0)


def circuit_values(inputs: List[List[UInt8]], circuit: List[Chain]) raises -> List[ChainValues]:
    """Every chain's values, public operands taken from `inputs` in circuit order."""
    var vals = List[ChainValues]()
    var next = 0
    for j in range(len(circuit)):
        var ops = List[List[Int]]()
        var ins = _in_slots(circuit[j].kind)
        for side in range(2):
            var src = circuit[j].a if side == 0 else circuit[j].b
            if (ins[0] if side == 0 else ins[1]) < 0:
                ops.append(List[Int]())
            elif src < 0:
                if next >= len(inputs) or len(inputs[next]) != VALUE or inputs[next][VALUE - 1] > 1:
                    raise Error("public values are " + String(VALUE) + " bytes below 2^" + String(FOLDED) + ", one per public reference")
                ops.append(bits_of(inputs[next], 0, FOLDED))
                next += 1
            elif src < j:
                var out = vals[src].f.copy() if circuit[src].kind != SUB else vals[src].a.copy()
                if circuit[src].kind == CANON:
                    raise Error("a canonical check has no output")
                ops.append(out^)
            else:
                raise Error("a chain operand comes from an earlier chain")
        vals.append(_values(circuit[j].kind, ops[0], ops[1]))
    if next != len(inputs):
        raise Error("more public values than the circuit references")
    return vals^


def _put(mut trace: List[UInt8], layout: Layout, N: Int, base: Int, name: String, bits: List[Int], grouped: Bool) raises:
    """A value's bits into the column set `name` (or a's pieces) of the chain at `base`."""
    var cols = _columns(bits, grouped)
    for g in range(PIECES if grouped else 1):
        for j in range(Q):
            for x1 in range(ROWS):
                trace[layout.col(name + (String(g) if grouped else String("")) + String(j)) * N + base + x1] = cols[(g * Q + j) * ROWS + x1]


def mulmod_trace[p: Params](layout: Layout, a: List[UInt8], b: List[UInt8], cheat: Int = -1) raises -> List[UInt8]:
    """One product chain from the 32-byte operands; see `circuit_trace`."""
    var vals = List[ChainValues]()
    vals.append(_values(MUL, bits_of(a, 0, BITS), bits_of(b, 0, BITS)))
    return circuit_trace[p](layout, vals, cheat)


def circuit_trace[p: Params](layout: Layout, vals: List[ChainValues], cheat: Int = -1) raises -> List[UInt8]:
    """Chain k from its values; the other chains idle. On chain `cheat` (if any) the idle row's slot 3 holds a
    pile of two and the product ripple starts from carry 1, so r = a b + 1 there satisfies every family and
    only the zero row catches it."""
    comptime h1 = p.h1()
    comptime N = p.N()
    if h1 != ROWS or cheat >= p.h2() or len(vals) > p.h2():
        raise Error("the instance needs " + String(ROWS) + " rows per chain and every chain on the grid")
    var trace = List[UInt8](length=layout.columns_w() * N, fill=0)
    for x2 in range(p.h2()):
        var live = x2 < len(vals)
        if not live and x2 != cheat:
            continue
        var base = x2 * h1
        var kind = vals[x2].kind if live else MUL
        if kind != MUL:
            _add_chain[p](layout, trace, x2, vals[x2])
            continue
        var ab = vals[x2].a.copy() if live else List[Int]()
        var bb = vals[x2].b.copy() if live else List[Int]()
        if len(ab) > WIDTH or len(bb) > WIDTH:
            raise Error("chain values are at most " + String(WIDTH) + " bits")
        _put(trace, layout, N, base, "a", ab, True)
        _put(trace, layout, N, base, "b", bb, False)
        for t in range(PIECES):
            for w in range(t * PIECE, t * PIECE + PIECE + len(bb) + Q):
                var c = 0
                for i in range(t * PIECE, min((t + 1) * PIECE, len(ab)) if t < PIECES - 1 else len(ab)):
                    if i <= w and w - i < len(bb):
                        c += ab[i] * bb[w - i]
                var b6 = 1 if c >= 64 else 0
                var b5 = 1 if c >= 32 and c < 64 else 0
                var v = c - 64 * b6 - 32 * b5
                for m in range(CBITS):
                    var bit = b6 if m == 6 else (b5 if m == 5 else (v >> m) & 1)
                    if bit == 1:
                        trace[layout.col(_c(t, m, (w + m) % Q)) * N + base + _row(w + m)] = 1
        var carry = 0
        if x2 == cheat:
            var off = base + h1 - 1
            trace[layout.col(_c(0, 0, Q - 1)) * N + off] = 1
            trace[layout.col(_c(0, 1, Q - 1)) * N + off] = 1
            trace[layout.col(_y(0, Q - 1)) * N + off] = 1
            carry = 1
        for w in range(SLOTS):
            var s = carry
            for t in range(PIECES):
                for m in range(CBITS):
                    s += Int(trace[layout.col(_c(t, m, w % Q)) * N + base + _row(w)])
            trace[layout.col("r" + String(w % Q)) * N + base + _row(w)] = UInt8(s & 1)
            carry = s >> 1
            for k in range(CARRY):
                trace[layout.col(_y(k, w % Q)) * N + base + _row(w)] = UInt8((carry >> k) & 1)
        _fold_chain[p](layout, trace, x2, "r", "h", "o", "z")
        _fold_chain[p](layout, trace, x2, "o", "g", "f", "v")
    return trace^


def _fold_chain[p: Params](layout: Layout, mut trace: List[UInt8], x2: Int, src: String, copy: String, dst: String, carry: String) raises:
    """The copy and the fold ripple of one chain, the families of `_fold_families` evaluated in row order."""
    comptime h1 = p.h1()
    comptime N = p.N()
    var base = x2 * h1
    for x1 in range(h1):
        if _cp_row(x1):
            for j in range(Q):
                trace[layout.col(copy + String(j)) * N + base + x1] = trace[layout.col(src + String(j)) * N + base + (x1 + UP) % h1]
    var cy = 0
    for w in range(SLOTS):
        var x1 = _row(w)
        var s = cy + (Int(trace[layout.col(src + String(w % Q)) * N + base + x1]) if _lo_row(x1) else 0)
        for sh in _shifts():
            var at = _below(w % Q, sh)
            s += Int(trace[layout.col(copy + String(at[1])) * N + base + (x1 + at[0]) % h1])
        trace[layout.col(dst + String(w % Q)) * N + base + x1] = UInt8(s & 1)
        cy = s >> 1
        for k in range(FOLD):
            trace[layout.col(carry + String(k) + String(w % Q)) * N + base + x1] = UInt8((cy >> k) & 1)


def _add_chain[p: Params](layout: Layout, mut trace: List[UInt8], x2: Int, v: ChainValues) raises:
    """x, y, s, q and the signed carries of `_add_families` in row order."""
    comptime h1 = p.h1()
    comptime N = p.N()
    var base = x2 * h1
    if len(v.a) > WIDTH or len(v.b) > WIDTH or len(v.f) > WIDTH:
        raise Error("chain values are at most " + String(WIDTH) + " bits")
    _put(trace, layout, N, base, "ax", v.a, False)
    _put(trace, layout, N, base, "ay", v.b, False)
    _put(trace, layout, N, base, "as", v.f, False)
    for x1 in range(h1):
        trace[layout.col("aq1") * N + base + x1] = UInt8(v.q & 1)
        trace[layout.col("aq2") * N + base + x1] = UInt8(v.q >> 1)
    var pb = p_bits(SLOTS)
    var c = 0
    for w in range(SLOTS):
        var l = c + (v.a[w] if w < len(v.a) else 0) + (v.b[w] if w < len(v.b) else 0)
        var r = (v.f[w] if w < len(v.f) else 0) + (v.q & 1) * pb[w] + (v.q >> 1) * (pb[w - 1] if w > 0 else 0)
        c = (l - r) // 2
        if (l - r) % 2 != 0 or c < -4 or c > 3:
            raise Error("addition carry out of range")
        var e = c + 4 if c < 0 else c
        trace[layout.col("ac0" + String(w % Q)) * N + base + _row(w)] = UInt8(e & 1)
        trace[layout.col("ac1" + String(w % Q)) * N + base + _row(w)] = UInt8(e >> 1)
        trace[layout.col("ac2" + String(w % Q)) * N + base + _row(w)] = UInt8(1 if c < 0 else 0)
    if c != 0:
        raise Error("addition does not close")


# ---- workload ----

def circuit_bytes(circuit: List[Chain]) -> List[UInt8]:
    """count u16, then per chain kind u8, a u16, b u16 (NONE_REF for a public operand)."""
    var v = List[UInt8]()
    v.append(UInt8(len(circuit) & 255))
    v.append(UInt8(len(circuit) >> 8))
    for ch in circuit:
        v.append(UInt8(ch.kind))
        _u16(v, ch.a)
        _u16(v, ch.b)
    return v^


def _u16(mut v: List[UInt8], r: Int):
    var e = NONE_REF if r < 0 else r
    v.append(UInt8(e & 255))
    v.append(UInt8(e >> 8))


def parse_circuit(bytes: List[UInt8]) raises -> Tuple[List[Chain], Int]:
    """The circuit at the head of the public inputs and the offset of the values after it."""
    if len(bytes) < 2:
        raise Error("public inputs start with the circuit")
    var n = Int(bytes[0]) | Int(bytes[1]) << 8
    if len(bytes) < 2 + n * CHAIN_BYTES:
        raise Error("public inputs start with the circuit")
    var v = List[Chain]()
    for j in range(n):
        var o = 2 + j * CHAIN_BYTES
        var a = Int(bytes[o + 1]) | Int(bytes[o + 2]) << 8
        var b = Int(bytes[o + 3]) | Int(bytes[o + 4]) << 8
        v.append(Chain(Int(bytes[o]), -1 if a == NONE_REF else a, -1 if b == NONE_REF else b))
    return (v^, 2 + n * CHAIN_BYTES)


@fieldwise_init
struct Mulmod(Workload, Copyable, Movable):
    """A circuit of chains (products, additions, subtractions, canonical checks mod p) on chains 0 .. len - 1
    of a ROWS x h2 grid (`mulmod_statement`). Public inputs: the circuit bytes, then per public operand and
    per unconsumed output, in statement order, the VALUE-byte little-endian value."""
    var inputs: List[List[UInt8]]
    var circuit: List[Chain]

    def statement(self) raises -> Statement:
        return mulmod_statement(circuit=self.circuit)

    def trace[p: Params](self, layout: Layout) raises -> List[UInt8]:
        return circuit_trace[p](layout, circuit_values(self.inputs, self.circuit))

    def public_inputs[p: Params](self) raises -> List[UInt8]:
        var vals = circuit_values(self.inputs, self.circuit)
        var v = circuit_bytes(self.circuit)
        var next = 0
        var consumed = List[Bool](length=len(self.circuit), fill=False)
        for j in range(len(self.circuit)):
            var ins = _in_slots(self.circuit[j].kind)
            for side in range(2):
                var src = self.circuit[j].a if side == 0 else self.circuit[j].b
                if (ins[0] if side == 0 else ins[1]) < 0:
                    continue
                if src < 0:
                    v.extend(self.inputs[next].copy())
                    next += 1
                else:
                    consumed[src] = True
        for j in range(len(self.circuit)):
            if not consumed[j] and self.circuit[j].kind != CANON:
                var out = vals[j].f.copy() if self.circuit[j].kind != SUB else vals[j].a.copy()
                out.resize(FOLDED, 0)
                v.extend(bytes_of(out))
        return v^

    @staticmethod
    def public_data[p: Params](layout: Layout, public_inputs: List[UInt8]) raises -> List[UInt8]:
        """The public blocks lo, cp, bd (the same on every chain), cn (1 on the canonical chains), pb{j} (the
        bits of p), s{t} (the rows of piece t), then every factor's ingest columns in statement order: an a operand in its pieces (12
        columns), any other value plain (4); the canonical check's constant p - 1."""
        comptime N = p.N()
        if p.h1() != ROWS:
            raise Error("the instance needs " + String(ROWS) + " rows per chain")
        var parsed = parse_circuit(public_inputs)
        var circuit = parsed[0].copy()
        var off = parsed[1]
        if len(circuit) > p.h2() or (len(public_inputs) - off) % VALUE != 0:
            raise Error("public inputs are the circuit then " + String(VALUE) + "-byte values")
        var data = List[UInt8](capacity=(8 + PIECES) * N + (len(public_inputs) - off) // VALUE * PIECES * Q * ROWS)
        for _ in range(p.h2()):
            for x1 in range(ROWS):
                data.append(UInt8(1) if _lo_row(x1) else UInt8(0))
        for _ in range(p.h2()):
            for x1 in range(ROWS):
                data.append(UInt8(1) if _cp_row(x1) else UInt8(0))
        for _ in range(p.h2()):
            for x1 in range(ROWS):
                data.append(UInt8(1) if _bd_row(x1) else UInt8(0))
        for x2 in range(p.h2()):
            for _ in range(ROWS):
                data.append(UInt8(1) if x2 < len(circuit) and circuit[x2].kind == CANON else UInt8(0))
        var pcols = _columns(p_bits(FOLDED), False)
        for j in range(Q):
            for _ in range(p.h2()):
                for x1 in range(ROWS):
                    data.append(pcols[j * ROWS + x1])
        for t in range(PIECES):
            for _ in range(p.h2()):
                for x1 in range(ROWS):
                    data.append(UInt8(1) if _piece_row(t, x1) else UInt8(0))
        var pm1 = sub_bits(p_bits(FOLDED), [1], FOLDED)
        var consumed = List[Bool](length=len(circuit), fill=False)
        for j in range(len(circuit)):
            var ins = _in_slots(circuit[j].kind)
            for side in range(2):
                var src = circuit[j].a if side == 0 else circuit[j].b
                var slot = ins[0] if side == 0 else ins[1]
                if slot < 0:
                    continue
                if src < 0:
                    data.extend(_value_columns(public_inputs, off, slot == 0))
                    off += VALUE
                elif src < j:
                    consumed[src] = True
                else:
                    raise Error("a chain operand comes from an earlier chain")
            if circuit[j].kind == CANON:
                data.extend(_columns(pm1, False))
        for j in range(len(circuit)):
            if not consumed[j] and circuit[j].kind != CANON:
                data.extend(_value_columns(public_inputs, off, False))
                off += VALUE
        if off != len(public_inputs):
            raise Error("more public values than the circuit references")
        return data^


def _value_columns(bytes: List[UInt8], off: Int, grouped: Bool) raises -> List[UInt8]:
    if off + VALUE > len(bytes) or bytes[off + VALUE - 1] > 1:       # the fingerprint reads FOLDED bits: the top seven bits of the last byte must be zero
        raise Error("a public value is missing or exceeds " + String(FOLDED) + " bits")
    return _columns(bits_of(bytes, off, FOLDED), grouped)


def value_bytes_of(v: List[UInt8]) raises -> List[UInt8]:
    """A 32-byte value as a VALUE-byte operand."""
    if len(v) != BITS // 8:
        raise Error("a 32-byte value")
    var w = v.copy()
    w.append(0)
    return w^
