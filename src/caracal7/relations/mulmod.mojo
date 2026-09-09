"""One 256-bit product a b = r as a polynomial identity (polynomial-mulmod 1 to 6): no product is ever committed,
only bits. A chain of ROWS rows holds Q weight slots per row, row ROWS - 2 the lowest weights, position 0 the
lowest in a row (so a Horner scan with scale zeta^Q evaluates at zeta); the last row holds no weight and is
never ingested. a is split into PIECES pieces of PIECE bits, each in its own columns at the bit's global
weight; b is one column set; the coefficients of C_t = A_t B (at most 86 each) are certified as
64 b6 + 32 b5 + v with b6 b5 = 0, bit m at the slot of the coefficient's weight plus m. Then the pile at slot w
(the 21 coefficient bits stored there) ripples into r: pile + carry in = r + 2 carry out, the carry out of a
slot stored at the slot, five bits. The identity zeta^6 R_A R_B = R_C at the chain end binds every
coefficient; the zero rows on the idle row's carry bits keep the carry into weight 0 at zero (without them
the idle row, which no accumulator reads, could carry a one in).

The fold (polynomial-mulmod 7): 2^256 = 2^32 + 977 mod p, so r = lo + 2^256 hi is congruent to
lo + sum_s (hi << s) over the seven shifts SHIFTS. The public selector column `lo` is 1 on the rows of weights
below 256; h = lo r@80 copies the high half of r (64 rows up, cyclic) weight-aligned into the low rows, and the
selector keeps h zero everywhere else. Then the fold pile at slot w is lo r_w plus seven h reads at slots w - s
(same-row or a row below, k1 in 0..8), rippled into o with a 3-bit carry; o is below 2^290. The second fold
does the same from o into f, which is below 2^257 < 2 p and congruent to a b mod p. Neither fold needs zero
rows: the piles above weight 290 are zero, so the carry into the idle row is zero and the idle row's pile is at
most one bit.

Wiring (polynomial-mulmod 8): a circuit is a list of chains, each (a, b) an operand reference: -1 a public
value, k >= 0 the output f of an earlier chain k. Every chain's f (plain fingerprint R_f), a (plain
fingerprint H_A, one accumulator beside the piece-weighted R_A) and b (R_B) are wiring slots; an operand
from chain k is an edge to that chain's R_f, a public operand and every unconsumed output a public factor.
Operands and results are FOLDED-bit values (33 bytes, below 2^257): the copy selector `cp` covers weights
below 260 so the fold takes a product below 2^516."""

from caracal7.core.params import Params
from caracal7.relations.ir import FIX_E, CHAL_MUL
from caracal7.relations.statement import Statement, Layout, Term, BIT
from caracal7.workload import Workload

comptime ROWS = 144         # rows per chain: 572 weight slots, 517 live (C_2's top coefficient is at weight 510, its bit 6 at slot 516)
comptime Q = 4              # weight slots per row
comptime PIECES = 3
comptime PIECE = 86         # bits per piece of a: coefficients of A_t B stay at most 86 < 95, the certificate's range
comptime CBITS = 7          # coefficient bits: c = 64 b6 + 32 b5 + v, b6 b5 = 0
comptime CARRY = 5          # ripple carry bits: (21 + 31) / 2 < 32
comptime BITS = 256
comptime ZETA = 2           # gamma: the evaluation point
comptime RHO = 1            # delta: the piece weight
comptime SLOTS = Q * (ROWS - 1)
comptime FOLD = 3           # fold ripple carry bits: (8 + 7) / 2 < 8
comptime FOLDED = BITS + 1  # the folded result is below 2^257
comptime HI = ROWS - 2 - (BITS - 1) // Q    # first row of the low half: rows HI..ROWS-2 hold weights below 256
comptime UP = (ROWS - BITS // Q) % ROWS     # k1 that reads 256 weights up (64 rows), cyclic
comptime VALUE = FOLDED // 8 + 1            # bytes per operand or result
comptime TAG_A = 0                          # public input tags: an a operand, a b operand, an output
comptime TAG_B = 1
comptime TAG_OUT = 2


def _shifts() -> List[Int]:
    """2^256 = sum_s 2^s mod p: 977 = 2^9 + 2^8 + 2^7 + 2^6 + 2^4 + 1."""
    var v: List[Int] = [0, 4, 6, 7, 8, 9, 32]
    return v^


def _below(j: Int, s: Int) -> Tuple[Int, Int]:
    """(k1, position) of the slot s below position j of a row."""
    var u = Q * (ROWS // 2) + j - s
    return ((ROWS // 2 - u // Q + ROWS) % ROWS, u % Q)


def _lo_row(x1: Int) -> Bool:
    return x1 >= HI and x1 <= ROWS - 2


def _cp_row(x1: Int) -> Bool:
    """The copy's rows: weights below 260, one row more than the low half, for a high half of up to 260 bits."""
    return x1 >= HI - 1 and x1 <= ROWS - 2


def single_chain() -> List[Tuple[Int, Int]]:
    var v: List[Tuple[Int, Int]] = [(-1, -1)]
    return v^


def _c(t: Int, m: Int, j: Int) -> String:
    return "c" + String(t) + String(m) + String(j)


def _y(k: Int, j: Int) -> String:
    return "y" + String(k) + String(j)


def _row(w: Int) -> Int:
    return ROWS - 2 - w // Q


def mulmod_statement(zeros: Bool = True, circuit: List[Tuple[Int, Int]] = List[Tuple[Int, Int]]()) raises -> Statement:
    """`zeros = False` drops the zero rows: the unsound variant the test proves the idle-row carry against.
    `circuit` (default one chain of public operands): per chain (a, b), -1 a public operand or the chain whose
    output it is."""
    var chains = circuit.copy() if len(circuit) > 0 else single_chain()
    for j in range(len(chains)):
        if chains[j][0] >= j or chains[j][1] >= j:
            raise Error("a mulmod operand comes from an earlier chain")
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
    st.pub("lo", 1)     # ponytail: dense (h2, h1) blocks; m = h2 once the statement knows the grid
    st.pub("cp", 1)
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
    var ib = List[Term]()
    for j in range(Q):
        ib.append(Term(1, st.read("b" + String(j)), chal=rz[j]))
    st.horner("rb", ib, scale=rz[Q])
    var ic = List[Term]()
    for t in range(PIECES):
        for m in range(CBITS):
            for j in range(Q):
                ic.append(Term(1 << m, st.read(_c(t, m, j)), chal=rz[10 * t + j + 6 - m]))
    st.horner("rc", ic, scale=rz[Q])
    var iff = List[Term]()
    for j in range(Q):
        iff.append(Term(1, st.read("f" + String(j)), chal=rz[j]))
    st.horner("rf", iff, scale=rz[Q])
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
    if zeros:
        for k in range(CARRY):
            st.zero(_y(k, Q - 1), FIX_E)
    _fold_families(st, "r", "h", "o", "z")
    _fold_families(st, "o", "g", "f", "v")
    var sa = st.slot("ha")
    var sb = st.slot("rb")
    var sf = st.slot("rf")
    var consumed = List[Bool](length=len(chains), fill=False)
    for j in range(len(chains)):
        for side in range(2):
            var src = chains[j][0] if side == 0 else chains[j][1]
            var slot = sa if side == 0 else sb
            var acc = String("ha") if side == 0 else String("rb")
            if src < 0:
                st.public_factor("p" + acc + String(j), acc, slot, j)
            else:
                st.wire(slot, j, sf, src)
                consumed[src] = True
    for j in range(len(chains)):
        if not consumed[j]:
            st.public_factor("prf" + String(j), "rf", sf, j)
    return st^


def _fold_families(mut st: Statement, src: String, copy: String, dst: String, carry: String) raises:
    """copy = cp src@UP (the high half, up to 260 bits, weight-aligned in the low rows); then per position
    lo src + seven copy reads + carry in = out + 2 carry out. No zero rows: the pile is zero above weight 291
    and on the idle row, so every carry above the live weights is forced to zero."""
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


def _add_shifted(mut acc: List[Int], src: List[Int], lo: Int, hi: Int, shift: Int):
    """acc += sum_{i in [lo, hi)} src[i] 2^(i - lo + shift)."""
    var carry = 0
    for k in range(shift, len(acc)):
        var i = lo + k - shift
        var s = acc[k] + carry + (src[i] if i < hi else 0)
        acc[k] = s & 1
        carry = s >> 1


def fold_bits(bits: List[Int]) -> List[Int]:
    """lo + sum_s (hi << s) for the split of `bits` at BITS: congruent to the input mod p, FOLDED + 40 bits."""
    var acc = List[Int](length=FOLDED + 40, fill=0)
    _add_shifted(acc, bits, 0, min(BITS, len(bits)), 0)
    for s in _shifts():
        _add_shifted(acc, bits, BITS, len(bits), s)
    return acc^


def folded_bits(a: List[Int], b: List[Int]) raises -> List[Int]:
    """The FOLDED bits of the chain's output for the operand bits."""
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
        var g = i // PIECE if grouped else 0
        cols[(g * Q + i % Q) * ROWS + _row(i)] = UInt8(bits[i])
    return cols^


def mulmod_trace[p: Params](layout: Layout, a: List[UInt8], b: List[UInt8], cheat: Int = -1) raises -> List[UInt8]:
    """One chain from the 32-byte operands; see `circuit_trace`."""
    var ab = List[List[Int]]()
    var bb = List[List[Int]]()
    ab.append(bits_of(a, 0, BITS))
    bb.append(bits_of(b, 0, BITS))
    return circuit_trace[p](layout, ab, bb, cheat)


def circuit_trace[p: Params](layout: Layout, abits: List[List[Int]], bbits: List[List[Int]], cheat: Int = -1) raises -> List[UInt8]:
    """Chain k from its operand bits (at most FOLDED each); the other chains idle. On chain `cheat` (if any)
    the idle row's slot 3 holds a pile of two and the ripple starts from carry 1, so r = a b + 1 there
    satisfies every family and only the zero row catches it."""
    comptime h1 = p.h1()
    comptime N = p.N()
    if h1 != ROWS or cheat >= p.h2() or len(abits) > p.h2() or len(abits) != len(bbits):
        raise Error("the mulmod instance needs " + String(ROWS) + " rows per chain and every chain on the grid")
    var trace = List[UInt8](length=layout.columns_w() * N, fill=0)
    for x2 in range(p.h2()):
        var live = x2 < len(abits)
        if not live and x2 != cheat:
            continue
        var base = x2 * h1
        var ab = abits[x2].copy() if live else List[Int]()
        var bb = bbits[x2].copy() if live else List[Int]()
        if len(ab) > FOLDED or len(bb) > FOLDED:
            raise Error("mulmod operands are at most " + String(FOLDED) + " bits")
        var ac = _columns(ab, True)
        for t in range(PIECES):
            for j in range(Q):
                for x1 in range(h1):
                    trace[layout.col("a" + String(t) + String(j)) * N + base + x1] = ac[(t * Q + j) * ROWS + x1]
        var bc = _columns(bb, False)
        for j in range(Q):
            for x1 in range(h1):
                trace[layout.col("b" + String(j)) * N + base + x1] = bc[j * ROWS + x1]
        for t in range(PIECES):
            for w in range(t * PIECE, t * PIECE + PIECE + len(bb)):
                var c = 0
                for i in range(t * PIECE, min((t + 1) * PIECE, len(ab))):
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


def circuit_values(inputs: List[List[UInt8]], circuit: List[Tuple[Int, Int]]) raises -> Tuple[List[List[Int]], List[List[Int]], List[List[Int]]]:
    """(a bits, b bits, f bits) per chain: public operands taken from `inputs` in circuit order."""
    var abits = List[List[Int]]()
    var bbits = List[List[Int]]()
    var fbits = List[List[Int]]()
    var next = 0
    for j in range(len(circuit)):
        for side in range(2):
            var src = circuit[j][0] if side == 0 else circuit[j][1]
            var v: List[Int]
            if src < 0:
                if next >= len(inputs) or len(inputs[next]) != VALUE or inputs[next][VALUE - 1] > 1:
                    raise Error("mulmod public operands are " + String(VALUE) + " bytes below 2^" + String(FOLDED) + ", one per public reference")
                v = bits_of(inputs[next], 0, FOLDED)
                next += 1
            elif src < j:
                v = fbits[src].copy()
            else:
                raise Error("a mulmod operand comes from an earlier chain")
            if side == 0:
                abits.append(v^)
            else:
                bbits.append(v^)
        fbits.append(folded_bits(abits[j], bbits[j]))
    if next != len(inputs):
        raise Error("mulmod has more public operands than the circuit references")
    return (abits^, bbits^, fbits^)


@fieldwise_init
struct Mulmod(Workload, Copyable, Movable):
    """A circuit of products mod p on chains 0 .. len(circuit) - 1 of a ROWS x h2 grid (`mulmod_statement`).
    Public inputs: per public operand and per unconsumed output, in statement order, a tag byte (TAG_A, TAG_B,
    TAG_OUT) then the VALUE-byte little-endian value."""
    var inputs: List[List[UInt8]]
    var circuit: List[Tuple[Int, Int]]

    def statement(self) raises -> Statement:
        return mulmod_statement(circuit=self.circuit)

    def trace[p: Params](self, layout: Layout) raises -> List[UInt8]:
        var vals = circuit_values(self.inputs, self.circuit)
        return circuit_trace[p](layout, vals[0], vals[1])

    def public_inputs[p: Params](self) raises -> List[UInt8]:
        var vals = circuit_values(self.inputs, self.circuit)
        var v = List[UInt8]()
        var next = 0
        var consumed = List[Bool](length=len(self.circuit), fill=False)
        for j in range(len(self.circuit)):
            for side in range(2):
                var src = self.circuit[j][0] if side == 0 else self.circuit[j][1]
                if src < 0:
                    v.append(UInt8(TAG_A) if side == 0 else UInt8(TAG_B))
                    v.extend(self.inputs[next].copy())
                    next += 1
                else:
                    consumed[src] = True
        for j in range(len(self.circuit)):
            if not consumed[j]:
                v.append(UInt8(TAG_OUT))
                v.extend(bytes_of(vals[2][j]))
        return v^

    @staticmethod
    def public_data[p: Params](layout: Layout, public_inputs: List[UInt8]) raises -> List[UInt8]:
        """The two selector blocks (the same on every chain), then every tagged value's ingest columns: an a
        operand in its pieces (12 columns), a b operand or an output plain (4)."""
        if len(public_inputs) % (VALUE + 1) != 0 or p.h1() != ROWS:
            raise Error("mulmod public inputs are tagged " + String(VALUE) + "-byte values on " + String(ROWS) + " rows per chain")
        var data = List[UInt8](capacity=2 * p.N() + len(public_inputs) // (VALUE + 1) * PIECES * Q * ROWS)
        for _ in range(p.h2()):
            for x1 in range(ROWS):
                data.append(UInt8(1) if _lo_row(x1) else UInt8(0))
        for _ in range(p.h2()):
            for x1 in range(ROWS):
                data.append(UInt8(1) if _cp_row(x1) else UInt8(0))
        for i in range(len(public_inputs) // (VALUE + 1)):
            var off = i * (VALUE + 1)
            if public_inputs[off] > TAG_OUT or public_inputs[off + VALUE] > 1:       # the fingerprint reads FOLDED bits: the top seven bits of the last byte must be zero
                raise Error("mulmod value tag or value exceeds " + String(FOLDED) + " bits")
            data.extend(_columns(bits_of(public_inputs, off + 1, FOLDED), Int(public_inputs[off]) == TAG_A))
        return data^


def value_bytes_of(v: List[UInt8]) raises -> List[UInt8]:
    """A 32-byte value as a VALUE-byte operand."""
    if len(v) != BITS // 8:
        raise Error("a 32-byte value")
    var w = v.copy()
    w.append(0)
    return w^
