"""One 256-bit product a b = r as a polynomial identity (polynomial-mulmod 1 to 6): no product is ever committed,
only bits. A chain of ROWS rows holds Q weight slots per row, row ROWS - 2 the lowest weights, position 0 the
lowest in a row (so a Horner scan with scale zeta^Q evaluates at zeta); the last row holds no weight and is
never ingested. a is split into PIECES pieces of PIECE bits, each in its own columns at the bit's global
weight; b is one column set; the coefficients of C_t = A_t B (at most 86 each) are certified as
64 b6 + 32 b5 + v with b6 b5 = 0, bit m at the slot of the coefficient's weight plus m. Then the pile at slot w
(the 21 coefficient bits stored there) ripples into r: pile + carry in = r + 2 carry out, the carry out of a
slot stored at the slot, five bits. The identity zeta^6 R_A R_B = R_C at the chain end binds every
coefficient; the zero rows on the idle row's carry bits keep the carry into weight 0 at zero (without them
the idle row, which no accumulator reads, could carry a one in). a, b, r are public factors on the
fingerprints R_A, R_B, R_r."""

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


def _c(t: Int, m: Int, j: Int) -> String:
    return "c" + String(t) + String(m) + String(j)


def _y(k: Int, j: Int) -> String:
    return "y" + String(k) + String(j)


def _row(w: Int) -> Int:
    return ROWS - 2 - w // Q


def mulmod_statement(zeros: Bool = True) raises -> Statement:
    """`zeros = False` drops the zero rows: the unsound variant the test proves the idle-row carry against."""
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
    var ir = List[Term]()
    for j in range(Q):
        ir.append(Term(1, st.read("r" + String(j)), chal=rz[j]))
    st.horner("rr", ir, scale=rz[Q])
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
    var sa = st.slot("ra")
    var sb = st.slot("rb")
    var sr = st.slot("rr")
    st.public_factor("pa", "ra", sa, 0)
    st.public_factor("pb", "rb", sb, 0)
    st.public_factor("pr", "rr", sr, 0)
    return st^


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
    """The 2 BITS-bit product of two bit vectors."""
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


def _columns(bits: List[Int], grouped: Bool) -> List[UInt8]:
    """Bit i at the slot of weight i, in the column set of its piece when `grouped`: (PIECES or 1) x Q columns of
    ROWS bytes, column-major, in the accumulators' ingest order."""
    var cols = List[UInt8](length=(PIECES if grouped else 1) * Q * ROWS, fill=0)
    for i in range(len(bits)):
        var g = i // PIECE if grouped else 0
        cols[(g * Q + i % Q) * ROWS + _row(i)] = UInt8(bits[i])
    return cols^


def mulmod_trace[p: Params](layout: Layout, a: List[UInt8], b: List[UInt8], cheat: Int = -1) raises -> List[UInt8]:
    """Chain 0 from the 32-byte operands; the other chains idle. On chain `cheat` (if any) the idle row's slot 3
    holds a pile of two and the ripple starts from carry 1, so r = a b + 1 there satisfies every family and
    only the zero row catches it."""
    comptime h1 = p.h1()
    comptime N = p.N()
    if h1 != ROWS or cheat >= p.h2():
        raise Error("the mulmod instance needs " + String(ROWS) + " rows per chain and a cheat chain on the grid")
    if len(a) != BITS // 8 or len(b) != BITS // 8:
        raise Error("mulmod operands are 32 bytes")
    var trace = List[UInt8](length=layout.columns_w() * N, fill=0)
    var ab = bits_of(a, 0, BITS)
    var bb = bits_of(b, 0, BITS)
    var ac = _columns(ab, True)
    for t in range(PIECES):
        for j in range(Q):
            for x1 in range(h1):
                trace[layout.col("a" + String(t) + String(j)) * N + x1] = ac[(t * Q + j) * ROWS + x1]
    var bc = _columns(bb, False)
    for j in range(Q):
        for x1 in range(h1):
            trace[layout.col("b" + String(j)) * N + x1] = bc[j * ROWS + x1]
    for t in range(PIECES):
        for w in range(t * PIECE, t * PIECE + PIECE + BITS - 1):
            var c = 0
            for i in range(t * PIECE, min((t + 1) * PIECE, BITS)):
                if i <= w and w - i < BITS:
                    c += ab[i] * bb[w - i]
            var b6 = 1 if c >= 64 else 0
            var b5 = 1 if c >= 32 and c < 64 else 0
            var v = c - 64 * b6 - 32 * b5
            for m in range(CBITS):
                var bit = b6 if m == 6 else (b5 if m == 5 else (v >> m) & 1)
                if bit == 1:
                    trace[layout.col(_c(t, m, (w + m) % Q)) * N + _row(w + m)] = 1
    if cheat >= 0:
        var off = cheat * h1 + h1 - 1
        trace[layout.col(_c(0, 0, Q - 1)) * N + off] = 1
        trace[layout.col(_c(0, 1, Q - 1)) * N + off] = 1
        trace[layout.col(_y(0, Q - 1)) * N + off] = 1
    for x2 in range(p.h2()):
        var carry = 1 if x2 == cheat else 0
        if x2 != 0 and x2 != cheat:
            continue
        for w in range(SLOTS):
            var s = carry
            for t in range(PIECES):
                for m in range(CBITS):
                    s += Int(trace[layout.col(_c(t, m, w % Q)) * N + x2 * h1 + _row(w)])
            trace[layout.col("r" + String(w % Q)) * N + x2 * h1 + _row(w)] = UInt8(s & 1)
            carry = s >> 1
            for k in range(CARRY):
                trace[layout.col(_y(k, w % Q)) * N + x2 * h1 + _row(w)] = UInt8((carry >> k) & 1)
    return trace^


@fieldwise_init
struct Mulmod(Workload, Copyable, Movable):
    """a b = r on chain 0 of a ROWS x h2 grid. Public inputs: a, b (32 bytes each, little-endian), r (64 bytes)."""
    var a: List[UInt8]
    var b: List[UInt8]

    def statement(self) raises -> Statement:
        return mulmod_statement()

    def trace[p: Params](self, layout: Layout) raises -> List[UInt8]:
        return mulmod_trace[p](layout, self.a, self.b)

    def public_inputs[p: Params](self) raises -> List[UInt8]:
        var v = self.a.copy()
        v.extend(self.b.copy())
        v.extend(bytes_of(product_bits(bits_of(self.a, 0, BITS), bits_of(self.b, 0, BITS))))
        return v^

    @staticmethod
    def public_data[p: Params](layout: Layout, public_inputs: List[UInt8]) raises -> List[UInt8]:
        """The three factors' ingest columns: a in its pieces, b, r."""
        if len(public_inputs) != BITS // 2 or p.h1() != ROWS:       # 32 + 32 + 64 bytes
            raise Error("mulmod public inputs are a, b, r on " + String(ROWS) + " rows per chain")
        var data = _columns(bits_of(public_inputs, 0, BITS), True)
        data.extend(_columns(bits_of(public_inputs, BITS // 8, BITS), False))
        data.extend(_columns(bits_of(public_inputs, BITS // 4, 2 * BITS), False))
        return data^
