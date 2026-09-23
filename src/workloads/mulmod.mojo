"""Arithmetic mod secp256k1's p on the builder: a circuit of ops, each a product (polynomial-mulmod 1 to
8) on a chain's MUL lane or a signed three-operand addition on one of its LANES add lanes. No product is
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

The P-256 pass (CURVE_P256): P-256's prime has no small fold, so a product below 2^520 is reduced in one
pass by FIPS 186's word identity: product word i = 8..16 is congruent to a signed pattern on the words 0..7
(`_p256_words`), and output word k reads product word k + d under the public selector `w{d}` that holds
the coefficient (16 offsets d, coefficients in [-4, 5], zero above weight 255 and on the idle row). The
Solinas sum S is in (-8 p, 8 p), so f = S - q p with a chain-constant q in [-8, 7] (four bits encoded
like the add carry, p's bits from the constant block `fp`: `pb` is n on a mod-n chain), rippled with the
add lane's signed carry and zero rows on the idle row. The honest f is canonical; any satisfying f is
congruent to a b mod p and every consumer bounds it. Against secp256k1's two folds: 16 fewer bit
columns, 18 more public columns, 11 more opening points (the word offsets).

The add lane: x + sy y + sz z = s + q m with sy, sz in {-1, 0, 1} per-chain public columns, the modulus m
(p, or n on a chain whose public columns `pb{j}` hold n) and q = b0 + 2 b1 + 4 b2 - 2 b3 in [-2, 7]
chain-constant bits, q m as the modulus bits read k slots down for 2^k m. The ripple has a signed carry
c0 + 2 c1 + 4 c2 - 8 c3 in [-8, 7] (the sides are sums of bit vectors, so carries go negative); every value
is bounded below 2^WIDTH by the selector `bd`, and the family is then an exact integer identity for any
witness (its residual is far below 127), so congruence mod m holds against any prover; the honest carries
stay in [-6, 1] for canonical inputs. Two public masks per lane finish the kinds: `sm` forces s to zero (the
op checks x + sy y + sz z = 0 mod m, an EQ) and `qz` forces q to zero (the op is an exact integer identity:
the canonical check x + y = p - 1 with y a free witness, the guard x - y = 1).

Wiring (polynomial-mulmod 8): a circuit is a list of ops, each operand reference PUB a public value, FREE a
witness the host solves for, NIL absent, or k >= 0 the output of op k. The plain fingerprints of every lane
value (a by H_A beside the piece-weighted R_A; b, f; each lane's x, y, z, s) are wiring slots; an operand
from op k is an edge to that op's output slot, a public operand and every unconsumed output a public
factor. MUL ops take chains in order; add ops take lanes in order, a chain per modulus. Public inputs carry
the circuit (so the static public data can derive the per-chain columns and the factor list; the statement
pins them) then the public values, VALUE bytes each, below 2^FOLDED."""

from std.memory import unsafe_memcpy
from max.algorithm import parallelize
from core.params import Params
from relations.ir import FIX_E, CHAL_MUL
from relations.statement import Statement, Layout, Term, BIT
from workloads.bigint import Big
from workload import Workload

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
comptime LANES = 2                          # add lanes per chain
comptime QBITS = 4                          # q = b0 + 2 b1 + 4 b2 - 2 b3
comptime ACARRY = 4                         # add carry bits: c0 + 2 c1 + 4 c2 - 8 c3
comptime MUL = 0            # op kinds
comptime ADD = 1
comptime PUB = -1           # operand references: a public value, a free witness, no operand; s: a normal output is OUT
comptime FREE = -2
comptime NIL = -3
comptime HINT = -4          # hint(h) = HINT - h: witness h, wired between its occurrences, bounded, no factor
comptime OUT = -2
comptime MOD_P = 0          # the modulus of an add op
comptime MOD_N = 1
comptime OP_BYTES = 13      # kind u8, x u16, y u16, z u16, s u16, sy u8, sz u8, qz u8, mod u8
comptime CURVE_K1 = 0       # the curve of a statement: the moduli and the product reduction
comptime CURVE_P256 = 1
comptime WORD = 32          # the P-256 reduction works on 32-bit words of the product: WORD // Q rows each
comptime WORDS = 8          # words of a reduced value


def _shifts() -> List[Int]:
    """2^256 = sum_s 2^s mod p: 977 = 2^9 + 2^8 + 2^7 + 2^6 + 2^4 + 1."""
    var v: List[Int] = [0, 4, 6, 7, 8, 9, 32]
    return v^


def _p256_words() -> List[List[Int]]:
    """Product word i = 8..16 mod P-256 as signed coefficients on the words 0..7 (FIPS 186 D.2.3):
    2^256 = 2^224 - 2^192 - 2^96 + 1, and each next word shifts the last and folds its overflow."""
    var base: List[Int] = [1, 0, 0, -1, 0, 0, -1, 1]
    var rows = List[List[Int]]()
    var v = base.copy()
    for _ in range(9):
        rows.append(v.copy())
        var top = v[WORDS - 1]
        for k in range(WORDS - 1, 0, -1):
            v[k] = v[k - 1] + top * base[k]
        v[0] = top * base[0]
    return rows^


def _p256_coef(d: Int, k: Int) -> Int:
    """The coefficient of product word k + d in output word k: 1 for d = 0, else the fold pattern."""
    if d == 0:
        return 1
    var i = k + d
    if i < WORDS or i > 2 * WORDS:
        return 0
    return _p256_words()[i - WORDS][k]


def _p256_offsets() -> List[Int]:
    """Every word offset d that some output word reads with a nonzero coefficient (16 of 0..16)."""
    var v = List[Int]()
    for d in range(2 * WORDS + 1):
        for k in range(WORDS):
            if _p256_coef(d, k) != 0:
                v.append(d)
                break
    return v^


def _word(x1: Int) -> Int:
    """The output word of row x1 (weights below BITS), else -1."""
    return (Q * (ROWS - 2 - x1)) // WORD if _lo_row(x1) else -1


def _p256_pattern(i: Int) raises -> Big:
    """Product word i's value mod P-256 as a signed integer on the words 0..7."""
    var v = Big()
    for k in range(WORDS):
        var c = _p256_coef(i - k, k) if i >= k else 0
        if c != 0:
            v = v + Big(c).shl(WORD * k)
    return v^


def _p256_sum(r: Big) raises -> Big:
    """The Solinas sum of a product below 2^520: congruent to r mod P-256, in (-8 p, 8 p)."""
    var v = Big()
    for i in range(2 * WORDS + 1):
        var w = r.shr(WORD * i).low(WORD)
        if not w.is_zero():
            v = v + w * _p256_pattern(i)
    return v^


def _consts(curve: Int) raises -> Tuple[List[String], List[List[UInt8]]]:
    """The public columns that are the same on every chain, name and row values: the selectors `lo`,
    `cp` (secp256k1's fold) or `w{d}` (the P-256 word coefficients, -1 as 126) with `fp{j}` (P-256's p at
    its slots: the fold's modulus, since `pb` holds n on a mod-n chain), then `bd`."""
    var names = List[String]()
    var rows = List[List[UInt8]]()
    if curve == CURVE_K1:
        for name in ["lo", "cp"]:
            names.append(name)
            var v = List[UInt8](capacity=ROWS)
            for x1 in range(ROWS):
                v.append(UInt8(1) if (_lo_row(x1) if name == "lo" else _cp_row(x1)) else UInt8(0))
            rows.append(v^)
    elif curve == CURVE_P256:
        for d in _p256_offsets():
            names.append("w" + String(d))
            var v = List[UInt8](capacity=ROWS)
            for x1 in range(ROWS):
                var k = _word(x1)
                v.append(UInt8(((_p256_coef(d, k) if k >= 0 else 0) % 127 + 127) % 127))
            rows.append(v^)
        var pc = _columns(modulus(MOD_P, CURVE_P256).bits(FOLDED), False)
        for j in range(Q):
            names.append("fp" + String(j))
            var v = List[UInt8](capacity=ROWS)
            for x1 in range(ROWS):
                v.append(pc[j * ROWS + x1])
            rows.append(v^)
    else:
        raise Error("unknown curve")
    names.append("bd")
    var v = List[UInt8](capacity=ROWS)
    for x1 in range(ROWS):
        v.append(UInt8(1) if _bd_row(x1) else UInt8(0))
    rows.append(v^)
    return (names^, rows^)


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
struct Op(Copyable, Movable, ImplicitlyCopyable):
    """One op of a circuit. MUL: x y on the MUL lane (`z`, `s`, signs unused). ADD: x + sy y + sz z = s + q m
    on an add lane, `s` OUT (an output), PUB (a public constant: the op is a check) or NIL (masked to zero:
    the op checks congruence to zero); `qz` 1 forces q = 0; `mod` MOD_P or MOD_N. Operands are PUB, FREE (an
    add operand the host solves for, with `qz`), NIL (z only) or the index of an earlier op with an output."""
    var kind: Int
    var x: Int
    var y: Int
    var z: Int
    var s: Int
    var sy: Int
    var sz: Int
    var qz: Int
    var mod: Int


def mul(x: Int, y: Int) -> Op:
    return Op(MUL, x, y, NIL, OUT, 0, 0, 0, MOD_P)


def add(x: Int, y: Int, sy: Int = 1, z: Int = NIL, sz: Int = 0, s: Int = OUT, qz: Int = 0, mod: Int = MOD_P) -> Op:
    return Op(ADD, x, y, z, s, sy, sz if z != NIL else 0, qz, mod)


def sub(x: Int, y: Int) -> Op:
    return add(x, y, -1)


def eq(x: Int, y: Int, sy: Int = -1, z: Int = NIL, sz: Int = 0, mod: Int = MOD_P) -> Op:
    """x + sy y + sz z = 0 mod m."""
    return add(x, y, sy, z, sz, NIL, 0, mod)


def canon(x: Int) -> Op:
    """x + y = p - 1 for a free y: x < p. The public constant p - 1 is the op's public value."""
    return add(x, FREE, 1, NIL, 0, PUB, 1)


def guard(x: Int) -> Op:
    """x - y = 1 for a free y: x >= 1 as an integer; with `canon` before it, 1 <= x < p. The constant 1 is the
    op's public value."""
    return add(x, FREE, -1, NIL, 0, PUB, 1)


def hint(h: Int) -> Int:
    """Operand reference to witness `h` of the hints list (a slope, say): the prover supplies it, every
    occurrence is wired to the first, the bound `bd` holds on any lane."""
    return HINT - h


def single_op() -> List[Op]:
    var v: List[Op] = [mul(PUB, PUB)]
    return v^


def _place(ops: List[Op]) raises -> List[Tuple[Int, Int]]:
    """(chain, lane) of every op, lane -1 for the MUL lane: MUL ops take chains in order, add ops lanes in
    order, and a chain holds add ops of one modulus."""
    var v = List[Tuple[Int, Int]]()
    var muls = 0
    var chain = 0
    var lane = 0
    var mod = -1
    for j in range(len(ops)):
        var op = ops[j]
        _check_op(op, j)
        if op.kind == MUL:
            v.append((muls, -1))
            muls += 1
            continue
        if lane == LANES or (mod >= 0 and mod != op.mod and lane > 0):
            chain += 1
            lane = 0
        mod = op.mod
        v.append((chain, lane))
        lane += 1
    return v^


def _check_op(op: Op, j: Int) raises:
    """Every field of an op in its range, so that a circuit (from code or from a public-input header) never
    compiles to an unbound column set: operands are public, hints or earlier outputs (an add op may also
    take one free operand, with `qz` and a public `s`, and no `z`); `s` is an output, a constant or masked."""
    var refs: List[Int] = [op.x, op.y, op.z]
    var frees = 0
    for role in range(3):
        var r = refs[role]
        if r >= j or (r < 0 and r != PUB and r != FREE and r != NIL and r > HINT):
            raise Error("op " + String(j) + ": an operand is public, free, a hint or an earlier op")
        if r == FREE:
            frees += 1
        if op.kind == MUL and (r == FREE or (r == NIL) != (role == 2)):
            raise Error("op " + String(j) + ": a product takes two bound operands")
        if op.kind == ADD and r == NIL and role != 2:
            raise Error("op " + String(j) + ": only z may be absent")
    if op.kind == MUL:
        if op.s != OUT or op.sy != 0 or op.sz != 0 or op.qz != 0:
            raise Error("op " + String(j) + ": a product has an output and no signs or masks")
        return
    if op.kind != ADD:
        raise Error("unknown op kind")
    if op.s != OUT and op.s != PUB and op.s != NIL:
        raise Error("op " + String(j) + ": s is an output, a public constant or absent")
    if op.sy < -1 or op.sy > 1 or op.sz < -1 or op.sz > 1 or (op.z == NIL and op.sz != 0) or op.qz < 0 or op.qz > 1 or (op.mod != MOD_P and op.mod != MOD_N):
        raise Error("op " + String(j) + ": signs in {-1, 0, 1}, qz in {0, 1}, mod p or n")
    if frees > 1 or (frees == 1 and (op.qz != 1 or op.s != PUB)):
        raise Error("op " + String(j) + ": one free operand at most, with qz and a public s")


def chain_count(ops: List[Op]) raises -> Int:
    var at = _place(ops)
    var n = 0
    for j in range(len(ops)):
        n = max(n, at[j][0] + 1)
    return n


def _out_slot_of(op: Op, lane: Int) -> Int:
    return 2 if op.kind == MUL else 6 + 4 * lane


def _role_slot(op: Op, lane: Int, role: Int) -> Int:
    """The wiring slot of role 0 x, 1 y, 2 z, 3 s (a public constant), 4 the output."""
    if op.kind == MUL:
        return role if role < 2 else 2
    return 3 + 4 * lane + (role if role < 3 else 3)


def _slot_acc(slot: Int) -> String:
    if slot < 3:
        var names: List[String] = ["ha", "rb", "rf"]
        return names[slot]
    var lane = (slot - 3) // 4
    var names: List[String] = ["x", "y", "z", "s"]
    return "f" + String(lane) + names[(slot - 3) % 4]


def _has_out(op: Op) -> Bool:
    return op.s == OUT


def _factors(ops: List[Op]) -> List[Tuple[Int, Int]]:
    """(op, role) of every public factor in statement order: per op its PUB operands x, y, z then a PUB s;
    then every unconsumed output."""
    var v = List[Tuple[Int, Int]]()
    var consumed = List[Bool](length=len(ops), fill=False)
    for j in range(len(ops)):
        var op = ops[j]
        var refs: List[Int] = [op.x, op.y, op.z]
        for role in range(2 if op.kind == MUL else 3):
            if refs[role] == PUB:
                v.append((j, role))
            elif refs[role] >= 0:
                consumed[refs[role]] = True
        if op.kind == ADD and op.s == PUB:
            v.append((j, 3))
    for j in range(len(ops)):
        if _has_out(ops[j]) and not consumed[j]:
            v.append((j, 4))
    return v^


def _lane(L: Int, name: String) -> String:
    return "l" + String(L) + name


def product_columns(mut st: Statement) raises:
    """The product lane's W columns: a's pieces, b, the coefficient bits, r and the ripple carries."""
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


def plain_fingerprint(mut st: Statement, acc: String, name: String, rz: List[Int]) raises:
    """Accumulator `acc`: the plain fingerprint (at zeta) of the Q-column value `name`."""
    var terms = List[Term]()
    for j in range(Q):
        terms.append(Term(1, st.read(name + String(j)), chal=rz[j]))
    st.horner(acc, terms, scale=rz[Q])


def product_certificate(mut st: Statement, zeros: Bool = True) raises -> List[Int]:
    """The product lane on a statement with `product_columns` and the publics `s{t}` (piece rows) and `bd`
    (b's bound): the derived elements rz[10 t + k] = rho^t zeta^k (element indices, -1 for 1, returned), the
    accumulators ra (a piece-weighted), ha (a plain), rb, rc, the chain-end identity zeta^6 R_A R_B = R_C,
    the carry ripple into r, the coefficient alias, the piece selection, b's bound and (`zeros`) the zero
    rows on the idle row's carries."""
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
    plain_fingerprint(st, "rb", "b", rz)
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
    # b below 2^260 like the add-lane values: a hint operand has no factor or wire to bound it
    for j in range(Q):
        st.family("bbd" + String(j), [Term(1, st.read("b" + String(j))), Term(-1, st.read("bd"), st.read("b" + String(j)))])
    if zeros:
        for k in range(CARRY):
            st.zero(_y(k, Q - 1), FIX_E)
    return rz^


def mulmod_statement(zeros: Bool = True, circuit: List[Op] = List[Op](), pin: Bool = True, curve: Int = CURVE_K1, m: Int = 1) raises -> Statement:
    """`zeros = False` drops the zero rows: the unsound variant the test proves the idle-row carry against.
    `circuit` (default one product of public operands): see `Op`. `pin` puts the circuit bytes at the head
    of the public inputs (a workload whose circuit is fixed in code needs no header). `curve` picks the
    moduli and the product reduction: secp256k1's two folds, or the P-256 word pass of `_p256_families`. `m` is
    the period parameter of the chain-constant public columns: `m = h2` (the workload knows its grid) makes
    their public data one chain of values, which the verifier evaluates in `h1` products instead of `N`."""
    var ops = circuit.copy() if len(circuit) > 0 else single_op()
    var at = _place(ops)
    var st = Statement()
    product_columns(st)
    if curve == CURVE_K1:
        for name in ["h", "o", "g", "f"]:
            for j in range(Q):
                st.col(name + String(j), BIT)
        for name in ["z", "v"]:
            for k in range(FOLD):
                for j in range(Q):
                    st.col(name + String(k) + String(j), BIT)
    else:
        for j in range(Q):
            st.col("f" + String(j), BIT)
        for k in range(ACARRY):
            st.col("fq" + String(k), BIT)
        for k in range(ACARRY):
            for j in range(Q):
                st.col("fc" + String(k) + String(j), BIT)
    for L in range(LANES):
        for name in ["x", "y", "z", "s"]:
            for j in range(Q):
                st.col(_lane(L, name + String(j)), BIT)
        for k in range(QBITS):
            st.col(_lane(L, "q" + String(k)), BIT)
        for k in range(ACARRY):
            for j in range(Q):
                st.col(_lane(L, "c" + String(k) + String(j)), BIT)
    # the chain-constant columns (`_consts`, the piece rows) take period m: m = h2 makes them one chain of data
    for name in _consts(curve)[0]:
        st.pub(name, m)
    for j in range(Q):
        st.pub("pb" + String(j), 1)
    for t in range(PIECES):
        st.pub("s" + String(t), m)
    for L in range(LANES):
        for name in ["sy", "sz", "sm", "qz"]:
            st.pub(_lane(L, name), 1)
    if pin:
        st.pin(circuit_bytes(ops))
    var rz = product_certificate(st, zeros)
    plain_fingerprint(st, "rf", "f", rz)
    for L in range(LANES):
        for name in ["x", "y", "z", "s"]:
            plain_fingerprint(st, "f" + String(L) + name, _lane(L, name), rz)
    if zeros:
        for L in range(LANES):
            for k in range(ACARRY):
                st.zero(_lane(L, "c" + String(k) + String(Q - 1)), FIX_E)
        if curve == CURVE_P256:
            for k in range(ACARRY):
                st.zero("fc" + String(k) + String(Q - 1), FIX_E)
    if curve == CURVE_K1:
        _fold_families(st, "r", "h", "o", "z")
        _fold_families(st, "o", "g", "f", "v")
    else:
        _p256_families(st)
    for L in range(LANES):
        _add_families(st, L)
    for slot in range(3 + 4 * LANES):
        _ = st.slot(_slot_acc(slot))
    for f in _factors(ops):
        var slot = _role_slot(ops[f[0]], at[f[0]][1], f[1])
        st.public_factor("p" + String(f[0]) + "r" + String(f[1]), _slot_acc(slot), slot, at[f[0]][0])
    var first = Dict[Int, Tuple[Int, Int]]()
    for j in range(len(ops)):
        var op = ops[j]
        var refs: List[Int] = [op.x, op.y, op.z]
        for role in range(2 if op.kind == MUL else 3):
            var src = refs[role]
            var slot = _role_slot(op, at[j][1], role)
            if src >= 0:
                if not _has_out(ops[src]):
                    raise Error("op " + String(src) + " has no output")
                st.wire(slot, at[j][0], _out_slot_of(ops[src], at[src][1]), at[src][0])
            elif src <= HINT:
                var h = HINT - src
                if h in first:
                    st.wire(slot, at[j][0], first[h][0], first[h][1])
                else:
                    first[h] = (slot, at[j][0])
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


def _p256_families(mut st: Statement) raises:
    """The P-256 reduction in one pass (docs/mulmod.md): per position, the Solinas sum of the product's
    words read at their offsets under the coefficient selectors `w{d}`, minus q p (q = c0 + 2 c1 + 4 c2 -
    8 c3 chain-constant, p's bits from `fp` read k slots down for 2^k p), plus the carry in, equal f + 2 carry out, the carry signed
    like the add lane's. With the zero rows on the idle row's carry the pass is an exact integer identity:
    f = S - q p, congruent to a b mod p, and canonical when honest."""
    for j in range(Q):
        var terms = List[Term]()
        for d in _p256_offsets():
            terms.append(Term(1, st.read("w" + String(d)), st.read("r" + String(j), k1=(ROWS - WORD // Q * d) % ROWS)))
        for k in range(ACARRY):
            var at = _below(j, k)
            terms.append(Term(-_fsign(k), st.read("fq" + String(k)), st.read("fp" + String(at[1]), k1=at[0])))
        for k in range(ACARRY):
            terms.append(Term(_cweight(k), st.read("fc" + String(k) + String(j - 1)) if j > 0 else st.read("fc" + String(k) + String(Q - 1), k1=1)))
        terms.append(Term(-1, st.read("f" + String(j))))
        for k in range(ACARRY):
            terms.append(Term(-2 * _cweight(k), st.read("fc" + String(k) + String(j))))
        st.family("ffold" + String(j), terms)
    for k in range(ACARRY):
        st.family("fq" + String(k) + "const", [Term(1, st.read("fq" + String(k))), Term(-1, st.read("fq" + String(k), k1=1))])


def _qsign(k: Int) -> Int:
    """q = b0 + 2 b1 + 4 b2 - 2 b3: bit k weighs sign 2^shift."""
    return -1 if k == QBITS - 1 else 1


def _qshift(k: Int) -> Int:
    return 1 if k == QBITS - 1 else k


def _fsign(k: Int) -> Int:
    """The fold quotient q = c0 + 2 c1 + 4 c2 - 8 c3 (the carry's encoding): bit k weighs sign 2^k, the
    2^k from reading p's bits k slots down."""
    return -1 if k == ACARRY - 1 else 1


def _cweight(k: Int) -> Int:
    return -8 if k == ACARRY - 1 else 1 << k


def _carry_terms(mut st: Statement, L: Int, j: Int, sign: Int, k1: Int) -> List[Term]:
    """sign (c0 + 2 c1 + 4 c2 - 8 c3) at position j of the row k1 rows down."""
    var v = List[Term]()
    for k in range(ACARRY):
        v.append(Term(sign * _cweight(k), st.read(_lane(L, "c" + String(k) + String(j)), k1=k1)))
    return v^


def _add_families(mut st: Statement, L: Int) raises:
    """x + sy y + sz z + carry in = s + q m + 2 carry out per position, the carry signed; q chain-constant;
    x, y, z, s zero above WIDTH by `bd`; q zero under `qz`, s zero under `sm`."""
    for j in range(Q):
        var terms: List[Term] = [Term(1, st.read(_lane(L, "x" + String(j)))), Term(1, st.read(_lane(L, "sy")), st.read(_lane(L, "y" + String(j)))),
                                 Term(1, st.read(_lane(L, "sz")), st.read(_lane(L, "z" + String(j)))), Term(-1, st.read(_lane(L, "s" + String(j))))]
        for k in range(QBITS):
            var at = _below(j, _qshift(k))
            terms.append(Term(-_qsign(k), st.read(_lane(L, "q" + String(k))), st.read("pb" + String(at[1]), k1=at[0])))
        if j > 0:
            terms.extend(_carry_terms(st, L, j - 1, 1, 0))
        else:
            terms.extend(_carry_terms(st, L, Q - 1, 1, 1))
        terms.extend(_carry_terms(st, L, j, -2, 0))
        st.family(_lane(L, "add" + String(j)), terms)
    for name in ["x", "y", "z", "s"]:
        for j in range(Q):
            st.family(_lane(L, name + "bd" + String(j)), [Term(1, st.read(_lane(L, name + String(j)))), Term(-1, st.read("bd"), st.read(_lane(L, name + String(j))))])
    for k in range(QBITS):
        st.family(_lane(L, "q" + String(k) + "const"), [Term(1, st.read(_lane(L, "q" + String(k)))), Term(-1, st.read(_lane(L, "q" + String(k)), k1=1))])
        st.family(_lane(L, "q" + String(k) + "z"), [Term(1, st.read(_lane(L, "qz")), st.read(_lane(L, "q" + String(k))))])
    for j in range(Q):
        st.family(_lane(L, "s" + String(j) + "m"), [Term(1, st.read(_lane(L, "sm")), st.read(_lane(L, "s" + String(j))))])


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


def folded(a: Big, b: Big) raises -> Big:
    """The folded output of a product chain: a b under the fold twice, below 2^FOLDED."""
    var c = Big(977) + Big(1).shl(32)
    var t = a * b
    t = t.low(BITS) + t.shr(BITS) * c
    t = t.low(BITS) + t.shr(BITS) * c
    if t.bit_length() > FOLDED:
        raise Error("the folded result exceeds " + String(FOLDED) + " bits")
    return t^


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

def modulus(mod: Int, curve: Int = CURVE_K1) raises -> Big:
    """MOD_P: the curve's p; MOD_N: its group order n."""
    if curve == CURVE_K1:
        if mod == MOD_P:
            return Big.from_hex("fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f")
        if mod == MOD_N:
            return Big.from_hex("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141")
    elif curve == CURVE_P256:
        if mod == MOD_P:
            return Big.from_hex("ffffffff00000001000000000000000000000000ffffffffffffffffffffffff")
        if mod == MOD_N:
            return Big.from_hex("ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551")
    raise Error("unknown modulus")


def reduced(a: Big, b: Big, curve: Int) raises -> Tuple[Big, Int]:
    """A product chain's output and its quotient: secp256k1's fold twice (q unused), or a b mod P-256 with
    q = (S - f) / p for the Solinas sum S of a b, in [-8, 7]."""
    if curve == CURVE_K1:
        return (folded(a, b), 0)
    var p = modulus(MOD_P, CURVE_P256)
    var f = a.mulmod(b, p)
    var qr = (_p256_sum(a * b) - f).divmod(p)
    if not qr[1].is_zero():
        raise Error("the Solinas sum is not congruent to the product")
    var q = qr[0].to_int()
    if q < -8 or q > 7:
        raise Error("P-256 quotient out of range")
    return (f^, q)


@fieldwise_init
struct OpValues(Copyable, Movable):
    """The values on one op: x, y and the output s of a product; x, y, z, s and q of an add op."""
    var kind: Int
    var x: Big
    var y: Big
    var z: Big
    var s: Big
    var q: Int


def _bounded(v: Big, curve: Int) raises -> Big:
    """A chain value: below 2^WIDTH; below 2^BITS on P-256 (hints included: the honest carry range of the
    P-256 pass needs products below 2^512)."""
    var width = WIDTH if curve == CURVE_K1 else BITS
    if v.neg or v.bit_length() > width:
        raise Error("chain values are below 2^" + String(width))
    return v.copy()


def _input(inputs: List[List[UInt8]], next: Int, curve: Int) raises -> Big:
    """The next public value: VALUE bytes below 2^FOLDED; below 2^BITS on P-256, where the honest fold
    carries are bounded for products below 2^512 (the reduction stays exact on any bounded operand)."""
    var top = 1 if curve == CURVE_K1 else 0
    if next >= len(inputs) or len(inputs[next]) != VALUE or Int(inputs[next][VALUE - 1]) > top:
        raise Error("public values are " + String(VALUE) + " bytes below 2^" + String(BITS + top) + ", one per public reference")
    return Big.from_bytes(inputs[next])


def circuit_values(inputs: List[List[UInt8]], ops: List[Op], hints: List[Big] = List[Big](), curve: Int = CURVE_K1) raises -> List[OpValues]:
    """Every op's values, public operands taken from `inputs` in circuit order (x, y, z, then a PUB s), hint
    operands from `hints` by index; a product's q is the P-256 fold quotient (0 on secp256k1)."""
    _ = _place(ops)
    var vals = List[OpValues]()
    var next = 0
    var ms: List[Big] = [modulus(MOD_P, curve), modulus(MOD_N, curve)]
    for j in range(len(ops)):
        var op = ops[j]
        var refs: List[Int] = [op.x, op.y, op.z]
        var signs: List[Int] = [1, op.sy, op.sz]
        var v = List[Big]()
        var free = -1
        for role in range(2 if op.kind == MUL else 3):
            var r = refs[role]
            if r == PUB:
                v.append(_input(inputs, next, curve))
                next += 1
            elif r >= 0:
                if not _has_out(ops[r]):
                    raise Error("op " + String(r) + " has no output")
                v.append(vals[r].s.copy())
            elif r == FREE and op.kind == ADD:
                free = role
                v.append(Big())
            elif r == NIL and op.kind == ADD and role == 2:
                v.append(Big())
            elif r <= HINT and HINT - r < len(hints):
                v.append(_bounded(hints[HINT - r], curve))
            else:
                raise Error("bad operand reference on op " + String(j))
        if op.kind == MUL:
            var prod = reduced(v[0], v[1], curve)
            vals.append(OpValues(MUL, _bounded(v[0], curve), _bounded(v[1], curve), Big(), prod[0].copy(), prod[1]))
            continue
        var known = Big()
        for role in range(3):
            if role == free or signs[role] == 0:
                continue
            if signs[role] > 0:
                known = known + v[role]
            else:
                known = known - v[role]
        var m = ms[op.mod].copy()
        var sval = Big()
        var q = 0
        if op.s == PUB:
            sval = _input(inputs, next, curve)
            next += 1
        if free >= 0:
            if op.qz != 1 or op.s == OUT or signs[free] == 0:
                raise Error("a free operand needs qz, a signed slot and a checked s")
            var d = sval - known
            if signs[free] < 0:
                d = -d
            v[free] = _bounded(d, curve)
        elif op.s == OUT:
            var qr = known.divmod(m)
            q = qr[0].to_int()
            sval = qr[1].copy()
        else:
            var qr = (known - sval).divmod(m)
            if not qr[1].is_zero():
                raise Error("the check on op " + String(j) + " does not hold")
            q = qr[0].to_int()
        if (op.qz == 1 and q != 0) or q < -2 or q > 7:
            raise Error("q out of range on op " + String(j))
        vals.append(OpValues(ADD, _bounded(v[0], curve), _bounded(v[1], curve), _bounded(v[2], curve), _bounded(sval, curve), q))
    if next != len(inputs):
        raise Error("more public values than the circuit references")
    return vals^


def _cols(layout: Layout, name: String, grouped: Bool) raises -> List[Int]:
    """The column indices of a value's column set (or a's pieces), (g * Q + j) order."""
    var v = List[Int]()
    for g in range(PIECES if grouped else 1):
        for j in range(Q):
            v.append(layout.col(name + (String(g) if grouped else String("")) + String(j)))
    return v^


def _put(mut chain: List[UInt8], cols: List[Int], bits: List[Int], grouped: Bool):
    """A value's bits into the column set `cols` (or a's pieces) of a chain buffer (column stride ROWS)."""
    for i in range(len(bits)):
        if bits[i] != 0:
            var g = _piece(i) if grouped else 0
            chain[cols[g * Q + i % Q] * ROWS + _row(i)] = 1


def mulmod_trace[p: Params](layout: Layout, a: List[UInt8], b: List[UInt8], cheat: Int = -1) raises -> List[UInt8]:
    """One product chain from the 32-byte operands; see `circuit_trace`."""
    var inputs: List[List[UInt8]] = [value_bytes_of(a), value_bytes_of(b)]
    return circuit_trace[p](layout, circuit_values(inputs, single_op()), single_op(), cheat)


def circuit_trace[p: Params](layout: Layout, vals: List[OpValues], ops: List[Op], cheat: Int = -1, curve: Int = CURVE_K1) raises -> List[UInt8]:
    """Every op from its values on its chain and lane; the other chains idle. On chain `cheat` (if any) the
    idle row's slot 3 holds a pile of two and the product ripple starts from carry 1, so r = a b + 1 there
    satisfies every family and only the zero row catches it. Each chain is written into a (columns, h1)
    buffer that fits the cache, then copied into the column-major trace; the chains run in parallel."""
    comptime h1 = p.h1()
    comptime N = p.N()
    var at = _place(ops)
    if h1 != ROWS or cheat >= p.h2() or chain_count(ops) > p.h2() or len(vals) != len(ops) or (cheat >= 0 and len(ops) == 0):
        raise Error("the instance needs " + String(ROWS) + " rows per chain, every chain on the grid, a value per op")
    var columns = layout.columns_w()
    var trace = List[UInt8](length=columns * N, fill=0)
    var mul_at = List[Int](length=p.h2(), fill=-1)
    var adds = List[List[Int]]()                   # per chain: its add ops
    for _ in range(p.h2()):
        adds.append(List[Int]())
    var lanes = List[List[Int]]()                  # per lane: x, y, z, s (Q each), q (QBITS), c (ACARRY x Q)
    for L in range(LANES):
        var v = List[Int]()
        for name in ["x", "y", "z", "s"]:
            v.extend(_cols(layout, _lane(L, name), False))
        for k in range(QBITS):
            v.append(layout.col(_lane(L, "q" + String(k))))
        for k in range(ACARRY):
            v.extend(_cols(layout, _lane(L, "c" + String(k)), False))
        lanes.append(v^)
    var mbits = List[List[Int]]()
    for mod in range(2):
        mbits.append(modulus(mod, curve).bits(SLOTS))
    for j in range(len(ops)):
        if ops[j].kind == ADD:
            adds[at[j][0]].append(j)
        else:
            mul_at[at[j][0]] = j
    var ca = _cols(layout, "a", True)
    var cb = _cols(layout, "b", False)
    var cr = _cols(layout, "r", False)
    var cc = List[Int]()                           # (t * CBITS + m) * Q + j
    for t in range(PIECES):
        for m in range(CBITS):
            cc.extend(_cols(layout, "c" + String(t) + String(m), False))
    var cy = List[Int]()                           # k * Q + j
    for k in range(CARRY):
        cy.extend(_cols(layout, "y" + String(k), False))
    var k1_fold = curve == CURVE_K1
    var fold1 = _fold_cols(layout, "r", "h", "o", "z") if k1_fold else _p256_cols(layout)
    var fold2 = _fold_cols(layout, "o", "g", "f", "v") if k1_fold else List[Int]()
    var table = List[List[Int]]() if k1_fold else _p256_table()
    var failed = List[Int](length=p.h2(), fill=0)
    var errors = List[String](length=p.h2(), fill=String(""))

    @parameter
    def one_chain(x2: Int):
        """Chain x2 into its own buffer, then into its rows of the trace; an error is kept for the caller."""
        var live = mul_at[x2] >= 0
        if not live and len(adds[x2]) == 0 and x2 != cheat:
            return
        var chain = List[UInt8](length=columns * h1, fill=0)
        try:
            for j in adds[x2]:
                _add_lane[p](chain, lanes[at[j][1]], vals[j], ops[j], mbits[ops[j].mod])
            if live or x2 == cheat:
                _mul_chain[p](chain, x2 == cheat, vals[mul_at[x2] if live else 0], live, ca, cb, cr, cc, cy)
                if k1_fold:
                    _fold_chain[p](chain, fold1)
                    _fold_chain[p](chain, fold2)
                else:
                    _p256_chain[p](chain, fold1, vals[mul_at[x2]].q if live else 0, mbits[MOD_P], table)
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


def _mul_chain[p: Params](mut chain: List[UInt8], cheat: Bool, v: OpValues, live: Bool,
                          ca: List[Int], cb: List[Int], cr: List[Int], cc: List[Int], cy: List[Int]) raises:
    """The product lane of one chain buffer (column stride h1): a's pieces, b, the certified
    coefficient bits and the pile per slot, then the carry ripple into r."""
    comptime h1 = p.h1()
    var ab = v.x.bits(WIDTH) if live else List[Int]()
    var bb = v.y.bits(WIDTH) if live else List[Int]()
    _put(chain, ca, ab, True)
    _put(chain, cb, bb, False)
    var ones_b = List[Int]()
    for j in range(len(bb)):
        if bb[j] != 0:
            ones_b.append(j)
    var pile = List[Int](length=SLOTS + CBITS + Q, fill=0)     # certified bits stored per slot
    for t in range(PIECES):
        var lo = t * PIECE
        var hi = min((t + 1) * PIECE, len(ab)) if t < PIECES - 1 else len(ab)
        var coef = List[Int](length=PIECE + len(bb) + Q, fill=0)   # the coefficient at weight lo + k
        for i in range(lo, hi):
            if ab[i] != 0:
                for j in ones_b:
                    coef[i - lo + j] += 1
        for k in range(len(coef)):
            var c = coef[k]
            if c == 0:
                continue
            var w = lo + k
            var b6 = 1 if c >= 64 else 0
            var b5 = 1 if c >= 32 and c < 64 else 0
            var val = c - 64 * b6 - 32 * b5
            for m in range(CBITS):
                var bit = b6 if m == 6 else (b5 if m == 5 else (val >> m) & 1)
                if bit == 1:
                    chain[cc[(t * CBITS + m) * Q + (w + m) % Q] * h1 + _row(w + m)] = 1
                    pile[w + m] += 1
    var carry = 0
    if cheat:
        var off = h1 - 1
        chain[cc[Q - 1] * h1 + off] = 1
        chain[cc[Q + Q - 1] * h1 + off] = 1
        chain[cy[Q - 1] * h1 + off] = 1
        carry = 1
    for w in range(SLOTS):
        var s = carry + pile[w]
        var at_w = _row(w)
        chain[cr[w % Q] * h1 + at_w] = UInt8(s & 1)
        carry = s >> 1
        for k in range(CARRY):
            chain[cy[k * Q + w % Q] * h1 + at_w] = UInt8((carry >> k) & 1)

def _fold_cols(layout: Layout, src: String, copy: String, dst: String, carry: String) raises -> List[Int]:
    """src, copy, dst (Q each), then carry (FOLD x Q)."""
    var v = _cols(layout, src, False)
    v.extend(_cols(layout, copy, False))
    v.extend(_cols(layout, dst, False))
    for k in range(FOLD):
        v.extend(_cols(layout, carry + String(k), False))
    return v^


def _fold_chain[p: Params](mut chain: List[UInt8], cols: List[Int]):
    """The copy and the fold ripple of one chain buffer (column stride h1), the families of
    `_fold_families` evaluated in row order; `cols` from `_fold_cols`."""
    comptime h1 = p.h1()
    var shifts = _shifts()
    for x1 in range(h1):
        if _cp_row(x1):
            for j in range(Q):
                chain[cols[Q + j] * h1 + x1] = chain[cols[j] * h1 + (x1 + UP) % h1]
    var cy = 0
    for w in range(SLOTS):
        var x1 = _row(w)
        var s = cy + (Int(chain[cols[w % Q] * h1 + x1]) if _lo_row(x1) else 0)
        for sh in shifts:
            var at = _below(w % Q, sh)
            s += Int(chain[cols[Q + at[1]] * h1 + (x1 + at[0]) % h1])
        chain[cols[2 * Q + w % Q] * h1 + x1] = UInt8(s & 1)
        cy = s >> 1
        for k in range(FOLD):
            chain[cols[3 * Q + k * Q + w % Q] * h1 + x1] = UInt8((cy >> k) & 1)


def _p256_cols(layout: Layout) raises -> List[Int]:
    """r, f (Q each), fq (ACARRY), then fc (ACARRY x Q)."""
    var v = _cols(layout, "r", False)
    v.extend(_cols(layout, "f", False))
    for k in range(ACARRY):
        v.append(layout.col("fq" + String(k)))
    for k in range(ACARRY):
        v.extend(_cols(layout, "fc" + String(k), False))
    return v^


def _p256_table() -> List[List[Int]]:
    """table[d][k]: the coefficient of product word k + d in output word k, d = 0..16."""
    var t = List[List[Int]]()
    for d in range(2 * WORDS + 1):
        var v = List[Int]()
        for k in range(WORDS):
            v.append(_p256_coef(d, k))
        t.append(v^)
    return t^


def _p256_chain[p: Params](mut chain: List[UInt8], cols: List[Int], q: Int, mb: List[Int], table: List[List[Int]]) raises:
    """The P-256 pass of one chain buffer (column stride h1): q's bits, then per slot the Solinas pile
    from r's bits minus q p, rippled into f with the signed carry of `_p256_families`."""
    comptime h1 = p.h1()
    var rb = List[Int](capacity=SLOTS)
    for w in range(SLOTS):
        rb.append(Int(chain[cols[w % Q] * h1 + _row(w)]))
    var e = q + 16 if q < 0 else q
    for k in range(ACARRY):
        for x1 in range(h1):
            chain[cols[2 * Q + k] * h1 + x1] = UInt8((e >> k) & 1)
    var c = 0
    for w in range(SLOTS):
        var s = c
        if w < BITS:
            var word = w // WORD
            for d in range(len(table)):
                var coef = table[d][word]
                if coef != 0 and w + WORD * d < SLOTS:
                    s += coef * rb[w + WORD * d]
        for k in range(ACARRY):
            if w >= k:
                s -= _fsign(k) * ((e >> k) & 1) * mb[w - k]
        var out = s & 1
        c = (s - out) // 2
        if c < -8 or c > 7:
            raise Error("P-256 fold carry out of range")
        var x1 = _row(w)
        chain[cols[Q + w % Q] * h1 + x1] = UInt8(out)
        var ce = c + 16 if c < 0 else c
        for k in range(ACARRY):
            chain[cols[2 * Q + ACARRY + k * Q + w % Q] * h1 + x1] = UInt8((ce >> k) & 1)
    if c != 0:
        raise Error("P-256 fold does not close")


def _qbits(q: Int) -> List[Int]:
    """q = b0 + 2 b1 + 4 b2 - 2 b3."""
    var b3 = 1 if q < 0 else 0
    var e = q + 2 * b3
    var v: List[Int] = [e & 1, (e >> 1) & 1, (e >> 2) & 1, b3]
    return v^


def _add_lane[p: Params](mut chain: List[UInt8], cols: List[Int], v: OpValues, op: Op, mb: List[Int]) raises:
    """x, y, z, s, q and the signed carries of `_add_families` in row order on a chain buffer (column stride
    h1); `cols` the lane's columns as `circuit_trace` lists them, `mb` the modulus bits."""
    comptime h1 = p.h1()
    var xb = v.x.bits(WIDTH)
    var yb = v.y.bits(WIDTH)
    var zb = v.z.bits(WIDTH)
    var sb = v.s.bits(WIDTH)
    var vals: List[List[Int]] = [xb.copy(), yb.copy(), zb.copy(), sb.copy()]
    for r in range(4):
        var sub = List[Int]()
        for j in range(Q):
            sub.append(cols[r * Q + j])
        _put(chain, sub, vals[r], False)
    var qb = _qbits(v.q)
    for k in range(QBITS):
        for x1 in range(h1):
            chain[cols[4 * Q + k] * h1 + x1] = UInt8(qb[k])
    var sy = op.sy
    var sz = op.sz
    var c = 0
    for w in range(SLOTS):
        var l = c + (xb[w] if w < WIDTH else 0) + sy * (yb[w] if w < WIDTH else 0) + sz * (zb[w] if w < WIDTH else 0)
        var r = sb[w] if w < WIDTH else 0
        for k in range(QBITS):
            if w >= _qshift(k):
                r += _qsign(k) * qb[k] * mb[w - _qshift(k)]
        if (l - r) % 2 != 0:
            raise Error("addition carry is not integral")
        c = (l - r) // 2
        if c < -8 or c > 7:
            raise Error("addition carry out of range")
        var e = c + 16 if c < 0 else c
        for k in range(ACARRY):
            chain[cols[4 * Q + QBITS + k * Q + w % Q] * h1 + _row(w)] = UInt8((e >> k) & 1)
    if c != 0:
        raise Error("addition does not close")


# ---- workload ----

def _u16(mut v: List[UInt8], r: Int):
    """Two's complement."""
    var e = 65536 + r if r < 0 else r
    v.append(UInt8(e & 255))
    v.append(UInt8(e >> 8))


def _ref(bytes: List[UInt8], o: Int) -> Int:
    var e = Int(bytes[o]) | Int(bytes[o + 1]) << 8
    return e - 65536 if e >= 32768 else e


def circuit_bytes(ops: List[Op]) -> List[UInt8]:
    """count u16, then OP_BYTES per op: kind, x, y, z, s, sy, sz, qz, mod (signs as 0, 1, 2 for -1)."""
    var v = List[UInt8]()
    _u16(v, len(ops))
    for op in ops:
        v.append(UInt8(op.kind))
        _u16(v, op.x)
        _u16(v, op.y)
        _u16(v, op.z)
        _u16(v, op.s)
        v.append(UInt8(2 if op.sy < 0 else op.sy))
        v.append(UInt8(2 if op.sz < 0 else op.sz))
        v.append(UInt8(op.qz))
        v.append(UInt8(op.mod))
    return v^


def parse_circuit(bytes: List[UInt8]) raises -> Tuple[List[Op], Int]:
    """The circuit at the head of the public inputs and the offset of the values after it."""
    if len(bytes) < 2:
        raise Error("public inputs start with the circuit")
    var n = Int(bytes[0]) | Int(bytes[1]) << 8
    if len(bytes) < 2 + n * OP_BYTES:
        raise Error("public inputs start with the circuit")
    var v = List[Op]()
    for j in range(n):
        var o = 2 + j * OP_BYTES
        var sy = Int(bytes[o + 9])
        var sz = Int(bytes[o + 10])
        v.append(Op(Int(bytes[o]), _ref(bytes, o + 1), _ref(bytes, o + 3), _ref(bytes, o + 5), _ref(bytes, o + 7),
                    -1 if sy == 2 else sy, -1 if sz == 2 else sz, Int(bytes[o + 11]), Int(bytes[o + 12])))
    return (v^, 2 + n * OP_BYTES)


def const_bytes(v: Big) raises -> List[UInt8]:
    """A public constant as a VALUE-byte value."""
    return v.bytes(VALUE)


struct Mulmod(Workload, Copyable, Movable):
    """A circuit of ops (products, signed additions, checks mod p or n) on the chains of a ROWS x h2 grid
    (`mulmod_statement`). Public inputs: the circuit bytes, then per public operand or constant and per
    unconsumed output, in statement order, the VALUE-byte little-endian value."""
    var inputs: List[List[UInt8]]
    var circuit: List[Op]
    var hints: List[Big]

    def __init__(out self, var inputs: List[List[UInt8]], var circuit: List[Op], var hints: List[Big] = List[Big]()):
        self.inputs = inputs^
        self.circuit = circuit^ if len(circuit) > 0 else single_op()
        self.hints = hints^

    def statement[p: Params](self) raises -> Statement:
        return mulmod_statement(circuit=self.circuit)

    def trace[p: Params](self, layout: Layout) raises -> List[UInt8]:
        return circuit_trace[p](layout, circuit_values(self.inputs, self.circuit, self.hints), self.circuit)

    def public_inputs[p: Params](self) raises -> List[UInt8]:
        var vals = circuit_values(self.inputs, self.circuit, self.hints)
        var v = circuit_bytes(self.circuit)
        var next = 0
        for f in _factors(self.circuit):
            if f[1] < 4:
                v.extend(self.inputs[next].copy())
                next += 1
            else:
                v.extend(vals[f[0]].s.bytes(VALUE))
        return v^

    @staticmethod
    def public_data[p: Params](layout: Layout, public_inputs: List[UInt8]) raises -> List[UInt8]:
        var parsed = parse_circuit(public_inputs)
        return circuit_public_data[p](parsed[0], public_inputs, parsed[1])


def circuit_public_data[p: Params](ops: List[Op], values: List[UInt8], off: Int, curve: Int = CURVE_K1, m: Int = 1) raises -> List[UInt8]:
    """The public blocks of `_consts` (the same on every chain), pb{j} (the chain's modulus bits), s{t} (the
    rows of piece t), per lane sy, sz, sm, qz; then every factor's ingest columns in statement order: an a
    operand in its pieces (12 columns), any other value plain (4). The factor values are the VALUE-byte
    values at `off` of `values`, in factor order. `m` is the statement's period parameter of the constant
    columns: their blocks are `h2 / m` chains."""
    comptime N = p.N()
    if p.h1() != ROWS:
        raise Error("the instance needs " + String(ROWS) + " rows per chain")
    var at = _place(ops)
    if chain_count(ops) > p.h2() or (len(values) - off) % VALUE != 0:
        raise Error("public inputs are the circuit then " + String(VALUE) + "-byte values")
    var consts = _consts(curve)[1].copy()
    if p.h2() % m != 0:
        raise Error("the constant columns' period m divides h2")
    var data = List[UInt8](capacity=(len(consts) + PIECES) * (N // m) + (Q + 4 * LANES) * N + (len(values) - off) // VALUE * PIECES * Q * ROWS)
    for rows in consts:
        for _ in range(p.h2() // m):
            data.extend(rows.copy())
    var mods = List[Int](length=p.h2(), fill=MOD_P)
    var lanes = List[Int](length=p.h2() * LANES, fill=-1)
    for j in range(len(ops)):
        if at[j][1] >= 0:
            mods[at[j][0]] = ops[j].mod
            lanes[at[j][0] * LANES + at[j][1]] = j
    var mcols = List[List[UInt8]]()
    for mod in range(2):
        mcols.append(_columns(modulus(mod, curve).bits(FOLDED), False))
    for j in range(Q):
        for x2 in range(p.h2()):
            for x1 in range(ROWS):
                data.append(mcols[mods[x2]][j * ROWS + x1])
    for t in range(PIECES):
        for _ in range(p.h2() // m):
            for x1 in range(ROWS):
                data.append(UInt8(1) if _piece_row(t, x1) else UInt8(0))
    for L in range(LANES):
        for name in range(4):
            for x2 in range(p.h2()):
                var j = lanes[x2 * LANES + L]
                var b = 0
                if j >= 0:
                    var op = ops[j]
                    var sg = op.sy if name == 0 else op.sz
                    b = (126 if sg < 0 else sg) if name < 2 else ((1 if op.s == NIL else 0) if name == 2 else op.qz)
                for _ in range(ROWS):
                    data.append(UInt8(b))
    var o = off
    for f in _factors(ops):
        data.extend(_value_columns(values, o, ops[f[0]].kind == MUL and f[1] == 0))
        o += VALUE
    if o != len(values):
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
