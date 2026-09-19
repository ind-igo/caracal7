"""The 256-bit product and its fold on the builder: the trace against every bit-level family (public
selector reads included) and the fingerprint identity on the host, the folded output congruent to a b mod p
by an independent long division, then the prover round trip on the 144 x 8 grid with a wrong result and the
idle-row carry rejected."""

from std.testing import assert_equal, assert_true, assert_raises, TestSuite
from max.gpu.host import DeviceContext

from core.params import CLIENT, Params
from core.hash import Blake3
from core.field import E, ext_mul, E_LEVEL, E_BYTES
from core.bytes import list_e
from relations import entry, ENTRY, NONE, NO_BASIS, ACC, derived_chals, horner_chain_end
from workloads.mulmod import Mulmod, Op, OpValues, mulmod_statement, mulmod_trace, circuit_trace, circuit_values, circuit_bytes, parse_circuit, single_op, value_bytes_of, const_bytes, modulus, chain_count, mul, add, sub, eq, canon, guard, hint, bits_of, bytes_of, product_bits, fold_bits, folded_bits, p_bits, add_bits, sub_bits, ge_bits, BITS, FOLDED, VALUE, WIDTH, MUL, ADD, PUB, FREE, NIL, OUT, MOD_P, MOD_N
from workloads.bigint import Big
from prover import Prover, load_trace, load_public
from verifier import verify
from workload import prove_workload, verify_workload

comptime p = CLIENT.grid(144, 8)


def operand(seed: Int) -> List[UInt8]:
    var v = List[UInt8](capacity=32)
    var s = seed
    for _ in range(32):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        v.append(UInt8((s >> 8) & 255))
    return v^


def single(a: List[UInt8], b: List[UInt8]) raises -> Mulmod:
    var inputs: List[List[UInt8]] = [value_bytes_of(a), value_bytes_of(b)]
    return Mulmod(inputs^, single_op())


def _chals() -> List[UInt8]:
    var v = List[UInt8](capacity=3 * E_BYTES)
    for i in range(3 * E_BYTES):
        v.append(UInt8((i * 53 + 7) % 127))
    return v^


def _mod_p(bits: List[Int]) -> List[Int]:
    """Long division, bit by bit from the top: BITS + 2 bits."""
    var n = BITS + 2
    var pb = p_bits(n)
    var rem = List[Int](length=n, fill=0)
    for i in range(len(bits) - 1, -1, -1):
        for k in range(n - 1, 0, -1):
            rem[k] = rem[k - 1]
        rem[0] = bits[i]
        if ge_bits(rem, pb):
            rem = sub_bits(rem, pb, n)
    return rem^


def test_fold_is_congruent_mod_p() raises:
    """The folded output of random operands equals a b mod p by long division and is below 2 p."""
    for seed in range(1, 6):
        var a = operand(seed)
        var b = operand(seed + 7)
        var f = folded_bits(bits_of(a, 0, BITS), bits_of(b, 0, BITS))
        assert_equal(len(f), FOLDED)
        var r = product_bits(bits_of(a, 0, BITS), bits_of(b, 0, BITS))
        assert_equal(_mod_p(f), _mod_p(r))
        var ff = f.copy()
        ff.resize(BITS + 2, 0)
        var p2 = add_bits(p_bits(BITS + 2), p_bits(BITS + 2), BITS + 2)
        assert_true(not ge_bits(ff, p2))
    var one = List[Int](length=BITS + 1, fill=0)
    one[BITS] = 1
    var folded = fold_bits(one)
    assert_equal(folded[32], 1)
    assert_equal(folded[0], 1)
    assert_equal(folded[9], 1)
    assert_equal(folded[5], 0)


def test_product_bits() raises:
    var a: List[Int] = [1, 1, 0, 1]        # 11
    var b: List[Int] = [1, 0, 1]           # 5
    var r: List[Int] = [1, 1, 1, 0, 1, 1, 0]   # 55
    assert_equal(product_bits(a, b), r)
    assert_equal(bytes_of(bits_of(operand(5), 0, BITS)), operand(5))


def test_trace_satisfies_every_bit_family() raises:
    """Every family without a challenge or a Z read (Booleanity, ripple, alias, the fold with its selector
    reads) sums to zero on every row; the Horner fingerprints of chain 0 meet zeta^6 R_A R_B = R_C for
    arbitrary challenges; the public data's columns are the trace's."""
    comptime N = p.N()
    comptime h1 = p.h1()
    var c = mulmod_statement().compile[p]()
    var w = single(operand(1), operand(2))
    var trace = mulmod_trace[p](c.layout, operand(1), operand(2))
    var cw = c.layout.columns_w()
    var data = Mulmod.public_data[p](c.layout, w.public_inputs[p]())
    assert_true(_families_hold(c.families, c.shape.columns_z, cw, trace, data) > 236 + 4 * 32 + 12 + 8 * 2 + 8 * 15 + 2 * (4 * 24 + 16 + 12))
    var P = modulus(MOD_P)
    var ops: List[Op] = [mul(PUB, PUB), add(0, PUB, -1, PUB, -1), canon(0), guard(0), add(PUB, PUB, 1, NIL, 0, OUT, 0, MOD_N)]
    var inputs: List[List[UInt8]] = [value_bytes_of(operand(3)), value_bytes_of(operand(4)), const_bytes(P - Big(1)), const_bytes(P - Big(1)),
                                     const_bytes(P - Big(1)), const_bytes(Big(1)), value_bytes_of(operand(5)), value_bytes_of(operand(6))]
    var wa = Mulmod(inputs.copy(), ops.copy())
    var ca = mulmod_statement(True, ops).compile[p]()
    var ta = circuit_trace[p](ca.layout, circuit_values(inputs, ops), ops)
    _ = _families_hold(ca.families, ca.shape.columns_z, ca.layout.columns_w(), ta, Mulmod.public_data[p](ca.layout, wa.public_inputs[p]()))
    var chals = _chals()
    derived_chals(chals, c.shape.chals)
    var r = List[E]()
    for k in range(5):
        var first = Int(c.shape.accs[k * ACC + 2]) | Int(c.shape.accs[k * ACC + 3]) << 8
        var n = Int(c.shape.accs[k * ACC + 4]) | Int(c.shape.accs[k * ACC + 5]) << 8
        var cols = List[UInt8](capacity=n * h1)
        for i in range(n):
            var col = entry(c.families, first + i).col_a
            for x1 in range(h1):
                cols.append(trace[col * N + x1])
        r.append(horner_chain_end[p](c.families, c.shape.accs, k, cols, chals))
    var z6 = list_e(chals, Int(c.shape.ends[7]) - 1)
    assert_equal(ext_mul[E_LEVEL](ext_mul[E_LEVEL](r[0], r[2]), z6), r[3])   # ra, ha, rb, rc, rf
    var off = 18 * N
    for name in ["a00", "a01", "a02", "a03", "a10", "a11", "a12", "a13", "a20", "a21", "a22", "a23", "b0", "b1", "b2", "b3", "f0", "f1", "f2", "f3"]:
        for x1 in range(h1):
            assert_equal(data[off + x1], trace[c.layout.col(name) * N + x1])
        off += h1
    assert_equal(off, len(data))


def _families_hold(fams: List[UInt8], cz: Int, cw: Int, trace: List[UInt8], data: List[UInt8]) raises -> Int:
    """Every family without a challenge or a Z read sums to zero on every row; the count checked."""
    comptime N = p.N()
    comptime h1 = p.h1()
    var count = len(fams) // ENTRY
    var families = 0
    for k in range(count):
        families = max(families, entry(fams, k).family + 1)
    var skip = List[Bool](length=families, fill=False)
    for k in range(count):
        var en = entry(fams, k)
        if en.chal != 0 or en.basis != NO_BASIS or (en.col_a >= cw and en.col_a < cw + cz) or (en.col_b != NONE and en.col_b >= cw and en.col_b < cw + cz):
            skip[en.family] = True
    var sums = List[Int](length=families * N, fill=0)
    var checked = 0
    for k in range(count):
        var en = entry(fams, k)
        if skip[en.family]:
            continue
        checked += 1
        for x2 in range(p.h2()):
            for x1 in range(h1):
                if en.mult == 1 and x1 == h1 - 1:
                    continue
                var v = en.coef * _at(trace, data, cw, cz, en.col_a, en.dj1_a // 2, en.dj2_a // 2, x1, x2)
                if en.col_b != NONE:
                    v *= _at(trace, data, cw, cz, en.col_b, en.dj1_b // 2, en.dj2_b // 2, x1, x2)
                sums[en.family * N + x2 * h1 + x1] = (sums[en.family * N + x2 * h1 + x1] + v) % 127
    for i in range(families * N):
        if sums[i] != 0:
            raise Error("family " + String(i // N) + " fails at row " + String(i % N))
    return checked


def _at(trace: List[UInt8], data: List[UInt8], cw: Int, cz: Int, col: Int, k1: Int, k2: Int, x1: Int, x2: Int) -> Int:
    """A read at (x1 + k1, x2 + k2): W from the trace, the public selector (past W and Z) from its dense block."""
    comptime N = p.N()
    comptime h1 = p.h1()
    var row = ((x2 + k2) % p.h2()) * h1 + (x1 + k1) % h1
    if col < cw:
        return Int(trace[col * N + row])
    return Int(data[(col - cw - cz) * N + row])


def test_prover_round_trip() raises:
    var ctx = DeviceContext()
    var w = single(operand(1), operand(2))
    var proof = prove_workload[p, Blake3, Mulmod](ctx, w)
    print("mulmod proof bytes:", len(proof))
    assert_true(verify_workload[p, Blake3, Mulmod](proof^, w, w.public_inputs[p]()))
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks (decisions.md 2026-09-16)


def _rejected(ctx: DeviceContext, trace: List[UInt8], claim: List[UInt8], zeros: Bool = True, circuit: List[Op] = List[Op]()) raises -> String:
    var c = mulmod_statement(zeros, circuit).compile[p]()
    var shape = mulmod_statement(zeros, circuit).compile[p]().take_shape()
    var cc = mulmod_statement(zeros, circuit).compile[p]()
    var data = Mulmod.public_data[p](cc.layout, claim)
    var prover = Prover[p, Blake3](ctx, c^.take_shape(), cc.families.copy())
    load_trace[p, Blake3](ctx, prover, trace)
    load_public[p, Blake3](ctx, prover, data)
    var proof = prover.prove(ctx, claim)
    try:
        _ = verify[p, Blake3](proof^, shape, claim, cc.families, data)
    except e:
        return String(e)
    return String("accepted")


def test_wrong_product_and_idle_carry_are_rejected() raises:
    """An honest trace with a claimed f off by one fails the public factor; a trace that carries a one into
    weight 0 from the idle row (every family holds, r = a b + 1, and f + 1 as claimed because the increment
    does not carry out of bit 255 for these operands) fails the zero row, and is accepted by the statement
    without zero rows; the same carry on an idle chain, where no public factor looks, is caught by the zero
    row alone. A claim past FOLDED bits is refused by the public data; a flipped copy bit fails a family."""
    var ctx = DeviceContext()
    var w = single(operand(3), operand(4))
    var c = mulmod_statement().compile[p]()
    var claim = w.public_inputs[p]()
    var head = parse_circuit(claim)[1] + 2 * VALUE
    var bits = bits_of(claim, head, FOLDED)
    var carry = 1
    for i in range(len(bits)):
        var s = bits[i] + carry
        bits[i] = s & 1
        carry = s >> 1
    var wrong = List[UInt8]()
    for i in range(head):
        wrong.append(claim[i])
    wrong.extend(bytes_of(bits))
    var a = w.inputs[0].copy()
    var b = w.inputs[1].copy()
    a.resize(BITS // 8, 0)
    b.resize(BITS // 8, 0)
    assert_equal(_rejected(ctx, mulmod_trace[p](c.layout, a, b), wrong), "wiring grand product is not the public factor")
    assert_equal(_rejected(ctx, mulmod_trace[p](c.layout, a, b, 0), wrong), "chain row is not zero")
    assert_equal(_rejected(ctx, mulmod_trace[p](c.layout, a, b, 0), wrong, False), "accepted")
    assert_equal(_rejected(ctx, mulmod_trace[p](c.layout, a, b, 3), claim), "chain row is not zero")
    assert_equal(_rejected(ctx, mulmod_trace[p](c.layout, a, b, 3), claim, False), "accepted")
    var high = claim.copy()
    high[len(high) - 1] |= 2
    with assert_raises(contains="exceeds"):
        _ = Mulmod.public_data[p](c.layout, high)
    var tampered = mulmod_trace[p](c.layout, a, b)
    tampered[c.layout.col("h1") * p.N() + 100] ^= 1
    assert_true(_rejected(ctx, tampered, claim) != "accepted")
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks (decisions.md 2026-09-16)


def test_three_wired_chains() raises:
    """Chains x y, then (x y) z with a from chain 0, then w (x y z) with b from chain 1: the output is x y z w mod p by
    long division, the proof round-trips, and a chain fed a wrong operand (every family holds) fails the
    wiring."""
    var circuit: List[Op] = [mul(PUB, PUB), mul(0, PUB), mul(PUB, 1)]
    var inputs: List[List[UInt8]] = [value_bytes_of(operand(11)), value_bytes_of(operand(12)), value_bytes_of(operand(13)), value_bytes_of(operand(14))]
    var w = Mulmod(inputs.copy(), circuit.copy())
    var claim = w.public_inputs[p]()
    var head = parse_circuit(claim)[1]
    assert_equal(len(claim), head + 5 * VALUE)
    var f = bits_of(claim, head + 4 * VALUE, FOLDED)
    var prod = product_bits(product_bits(bits_of(operand(11), 0, BITS), bits_of(operand(12), 0, BITS)),
                            product_bits(bits_of(operand(13), 0, BITS), bits_of(operand(14), 0, BITS)))
    assert_equal(_mod_p(f), _mod_p(prod))
    var ctx = DeviceContext()
    var proof = prove_workload[p, Blake3, Mulmod](ctx, w)
    print("three-chain proof bytes:", len(proof))
    assert_true(verify_workload[p, Blake3, Mulmod](proof^, w, claim))
    var c = mulmod_statement(True, circuit).compile[p]()
    var vals = circuit_values(inputs, circuit)
    vals[1].x = vals[1].x + Big(32)
    assert_true(_rejected(ctx, circuit_trace[p](c.layout, vals, circuit), claim, True, circuit) != "accepted")
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks (decisions.md 2026-09-16)


def _modp_mul(a: List[Int], b: List[Int]) -> List[Int]:
    return _mod_p(product_bits(a, b))


def _modp_sub(a: List[Int], b: List[Int]) -> List[Int]:
    """a - b mod p for a, b below p."""
    var n = BITS + 2
    var x = a.copy()
    x.resize(n, 0)
    var y = b.copy()
    y.resize(n, 0)
    if not ge_bits(x, y):
        x = add_bits(x, p_bits(n), n)
    return sub_bits(x, y, n)


def test_add_sub_canon_circuit() raises:
    """Ops x y, + z, - w, squared, checked canonical, - w: the output by independent host arithmetic, the round
    trip on two chains (two add lanes each), a non-canonical value refused by the honest prover, and a prover
    that passes the canonical check with q = 1 on a value past p (every other family holds) rejected by the
    mask; the same trace under a circuit header without the mask is refused as unpinned."""
    var circuit: List[Op] = [mul(PUB, PUB), add(0, PUB), sub(1, PUB), mul(2, 2), canon(3), sub(3, PUB)]
    var pm1 = modulus(MOD_P) - Big(1)
    var inputs: List[List[UInt8]] = [value_bytes_of(operand(21)), value_bytes_of(operand(22)), value_bytes_of(operand(23)), value_bytes_of(operand(24)), const_bytes(pm1), value_bytes_of(operand(24))]
    assert_equal(chain_count(circuit), 2)
    var w = Mulmod(inputs.copy(), circuit.copy())
    var claim = w.public_inputs[p]()
    var head = parse_circuit(claim)[1]
    assert_equal(len(claim), head + 7 * VALUE)
    var x = bits_of(operand(21), 0, BITS)
    var y = bits_of(operand(22), 0, BITS)
    var z = bits_of(operand(23), 0, BITS)
    var v = bits_of(operand(24), 0, BITS)
    var e = _mod_p(add_bits(_modp_mul(x, y), z, BITS + 2))
    e = _modp_sub(e, v)
    e = _modp_mul(e, e)
    e = _modp_sub(e, v)
    assert_equal(_mod_p(bits_of(claim, head + 6 * VALUE, FOLDED)), e)
    var ctx = DeviceContext()
    var proof = prove_workload[p, Blake3, Mulmod](ctx, w)
    print("six-op proof bytes:", len(proof))
    assert_true(verify_workload[p, Blake3, Mulmod](proof^, w, claim))
    var big = modulus(MOD_P) + Big(5)
    var check: List[Op] = [canon(PUB)]
    var bad: List[List[UInt8]] = [const_bytes(big), const_bytes(pm1)]
    with assert_raises(contains="below 2^260"):
        _ = Mulmod(bad.copy(), check.copy()).public_inputs[p]()
    var vals = List[OpValues]()
    vals.append(OpValues(ADD, big.copy(), pm1 + modulus(MOD_P) - big, Big(), pm1.copy(), 1))
    var c = mulmod_statement(True, check).compile[p]()
    var cheat = circuit_bytes(check)
    cheat.extend(const_bytes(big))
    cheat.extend(const_bytes(pm1))
    assert_true(_rejected(ctx, circuit_trace[p](c.layout, vals, check), cheat, True, check) != "accepted")
    var swap: List[Op] = [add(PUB, PUB, 1, NIL, 0, PUB, 0)]
    var swapped = circuit_bytes(swap)
    swapped.extend(const_bytes(big))
    swapped.extend(const_bytes(pm1 + modulus(MOD_P) - big))
    swapped.extend(const_bytes(pm1))
    var cs = mulmod_statement(True, check).compile[p]()
    var shape = mulmod_statement(True, check).compile[p]().take_shape()
    var prover = Prover[p, Blake3](ctx, cs^.take_shape(), mulmod_statement(True, check).compile[p]().families.copy())
    var ca = mulmod_statement(True, check).compile[p]()
    var swapped_data = Mulmod.public_data[p](ca.layout, swapped)
    load_trace[p, Blake3](ctx, prover, circuit_trace[p](c.layout, vals, check))
    load_public[p, Blake3](ctx, prover, swapped_data)
    var swapped_proof = prover.prove(ctx, swapped)
    with assert_raises(contains="pinned"):
        _ = verify[p, Blake3](swapped_proof^, shape, swapped, ca.families, swapped_data)
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks (decisions.md 2026-09-16)


def test_signed_ops_eq_guard_and_mod_n() raises:
    """A three-operand op with both signs (q negative), an EQ that closes it, the guard on a product, a
    sum mod n on its own chain: the values by host big-integer arithmetic, the round trip, a failing EQ and
    a failing guard refused by the honest prover, and a trace with a wrong EQ operand rejected."""
    var P = modulus(MOD_P)
    var Nn = modulus(MOD_N)
    var circuit: List[Op] = [mul(PUB, PUB), add(0, PUB, -1, PUB, -1), add(1, PUB, 1, PUB, 1), eq(0, 2), canon(0), guard(0), add(PUB, PUB, 1, NIL, 0, OUT, 0, MOD_N)]
    var a = Big.from_bytes(operand(31))
    var b = Big.from_bytes(operand(32))
    var y = P - Big(1)
    var z = P - Big(1)
    var u = Big.from_bytes(operand(35))
    var vv = Big.from_bytes(operand(36))
    var inputs: List[List[UInt8]] = [const_bytes(a), const_bytes(b), const_bytes(y), const_bytes(z), const_bytes(y), const_bytes(z),
                                     const_bytes(P - Big(1)), const_bytes(Big(1)), const_bytes(u), const_bytes(vv)]
    assert_equal(chain_count(circuit), 4)
    var vals = circuit_values(inputs, circuit)
    var f = a.mulmod(b, P)
    assert_equal(vals[0].s.mod(P), f)
    assert_equal(vals[1].s, (f - y - z).mod(P))
    assert_equal(vals[1].q, -2)
    assert_equal(vals[2].s, f.mod(P))
    assert_equal(vals[6].s, (u + vv).mod(Nn))
    var w = Mulmod(inputs.copy(), circuit.copy())
    var claim = w.public_inputs[p]()
    var ctx = DeviceContext()
    var proof = prove_workload[p, Blake3, Mulmod](ctx, w)
    print("signed-op proof bytes:", len(proof))
    assert_true(verify_workload[p, Blake3, Mulmod](proof^, w, claim))
    var wrong: List[Op] = [mul(PUB, PUB), eq(0, PUB)]
    var wi: List[List[UInt8]] = [const_bytes(a), const_bytes(b), const_bytes(f + Big(1))]
    with assert_raises(contains="does not hold"):
        _ = circuit_values(wi, wrong)
    var zero: List[Op] = [guard(PUB)]
    var zi: List[List[UInt8]] = [const_bytes(Big()), const_bytes(Big(1))]
    with assert_raises(contains="below 2^260"):
        _ = circuit_values(zi, zero)
    var c = mulmod_statement(True, circuit).compile[p]()
    vals[3].y = vals[3].y + Big(1)
    vals[3].x = vals[3].x + Big(1)
    assert_true(_rejected(ctx, circuit_trace[p](c.layout, vals, circuit), claim, True, circuit) != "accepted")
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks (decisions.md 2026-09-16)


def test_hint_operands() raises:
    """A slope hint on three operand slots (a of one product, a and b of another): the circuit holds with
    the honest hint, the honest prover refuses a wrong one, and a trace whose second occurrence is the hint
    plus p (every family holds) is rejected by the wire between the occurrences."""
    var P = modulus(MOD_P)
    var circuit: List[Op] = [mul(hint(0), PUB), eq(0, PUB), mul(hint(0), hint(0)), eq(2, PUB)]
    var dx = Big.from_bytes(operand(41)).mod(P)
    var l = Big.from_bytes(operand(42)).mod(P)
    var inputs: List[List[UInt8]] = [const_bytes(dx), const_bytes(l.mulmod(dx, P)), const_bytes(l.mulmod(l, P))]
    var hints: List[Big] = [l.copy()]
    assert_equal(chain_count(circuit), 2)
    var vals = circuit_values(inputs, circuit, hints)
    with assert_raises(contains="does not hold"):
        _ = circuit_values(inputs, circuit, [l + Big(1)])
    var w = Mulmod(inputs.copy(), circuit.copy(), hints.copy())
    var claim = w.public_inputs[p]()
    var ctx = DeviceContext()
    var proof = prove_workload[p, Blake3, Mulmod](ctx, w)
    print("hint proof bytes:", len(proof))
    assert_true(verify_workload[p, Blake3, Mulmod](proof^, w, claim))
    var lp = l + P
    vals[2] = OpValues(MUL, lp.copy(), lp.copy(), Big(), Big.from_bits(folded_bits(lp.bits(WIDTH), lp.bits(WIDTH))), 0)
    var c = mulmod_statement(True, circuit).compile[p]()
    assert_true(_rejected(ctx, circuit_trace[p](c.layout, vals, circuit), claim, True, circuit) != "accepted")
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks (decisions.md 2026-09-16)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
