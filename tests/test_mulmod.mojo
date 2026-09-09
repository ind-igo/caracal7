"""The 256-bit product on the builder: the trace against every bit-level family and the fingerprint identity
on the host, then the prover round trip on the 144 x 8 grid with a wrong product and the idle-row carry
rejected."""

from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceContext

from caracal7.core.params import CLIENT, Params
from caracal7.core.hash import Blake3
from caracal7.core.field import E, ext_mul
from caracal7.core.bytes import list_e
from caracal7.relations import entry, ENTRY, NONE, NO_BASIS, ACC, derived_chals, horner_chain_end
from caracal7.relations.mulmod import Mulmod, mulmod_statement, mulmod_trace, bits_of, bytes_of, product_bits, BITS
from caracal7.prover import Prover, load_trace
from caracal7.verifier import verify
from caracal7.workload import prove_workload, verify_workload

comptime p = CLIENT.grid(144, 8)


def operand(seed: Int) -> List[UInt8]:
    var v = List[UInt8](capacity=32)
    var s = seed
    for _ in range(32):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        v.append(UInt8((s >> 8) & 255))
    return v^


def _chals() -> List[UInt8]:
    var v = List[UInt8](capacity=48)
    for i in range(48):
        v.append(UInt8((i * 53 + 7) % 127))
    return v^


def test_product_bits() raises:
    var a: List[Int] = [1, 1, 0, 1]        # 11
    var b: List[Int] = [1, 0, 1]           # 5
    var r: List[Int] = [1, 1, 1, 0, 1, 1, 0]   # 55
    assert_equal(product_bits(a, b), r)
    assert_equal(bytes_of(bits_of(operand(5), 0, BITS)), operand(5))


def test_trace_satisfies_every_bit_family() raises:
    """Every family without a challenge or a Z read (Booleanity, ripple, alias) sums to zero on every row;
    the Horner fingerprints of chain 0 meet zeta^6 R_A R_B = R_C for arbitrary challenges; the public data's
    columns are the trace's."""
    comptime N = p.N()
    comptime h1 = p.h1()
    var c = mulmod_statement().compile[p]()
    var w = Mulmod(operand(1), operand(2))
    var trace = mulmod_trace[p](c.layout, w.a, w.b)
    var cw = c.layout.columns_w()
    var count = len(c.families) // ENTRY
    var families = 0
    for k in range(count):
        families = max(families, entry(c.families, k).family + 1)
    var skip = List[Bool](length=families, fill=False)
    for k in range(count):
        var en = entry(c.families, k)
        if en.chal != 0 or en.basis != NO_BASIS or en.col_a >= cw or (en.col_b != NONE and en.col_b >= cw):
            skip[en.family] = True
    var sums = List[Int](length=families * N, fill=0)
    var checked = 0
    for k in range(count):
        var en = entry(c.families, k)
        if skip[en.family]:
            continue
        checked += 1
        for x2 in range(p.h2()):
            for x1 in range(h1):
                if en.mult == 1 and x1 == h1 - 1:
                    continue
                var v = en.coef * Int(trace[en.col_a * N + ((x2 + en.dj2_a // 2) % p.h2()) * h1 + (x1 + en.dj1_a // 2) % h1])
                if en.col_b != NONE:
                    v *= Int(trace[en.col_b * N + ((x2 + en.dj2_b // 2) % p.h2()) * h1 + (x1 + en.dj1_b // 2) % h1])
                sums[en.family * N + x2 * h1 + x1] = (sums[en.family * N + x2 * h1 + x1] + v) % 127
    assert_true(checked > 124 + 4 * 32 + 12)
    for i in range(families * N):
        if sums[i] != 0:
            raise Error("family " + String(i // N) + " fails at row " + String(i % N))
    var chals = _chals()
    derived_chals(chals, c.shape.chals)
    var r = List[E]()
    for k in range(3):
        var first = Int(c.shape.accs[k * ACC + 2]) | Int(c.shape.accs[k * ACC + 3]) << 8
        var n = Int(c.shape.accs[k * ACC + 4]) | Int(c.shape.accs[k * ACC + 5]) << 8
        var cols = List[UInt8](capacity=n * h1)
        for i in range(n):
            var col = entry(c.families, first + i).col_a
            for x1 in range(h1):
                cols.append(trace[col * N + x1])
        r.append(horner_chain_end[p](c.families, c.shape.accs, k, cols, chals))
    var z6 = list_e(chals, Int(c.shape.ends[7]) - 1)
    assert_equal(ext_mul[4](ext_mul[4](r[0], r[1]), z6), r[2])
    var data = Mulmod.public_data[p](c.layout, w.public_inputs[p]())
    var off = 0
    for name in ["a00", "a01", "a02", "a03", "a10", "a11", "a12", "a13", "a20", "a21", "a22", "a23", "b0", "b1", "b2", "b3", "r0", "r1", "r2", "r3"]:
        for x1 in range(h1):
            assert_equal(data[off + x1], trace[c.layout.col(name) * N + x1])
        off += h1
    assert_equal(off, len(data))


def test_prover_round_trip() raises:
    var ctx = DeviceContext()
    var w = Mulmod(operand(1), operand(2))
    var proof = prove_workload[p, Blake3, Mulmod](ctx, w)
    print("mulmod proof bytes:", len(proof))
    assert_true(verify_workload[p, Blake3, Mulmod](proof^, w, w.public_inputs[p]()))


def _rejected(ctx: DeviceContext, trace: List[UInt8], claim: List[UInt8], zeros: Bool = True) raises -> String:
    var c = mulmod_statement(zeros).compile[p]()
    var shape = mulmod_statement(zeros).compile[p]().take_shape()
    var prover = Prover[p, Blake3](ctx, c^.take_shape(), mulmod_statement(zeros).compile[p]().families.copy())
    load_trace[p, Blake3](ctx, prover, trace)
    var proof = prover.prove(ctx, claim)
    var cc = mulmod_statement(zeros).compile[p]()
    var data = Mulmod.public_data[p](cc.layout, claim)
    try:
        _ = verify[p, Blake3](proof^, shape, claim, cc.families, data)
    except e:
        return String(e)
    return String("accepted")


def test_wrong_product_and_idle_carry_are_rejected() raises:
    """An honest trace with a claimed r off by one fails the public factor; a trace that carries a one into
    weight 0 from the idle row (every family holds, r = a b + 1 as claimed) fails the zero row, and is
    accepted by the statement without zero rows; the same carry on an idle chain, where no public factor
    looks, is caught by the zero row alone."""
    var ctx = DeviceContext()
    var w = Mulmod(operand(3), operand(4))
    var c = mulmod_statement().compile[p]()
    var claim = w.public_inputs[p]()
    var bits = bits_of(claim, 64, 2 * BITS)
    var carry = 1
    for i in range(len(bits)):
        var s = bits[i] + carry
        bits[i] = s & 1
        carry = s >> 1
    var wrong = w.a.copy()
    wrong.extend(w.b.copy())
    wrong.extend(bytes_of(bits))
    assert_equal(_rejected(ctx, mulmod_trace[p](c.layout, w.a, w.b), wrong), "wiring grand product is not the public factor")
    assert_equal(_rejected(ctx, mulmod_trace[p](c.layout, w.a, w.b, 0), wrong), "chain row is not zero")
    assert_equal(_rejected(ctx, mulmod_trace[p](c.layout, w.a, w.b, 0), wrong, False), "accepted")
    assert_equal(_rejected(ctx, mulmod_trace[p](c.layout, w.a, w.b, 3), claim), "chain row is not zero")
    assert_equal(_rejected(ctx, mulmod_trace[p](c.layout, w.a, w.b, 3), claim, False), "accepted")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
