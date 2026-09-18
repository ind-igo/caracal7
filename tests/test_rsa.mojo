"""RSA verify on the product chain (workloads/rsa.mojo): a 512-bit instance (2 limbs, e = 3, 22 chains: a triangle of 4 and two rectangles of 6) on
a 48-chain grid, so half the chains are idle. Every bit family holds on the trace, the proof verifies, a wrong m is refused on both sides."""

from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceContext

from core.params import CLIENT
from core.hash import Blake3
from relations import entry, ENTRY, NONE, NO_BASIS
from workloads.bigint import Big
from workloads.rsa import RSA, rsa_statement, rsa_trace, rsa_inputs, rsa_public_data, chain_count
from workload import prove_workload, verify_workload

comptime p = CLIENT.grid(144, 48)
comptime N = p.N()
comptime h1 = p.h1()


def _n() raises -> Big:
    return Big.from_hex("c3a9f2b4e6d8f0a1b2c3d4e5f60718293a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f7a8b9cad1")


def _s() raises -> Big:
    return Big.from_hex("7a1b2c3d4e5f60718293a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f7a8b9cadbecfd0e1f2a3b4c5")


def _families_hold(fams: List[UInt8], cz: Int, cw: Int, trace: List[UInt8], data: List[UInt8]) raises -> Int:
    """Every family without a challenge or a Z read sums to zero on every row; the count checked."""
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
    var row = ((x2 + k2) % p.h2()) * h1 + (x1 + k1) % h1
    if col < cw:
        return Int(trace[col * N + row])
    return Int(data[(col - cw - cz) * N + row])


def test_families_hold_and_proof_verifies() raises:
    var ctx = DeviceContext()
    var n = _n()
    var s = _s().mod(n)
    var m = s.mulmod(s, n).mulmod(s, n)
    assert_equal(chain_count(2, 2), 22)
    var w = RSA(2, 2, s, n, m)
    var c = rsa_statement(2, 2).compile[p]()
    var trace = rsa_trace[p](c.layout, 2, 2, s, n, m)
    var data = rsa_public_data[p](c.layout, rsa_inputs(2, 2, s, n, m))
    assert_true(_families_hold(c.families, c.shape.columns_z, c.layout.columns_w(), trace, data) > 200)
    var proof = prove_workload[p, Blake3, RSA](ctx, w)
    assert_true(verify_workload[p, Blake3, RSA](proof.copy(), w, w.public_inputs[p]()))
    var wrong = String("")
    try:
        _ = rsa_trace[p](c.layout, 2, 2, s, n, m + Big(1))
    except e:
        wrong = String(e)
    assert_equal(wrong, "the signature does not verify")
    try:
        _ = verify_workload[p, Blake3, RSA](proof.copy(), w, rsa_inputs(2, 2, s, n, m + Big(1)))
    except e:
        wrong = String(e)
    assert_equal(wrong, "public inputs differ")   # the transcript binds them before the factor check
    _ = ctx


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
