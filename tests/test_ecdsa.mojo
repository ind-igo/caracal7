"""The ECDSA circuit: op counts on the 144 x 896 grid, the live walk emitting the fixed circuit op for op,
the host solver closing on a valid signature and refusing a changed message, then the prover round trip,
its proof size, and the proof rejected against a claim with a changed message."""

from std.testing import assert_equal, assert_true, assert_raises, TestSuite
from max.gpu.host import DeviceContext

from caracal7.core.params import CLIENT
from caracal7.core.hash import Blake3
from caracal7.relations.bigint import Big
from caracal7.relations.ecdsa import Curve, Point, Ecdsa, walk
from caracal7.relations.mulmod import MUL, chain_count, circuit_values
from caracal7.workload import prove_workload, verify_workload

comptime p = CLIENT.grid(144, 896)


def _signature() raises -> Tuple[Big, Big, Big, Point]:
    var c = Curve()
    var d = Big.from_hex("1e99423a4ed27608a15a2616a2b0e9e52ced330ac530edcc32c8ffc6a526aedd")
    var k = Big.from_hex("a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90")
    var e = Big.from_hex("4b688df40bcedbe641ddb16ff0a1842d9c67ea1c3bf63f3e0471baa664531d1a")
    var q = c.mul(c.g, d)
    var r = c.mul(c.g, k).x.mod(c.n)
    var s = k.inv_mod(c.n).mulmod(e + r.mulmod(d, c.n), c.n)
    return (r^, s^, e^, q^)


def test_circuit_shape_and_host_walk() raises:
    var ops = Ecdsa.circuit()
    var muls = 0
    for op in ops:
        if op.kind == MUL:
            muls += 1
    assert_equal(muls, 786)
    assert_equal(len(ops) - muls, 1520)
    assert_true(chain_count(ops) <= p.h2())
    print("ops:", len(ops), "chains:", chain_count(ops))
    var sig = _signature()
    var w = walk(Curve(), sig[0], sig[1], sig[2], sig[3], True)
    assert_equal(len(w.ops), len(ops))
    for j in range(len(ops)):
        var a = ops[j]
        var b = w.ops[j]
        assert_true(a.kind == b.kind and a.x == b.x and a.y == b.y and a.z == b.z and a.s == b.s and a.sy == b.sy and a.sz == b.sz and a.qz == b.qz and a.mod == b.mod)
    print("public factors:", len(w.inputs), "hints:", len(w.hints))
    _ = circuit_values(w.inputs, w.ops, w.hints)
    var bad = walk(Curve(), sig[0], sig[1], sig[2] + Big(1), sig[3], True)
    with assert_raises(contains="does not hold"):
        _ = circuit_values(bad.inputs, bad.ops, bad.hints)
    with assert_raises(contains="(0, n)"):
        _ = walk(Curve(), sig[0], Big(), sig[2], sig[3], True)


def test_prover_round_trip() raises:
    var sig = _signature()
    var w = Ecdsa(sig[0].copy(), sig[1].copy(), sig[2].copy(), sig[3].copy())
    var claim = w.public_inputs[p]()
    var ctx = DeviceContext()
    var proof = prove_workload[p, Blake3, Ecdsa](ctx, w)
    print("ecdsa proof bytes:", len(proof))
    var again = proof.copy()
    assert_true(verify_workload[p, Blake3, Ecdsa](proof^, w, claim, profile=True))
    var tampered = claim.copy()
    tampered[64] ^= 1
    var accepted: Bool
    try:
        accepted = verify_workload[p, Blake3, Ecdsa](again^, w, tampered)
    except:
        accepted = False
    assert_true(not accepted)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
