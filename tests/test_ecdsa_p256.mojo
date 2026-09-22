"""The P-256 ECDSA workload on the 144 x 1152 grid: the prover round trip on the RFC 6979 A.2.5 signature,
its proof size, and the proof rejected against a claim with a changed message."""

from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from core.params import CLIENT
from core.hash import Blake3
from workloads.bigint import Big
from workloads.ecurve import Point
from workloads.ecdsa_p256 import EcdsaP256
from workload import prove_workload, verify_workload

comptime p = CLIENT.grid(144, 1152)


def _rfc6979_sample() raises -> Tuple[Big, Big, Big, Point]:
    """RFC 6979 A.2.5, P-256 with SHA-256, message "sample"."""
    var q = Point(Big.from_hex("60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6"),
                  Big.from_hex("7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299"), False)
    var r = Big.from_hex("EFD48B2AACB6A8FD1140DD9CD45E81D69D2C877B56AAF991C34D0EA84EAF3716")
    var s = Big.from_hex("F7CB1C942D657C41D436C7A1B6E29F65F3E900DBB9AFF4064DC4AB2F843ACDA8")
    var e = Big.from_hex("AF2BDBE1AA9B6EC1E2ADE1D694F41FC71A831D0268E9891562113D8A62ADD1BF")
    return (r^, s^, e^, q^)


def test_prover_round_trip() raises:
    var sig = _rfc6979_sample()
    var w = EcdsaP256(sig[0].copy(), sig[1].copy(), sig[2].copy(), sig[3].copy())
    var claim = w.public_inputs[p]()
    var ctx = DeviceContext()
    var proof = prove_workload[p, Blake3, EcdsaP256](ctx, w)
    print("ecdsa_p256 proof bytes:", len(proof))
    var again = proof.copy()
    assert_true(verify_workload[p, Blake3, EcdsaP256](proof^, w, claim, profile=True))
    var tampered = claim.copy()
    tampered[64] ^= 1
    var accepted: Bool
    try:
        accepted = verify_workload[p, Blake3, EcdsaP256](again^, w, tampered)
    except:
        accepted = False
    assert_true(not accepted)
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
