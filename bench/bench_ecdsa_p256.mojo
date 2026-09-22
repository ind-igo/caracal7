"""One secp256r1 ECDSA signature on the 144 x 1152 grid: setup by step (compile, the host walk, public data, advice),
prove warm, then profiled by stage; verify on the host with its profile."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from core.params import Params, CLIENT
from core.hash import Blake3
from prover import Prover, load_trace, load_advice, load_public
from relations.statement import advice
from workloads.bigint import Big
from workloads.ecurve import Point
from workloads.ecdsa_p256 import EcdsaP256
from workload import verify_workload


def _ms(t0: Int) -> Int:
    return (perf_counter_ns() - t0) // 1000000


def run[p: Params]() raises:
    var ctx = DeviceContext()
    # RFC 6979 A.2.5, P-256 with SHA-256, message "sample"
    var q = Point(Big.from_hex("60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6"),
                  Big.from_hex("7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299"), False)
    var r = Big.from_hex("EFD48B2AACB6A8FD1140DD9CD45E81D69D2C877B56AAF991C34D0EA84EAF3716")
    var sg = Big.from_hex("F7CB1C942D657C41D436C7A1B6E29F65F3E900DBB9AFF4064DC4AB2F843ACDA8")
    var e = Big.from_hex("AF2BDBE1AA9B6EC1E2ADE1D694F41FC71A831D0268E9891562113D8A62ADD1BF")
    var w = EcdsaP256(r^, sg^, e^, q^)
    var t0 = perf_counter_ns()
    var c = w.statement[p]().compile[p]()
    var t_compile = _ms(t0)
    t0 = perf_counter_ns()
    var trace = w.trace[p](c.layout)
    var t_trace = _ms(t0)
    var inputs = w.public_inputs[p]()
    t0 = perf_counter_ns()
    var data = EcdsaP256.public_data[p](c.layout, inputs)
    var t_public = _ms(t0)
    t0 = perf_counter_ns()
    var idx = advice[p](c.layout, trace)
    var t_advice = _ms(t0)
    t0 = perf_counter_ns()
    var prover = Prover[p, Blake3](ctx, c^)
    var t_prover = _ms(t0)
    t0 = perf_counter_ns()
    load_trace[p, Blake3](ctx, prover, trace)
    load_advice[p, Blake3](ctx, prover, idx)
    load_public[p, Blake3](ctx, prover, data)
    ctx.synchronize()
    var t_load = _ms(t0)
    t0 = perf_counter_ns()
    _ = prover.prove(ctx, inputs)
    var t_cold = _ms(t0)
    print("  setup ms: compile ", t_compile, "  trace ", t_trace, "  public values ", t_public, "  advice ", t_advice,
          "  prover (arena, tables) ", t_prover, "  loads ", t_load, "  cold prove ", t_cold)
    _ = prover.prove(ctx, inputs)
    t0 = perf_counter_ns()
    var proof = prover.prove(ctx, inputs)
    var warm = (perf_counter_ns() - t0) // 1000000
    _ = prover.prove(ctx, inputs, profile=True)
    t0 = perf_counter_ns()
    var ok = verify_workload[p, Blake3, EcdsaP256](proof.copy(), w, inputs, profile=True)
    var vms = (perf_counter_ns() - t0) // 1000000
    print("ecdsa_p256", p, " proof ", len(proof), " B, warm prove ", warm, " ms, verify ", vms, " ms ", ok)
    for i in range(len(prover.profile_names)):
        if prover.profile_ms[i] > 0:
            print("  ", prover.profile_ms[i], " ms  ", prover.profile_names[i])
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks


def main() raises:
    run[CLIENT.grid(144, 1152)]()
