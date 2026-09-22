"""One secp256k1 ECDSA signature on the 144 x 576 grid: setup by step (compile, the host walk, public data, advice),
prove warm, then profiled by stage; verify on the host with its profile."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from core.params import Params, CLIENT
from core.hash import Blake3
from prover import Prover, load_trace, load_advice, load_public
from relations.statement import advice
from workloads.bigint import Big
from workloads.ecdsa_k1 import secp256k1, EcdsaK1
from workload import verify_workload


def _ms(t0: Int) -> Int:
    return (perf_counter_ns() - t0) // 1000000


def run[p: Params]() raises:
    var ctx = DeviceContext()
    var c0 = secp256k1()
    var d = Big.from_hex("1e99423a4ed27608a15a2616a2b0e9e52ced330ac530edcc32c8ffc6a526aedd")
    var k = Big.from_hex("a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90")
    var e = Big.from_hex("4b688df40bcedbe641ddb16ff0a1842d9c67ea1c3bf63f3e0471baa664531d1a")
    var q = c0.mul(c0.g, d)
    var r = c0.mul(c0.g, k).x.mod(c0.n)
    var sg = k.inv_mod(c0.n).mulmod(e + r.mulmod(d, c0.n), c0.n)
    var w = EcdsaK1(r^, sg^, e^, q^)
    var t0 = perf_counter_ns()
    var c = w.statement[p]().compile[p]()
    var t_compile = _ms(t0)
    t0 = perf_counter_ns()
    var trace = w.trace[p](c.layout)
    var t_trace = _ms(t0)
    var inputs = w.public_inputs[p]()
    t0 = perf_counter_ns()
    var data = EcdsaK1.public_data[p](c.layout, inputs)
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
    var ok = verify_workload[p, Blake3, EcdsaK1](proof.copy(), w, inputs, profile=True)
    var vms = (perf_counter_ns() - t0) // 1000000
    print("ecdsa_k1  ", p, " proof ", len(proof), " B, warm prove ", warm, " ms, verify ", vms, " ms ", ok)
    for i in range(len(prover.profile_names)):
        if prover.profile_ms[i] > 0:
            print("  ", prover.profile_ms[i], " ms  ", prover.profile_names[i])
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks


def main() raises:
    run[CLIENT.grid(144, 576)]()
