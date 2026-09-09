"""One ECDSA signature on the 144 x 896 grid: setup by step (compile, the host walk, public data, advice),
prove warm, then profiled by stage; verify on the host with its profile."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from caracal7.core.params import Params, CLIENT
from caracal7.core.hash import Blake3
from caracal7.prover import Prover, load_trace, load_advice, load_public
from caracal7.relations import value_bytes
from caracal7.relations.statement import advice
from caracal7.workloads.bigint import Big
from caracal7.workloads.ecdsa import Curve, Ecdsa
from caracal7.workload import verify_workload


def _ms(t0: Int) -> Int:
    return (perf_counter_ns() - t0) // 1000000


def run[p: Params]() raises:
    var ctx = DeviceContext()
    var c0 = Curve()
    var d = Big.from_hex("1e99423a4ed27608a15a2616a2b0e9e52ced330ac530edcc32c8ffc6a526aedd")
    var k = Big.from_hex("a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90")
    var e = Big.from_hex("4b688df40bcedbe641ddb16ff0a1842d9c67ea1c3bf63f3e0471baa664531d1a")
    var q = c0.mul(c0.g, d)
    var r = c0.mul(c0.g, k).x.mod(c0.n)
    var sg = k.inv_mod(c0.n).mulmod(e + r.mulmod(d, c0.n), c0.n)
    var w = Ecdsa(r^, sg^, e^, q^)
    var t0 = perf_counter_ns()
    var c = w.statement().compile[p]()
    var t_compile = _ms(t0)
    t0 = perf_counter_ns()
    var trace = w.trace[p](c.layout)
    var t_trace = _ms(t0)
    var inputs = w.public_inputs[p]()
    t0 = perf_counter_ns()
    var data = Ecdsa.public_data[p](c.layout, inputs)
    var t_public = _ms(t0)
    var n_blocks = value_bytes(c.layout.publics, p.h1(), p.h2())
    var blocks = List[UInt8](capacity=n_blocks)
    for i in range(n_blocks):
        blocks.append(data[i])
    t0 = perf_counter_ns()
    var idx = advice[p](c.layout, trace)
    var t_advice = _ms(t0)
    var families = c.families.copy()
    t0 = perf_counter_ns()
    var prover = Prover[p, Blake3](ctx, c^.take_shape(), families^)
    var t_prover = _ms(t0)
    t0 = perf_counter_ns()
    load_trace[p, Blake3](ctx, prover, trace)
    load_advice[p, Blake3](ctx, prover, idx)
    load_public[p, Blake3](ctx, prover, blocks)
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
    var ok = verify_workload[p, Blake3, Ecdsa](proof.copy(), w, inputs, profile=True)
    var vms = (perf_counter_ns() - t0) // 1000000
    print("ecdsa  ", p, " proof ", len(proof), " B, warm prove ", warm, " ms, verify ", vms, " ms ", ok)
    for i in range(len(prover.profile_names)):
        if prover.profile_ms[i] > 0:
            print("  ", prover.profile_ms[i], " ms  ", prover.profile_names[i])


def main() raises:
    run[CLIENT.grid(144, 896)]()
