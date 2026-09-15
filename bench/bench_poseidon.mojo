"""Poseidon-M31 across input counts (the csp-benchmarks set 2, 4, 8, 12, 16): one chunk up to 8 inputs, two past.
Prove warm, then profiled by stage; verify on the host."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from caracal7.core.params import Params, CLIENT
from caracal7.core.hash import Blake3
from caracal7.prover import Prover, load_trace, load_advice, load_public
from caracal7.relations import value_bytes
from caracal7.relations.statement import advice
from caracal7.workloads.poseidon import Poseidon, M31
from caracal7.workload import verify_workload


def _ms(t0: Int) -> Int:
    return (perf_counter_ns() - t0) // 1000000


def run[p: Params](n: Int) raises:
    var ctx = DeviceContext()
    var elems = List[Int](capacity=n)
    for i in range(n):
        elems.append((i * 2654435761 + 12345) % M31)
    var w = Poseidon(elems^)
    var t0 = perf_counter_ns()
    var c = w.statement[p]().compile[p]()
    var t_compile = _ms(t0)
    t0 = perf_counter_ns()
    var trace = w.trace[p](c.layout)
    var t_trace = _ms(t0)
    var inputs = w.public_inputs[p]()
    t0 = perf_counter_ns()
    var data = Poseidon.public_data[p](c.layout, inputs)
    var t_public = _ms(t0)
    var n_blocks = value_bytes(c.layout.publics, p.h1(), p.h2())
    var blocks = List[UInt8](capacity=n_blocks)
    for i in range(n_blocks):
        blocks.append(data[i])
    t0 = perf_counter_ns()
    var idx = advice[p](c.layout, trace)
    var t_advice = _ms(t0)
    t0 = perf_counter_ns()
    var prover = Prover[p, Blake3](ctx, c^)
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
    var ok = verify_workload[p, Blake3, Poseidon](proof.copy(), w, inputs, profile=True)
    var vms = (perf_counter_ns() - t0) // 1000000
    print("poseidon-m31 of ", n, " elements  ", p, " proof ", len(proof), " B, warm prove ", warm, " ms, verify ", vms, " ms ", ok)
    for i in range(len(prover.profile_names)):
        if prover.profile_ms[i] > 0:
            print("  ", prover.profile_ms[i], " ms  ", prover.profile_names[i])
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks (decisions.md 2026-09-16)


def main() raises:
    run[CLIENT.grid(64, 368)](2)
    run[CLIENT.grid(64, 368)](8)
    run[CLIENT.grid(64, 720)](16)
