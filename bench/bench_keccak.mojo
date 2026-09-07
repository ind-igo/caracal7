"""Keccak-256 across message lengths: one implementation, the grid derived from the block count. Prove warm,
then profiled by stage; verify on the host."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from caracal7.core.params import Params, CLIENT
from caracal7.core.hash import Blake3
from caracal7.prover import Prover, load_trace, load_advice, load_public
from caracal7.relations import block_bytes
from caracal7.relations.statement import advice
from caracal7.relations.keccak import Keccak
from caracal7.workload import verify_workload


def run[p: Params](bytes: Int) raises:
    var ctx = DeviceContext()
    var msg = List[UInt8](capacity=bytes)
    for i in range(bytes):
        msg.append(UInt8((i * 37 + 11) % 256))
    var w = Keccak(msg^)
    var c = w.statement().compile[p]()
    var trace = w.trace[p](c.layout)
    var inputs = w.public_inputs[p]()
    var data = Keccak.public_data[p](c.layout, inputs)
    var n_blocks = block_bytes(c.layout.publics, p.h1())
    var blocks = List[UInt8](capacity=n_blocks)
    for i in range(n_blocks):
        blocks.append(data[i])
    var idx = advice[p](c.layout, trace)
    var families = c.families.copy()
    var prover = Prover[p, Blake3](ctx, c^.take_shape(), families^)
    load_trace[p, Blake3](ctx, prover, trace)
    load_advice[p, Blake3](ctx, prover, idx)
    load_public[p, Blake3](ctx, prover, blocks)
    _ = prover.prove(ctx, inputs)
    var t0 = perf_counter_ns()
    var proof = prover.prove(ctx, inputs)
    var warm = (perf_counter_ns() - t0) // 1000000
    _ = prover.prove(ctx, inputs, profile=True)
    t0 = perf_counter_ns()
    var ok = verify_workload[p, Blake3, Keccak](proof.copy(), w, inputs)
    var vms = (perf_counter_ns() - t0) // 1000000
    print("keccak-256 of ", bytes, " B  ", p, " proof ", len(proof), " B, warm prove ", warm, " ms, verify ", vms, " ms (dense public blocks, host) ", ok)
    for i in range(len(prover.profile_names)):
        if prover.profile_ms[i] > 0:
            print("  ", prover.profile_ms[i], " ms  ", prover.profile_names[i])


def main() raises:
    run[CLIENT.grid(64, 24)](128)
    run[CLIENT.grid(64, 192)](1024)
    run[CLIENT.grid(64, 384)](2048)
