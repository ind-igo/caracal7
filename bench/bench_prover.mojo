"""Whole-proof stage profile: prove once warm, then once with a synchronize after every stage.
The profiled run is slower in total (it serializes the stream); the split is what matters."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from caracal7.core.params import Params, CLIENT
from caracal7.core.hash import Blake3
from caracal7.proof import Shape
from caracal7.prover import Prover, load_trace, load_advice
from caracal7.workloads.synthetic import synthetic_statement, synthetic_trace, synthetic_advice, SYNTHETIC_COLUMNS, SYNTHETIC_LOOKUP_COLUMNS



def run[p: Params](name: String, lookup: Bool = False) raises:
    var ctx = DeviceContext()
    var cols = SYNTHETIC_LOOKUP_COLUMNS if lookup else SYNTHETIC_COLUMNS
    var c = synthetic_statement(cols, with_lookup=lookup).compile[p]()
    var prover = Prover[p, Blake3](ctx, synthetic_statement(cols, with_lookup=lookup).compile[p]().take_shape(), c.families.copy())
    load_trace[p, Blake3](ctx, prover, synthetic_trace[p](1, with_lookup=lookup))
    if lookup:
        load_advice[p, Blake3](ctx, prover, synthetic_advice[p]())
    _ = prover.prove(ctx, List[UInt8]())
    var t0 = perf_counter_ns()
    var proof = prover.prove(ctx, List[UInt8]())
    var warm = (perf_counter_ns() - t0) // 1000000
    _ = prover.prove(ctx, List[UInt8](), profile=True)
    print(name, " ", p, " proof ", len(proof), " B, warm prove ", warm, " ms")
    for i in range(len(prover.profile_names)):
        if prover.profile_ms[i] > 0:
            print("  ", prover.profile_ms[i], " ms  ", prover.profile_names[i])


def main() raises:
    comptime small = CLIENT.grid(72, 32)
    comptime wide = CLIENT.grid(288, 128)
    run[small]("72 x 32")
    run[small]("72 x 32 + lookup", lookup=True)
    run[wide]("288 x 128")
    run[wide]("288 x 128 + lookup", lookup=True)
