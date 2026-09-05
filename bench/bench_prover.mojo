"""Whole-proof stage profile: prove once warm, then once with a synchronize after every stage.
The profiled run is slower in total (it serializes the stream); the split is what matters."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from caracal7.params import Params, REFERENCE
from caracal7.hash import Blake3
from caracal7.proof import Shape
from caracal7.prover import Prover, load_trace
from caracal7.synthetic import synthetic_families, synthetic_trace, SYNTHETIC_COLUMNS

comptime WIDE = Params(e=16, a1=5, m1=9, a2=7, m2=1, L0=161280, m_cosets=1, leaf_bytes=1024,
                       tail_digits=3, tail_clear_max=2500, lambda_bits=103)


def run[p: Params](name: String) raises:
    var ctx = DeviceContext()
    var f = synthetic_families()
    var prover = Prover[p, Blake3](ctx, Shape.__init__[p](SYNTHETIC_COLUMNS, f.bytes, f.accs), f.bytes.copy())
    load_trace[p, Blake3](ctx, prover, synthetic_trace[p](1))
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
    run[REFERENCE]("reference 72 x 32")
    run[WIDE]("wide 288 x 128")
