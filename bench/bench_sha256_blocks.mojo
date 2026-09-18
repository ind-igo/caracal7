"""SHA-256 cost per compression block: one message of 1024, 2048, 4096, 7936 bytes (17, 33, 65, 125 blocks with
padding), the median warm prove of three runs (setup and trace excluded), milliseconds per block, proof size,
verify time, every proof verified. Single-message SHA-256 on an M1 Pro against Jolt's 0.3275 ms per block on an
M5 Max for a long repeated-digest chain: the grids are the smallest legal ones past 64 b + 1 chains, so 4096 B
runs on the 7936 B grid."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from core.params import Params, CLIENT
from core.hash import Blake3
from prover import Prover, load_trace, load_advice, load_public
from relations import value_bytes
from relations.statement import advice
from workloads.sha256 import Sha256, blocks
from workload import verify_workload

comptime JOLT_MS_PER_BLOCK = 0.3275


def _ms(t0: Int) -> Float64:
    return Float64(perf_counter_ns() - t0) / 1e6


def run[p: Params](bytes: Int) raises:
    var ctx = DeviceContext()
    var msg = List[UInt8](capacity=bytes)
    for i in range(bytes):
        msg.append(UInt8((i * 37 + 11) % 256))
    var w = Sha256(msg^)
    var n_blocks = blocks(w.message)
    var c = w.statement[p]().compile[p]()
    var trace = w.trace[p](c.layout)
    var inputs = w.public_inputs[p]()
    var data = Sha256.public_data[p](c.layout, inputs)
    var pub_bytes = value_bytes(c.layout.publics, p.h1(), p.h2())
    var pub = List[UInt8](capacity=pub_bytes)
    for i in range(pub_bytes):
        pub.append(data[i])
    var idx = advice[p](c.layout, trace)
    var t0 = perf_counter_ns()
    var prover = Prover[p, Blake3](ctx, c^)
    load_trace[p, Blake3](ctx, prover, trace)
    load_advice[p, Blake3](ctx, prover, idx)
    load_public[p, Blake3](ctx, prover, pub)
    ctx.synchronize()
    var setup = _ms(t0)
    _ = prover.prove(ctx, inputs)          # cold: kernel compilation
    var times = List[Float64]()
    var size = 0
    var verify_ms = 0.0
    for _ in range(3):
        t0 = perf_counter_ns()
        var proof = prover.prove(ctx, inputs)
        times.append(_ms(t0))
        size = len(proof)
        t0 = perf_counter_ns()
        if not verify_workload[p, Blake3, Sha256](proof^, w, inputs):
            raise Error("proof rejected")
        verify_ms += _ms(t0)
    sort(times)
    var median = times[1]
    print("sha256 ", bytes, " B: ", n_blocks, " blocks, grid ", p.h1(), " x ", p.h2(), ", setup ", Int(setup), " ms")
    print("  warm prove median ", median, " ms (", times[0], " ", times[1], " ", times[2], "), ",
          median / Float64(n_blocks), " ms per block (jolt ", JOLT_MS_PER_BLOCK, "), proof ", size, " B, verify ",
          verify_ms / 3, " ms")
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks (decisions.md 2026-09-16)


def main() raises:
    print("single-message SHA-256, M1 Pro (Jolt: repeated-digest chain, M5 Max, 0.3275 ms per block)")
    run[CLIENT.grid(32, 1089)](1024)
    run[CLIENT.grid(32, 2113)](2048)
    run[CLIENT.grid(32, 4161)](4096)
    run[CLIENT.grid(32, 8001)](7936)
