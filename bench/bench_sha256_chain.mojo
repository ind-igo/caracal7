"""SHA-256 hash chain, one proof per grid: m1 lanes of hashes_of(h2) hashes of 32 bytes, each of the previous
digest (bench/bench_sha256_blocks.mojo is the single-message case). Median warm prove of three runs, ms and bytes per
hash, committed cells per second, arena bytes (device memory), the host trace time, the stage split and the
share of the GEMM-skeleton stages (lde, quotient, open, fold). Jolt's SHA-2 chain on an M5 Max: 0.33 ms per hash."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from caracal7.core.params import Params, CLIENT
from caracal7.core.hash import Blake3
from caracal7.prover import Prover, load_trace, load_advice, load_public
from caracal7.relations import value_bytes
from caracal7.relations.statement import advice
from caracal7.workloads.sha256 import Sha256Chain, chain_hashes
from caracal7.workload import verify_workload

comptime JOLT_MS_PER_HASH = 0.33


def _ms(t0: Int) -> Float64:
    return Float64(perf_counter_ns() - t0) / 1e6


def _gemm_stage(name: String) -> Bool:
    for s in ["lde", "quotient", "open", "fold"]:
        if name.startswith(s):
            return True
    return False


def run[p: Params]() raises:
    var ctx = DeviceContext()
    var msg = List[UInt8](capacity=32)
    for i in range(32):
        msg.append(UInt8((i * 37 + 11) % 256))
    var w = Sha256Chain(msg^)
    var hashes = chain_hashes[p]()
    var c = w.statement[p]().compile[p]()
    var t0 = perf_counter_ns()
    var trace = w.trace[p](c.layout)
    var t_trace = _ms(t0)
    var inputs = w.public_inputs[p]()
    var data = Sha256Chain.public_data[p](c.layout, inputs)
    var pub_bytes = value_bytes(c.layout.publics, p.h1(), p.h2())
    var pub = List[UInt8](capacity=pub_bytes)
    for i in range(pub_bytes):
        pub.append(data[i])
    var idx = advice[p](c.layout, trace)
    t0 = perf_counter_ns()
    var prover = Prover[p, Blake3](ctx, c^)
    load_trace[p, Blake3](ctx, prover, trace)
    load_advice[p, Blake3](ctx, prover, idx)
    load_public[p, Blake3](ctx, prover, pub)
    ctx.synchronize()
    var setup = _ms(t0)
    var cells = prover.shape.columns_w * p.N()
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
        if not verify_workload[p, Blake3, Sha256Chain](proof^, w, inputs):
            raise Error("proof rejected")
        verify_ms += _ms(t0)
    sort(times)
    var median = times[1]
    _ = prover.prove(ctx, inputs, profile=True)
    var total = 0
    var gemm = 0
    for i in range(len(prover.profile_names)):
        total += prover.profile_ms[i]
        if _gemm_stage(prover.profile_names[i]):
            gemm += prover.profile_ms[i]
    print("sha256 chain ", hashes, " hashes, grid ", p.h1(), " x ", p.h2(), ", ", cells, " cells, ", prover.shape.columns_w,
          " columns, arena ", prover.layout.bytes >> 20, " MiB, setup ", Int(setup), " ms, host trace ", Int(t_trace), " ms")
    print("  warm prove median ", median, " ms (", times[0], " ", times[1], " ", times[2], "), ",
          median / Float64(hashes), " ms per hash (jolt ", JOLT_MS_PER_HASH, "), ",
          Float64(cells) / median / 1e3, " M cells/s, proof ", size, " B, ", size // hashes, " B per hash, verify ",
          verify_ms / 3, " ms")
    print("  profiled ", total, " ms, gemm stages ", gemm, " ms (", 100 * gemm // max(total, 1), "%)")
    for i in range(len(prover.profile_names)):
        if prover.profile_ms[i] > 0:
            print("    ", prover.profile_ms[i], " ms  ", prover.profile_names[i])
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks (decisions.md 2026-09-16)


def main() raises:
    print("SHA-256 hash chain, one proof per grid (Jolt: M5 Max, 0.33 ms per hash)")
    run[CLIENT.grid(32, 8064)]()       # 1 lane, 125 hashes
    run[CLIENT.grid(96, 8064)]()       # 3 lanes, 375
    run[CLIENT.grid(288, 2688)]()      # 9 lanes of 41, 369: two codewords per column
    run[CLIENT.grid(224, 8064)]()      # 7 lanes, 875
    run[CLIENT.grid(672, 2688)]()      # 21 lanes of 41, 861: the smaller chain axis
