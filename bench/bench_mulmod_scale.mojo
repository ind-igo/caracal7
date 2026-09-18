"""Prover cost against the chain count: a squaring chain of n 256-bit products on the 144 x n mulmod grid,
ECDSA's columns. 576 is ECDSA's grid, 1152 about one RSA-2048 verify as 3-level Karatsuba block products
(17 modmuls x 2 products x 27 blocks = 918). The largest mulmod grid is 1344 (11 wiring slots x h2 <= |F2*|),
whose circuit header pushes the transcript prefix past PREFIX_MAX (a pinned circuit carries none). Warm prove, proof size, verify (docs/decisions.md 2026-09-17, "RSA-2048 on the mulmod chain")."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from core.params import Params, CLIENT
from core.hash import Blake3
from prover import Prover, load_trace, load_advice, load_public
from relations import value_bytes
from relations.statement import advice
from workloads.mulmod import Mulmod, Op, mul, PUB, value_bytes_of
from workload import verify_workload


def operand(seed: Int) -> List[UInt8]:
    var v = List[UInt8](capacity=32)
    var s = seed
    for _ in range(32):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        v.append(UInt8((s >> 8) & 255))
    return v^


def run[p: Params](n: Int) raises:
    var ctx = DeviceContext()
    var ops = List[Op]()
    var inputs = List[List[UInt8]]()
    ops.append(mul(PUB, PUB))           # a squaring chain: two public values in, one factor out
    inputs.append(value_bytes_of(operand(1)))
    inputs.append(value_bytes_of(operand(2)))
    for i in range(1, n):
        ops.append(mul(i - 1, i - 1))
    var w = Mulmod(inputs^, ops^)
    var t0 = perf_counter_ns()
    var c = w.statement[p]().compile[p]()
    var t_compile = (perf_counter_ns() - t0) // 1000000
    t0 = perf_counter_ns()
    var trace = w.trace[p](c.layout)
    var t_trace = (perf_counter_ns() - t0) // 1000000
    var inp = w.public_inputs[p]()
    var data = Mulmod.public_data[p](c.layout, inp)
    var n_blocks = value_bytes(c.layout.publics, p.h1(), p.h2())
    var blocks = List[UInt8](capacity=n_blocks)
    for i in range(n_blocks):
        blocks.append(data[i])
    var idx = advice[p](c.layout, trace)
    var prover = Prover[p, Blake3](ctx, c^)
    load_trace[p, Blake3](ctx, prover, trace)
    load_advice[p, Blake3](ctx, prover, idx)
    load_public[p, Blake3](ctx, prover, blocks)
    ctx.synchronize()
    _ = prover.prove(ctx, inp)
    _ = prover.prove(ctx, inp)
    t0 = perf_counter_ns()
    var proof = prover.prove(ctx, inp)
    var warm = (perf_counter_ns() - t0) // 1000000
    t0 = perf_counter_ns()
    var ok = verify_workload[p, Blake3, Mulmod](proof.copy(), w, inp, profile=False)
    var vms = (perf_counter_ns() - t0) // 1000000
    print("products ", n, "  grid ", p, "  compile ", t_compile, " ms  trace ", t_trace, " ms  proof ", len(proof),
          " B  warm prove ", warm, " ms  verify ", vms, " ms ", ok)
    _ = ctx


def main() raises:
    run[CLIENT.grid(144, 576)](576)
    run[CLIENT.grid(144, 1152)](1152)
