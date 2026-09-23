"""SHA-256 hash chain in the shape of OpenVM's `sha256_iter` guest: start from SHA-256 of the empty message and
hash the 32-byte digest n times (docs/bench-plan.md, Track 3). The chain is split into segments of one grid;
the last digest of a segment is the first message of the next, and every segment is its own proof (no
aggregation). One timer from the start value to the last proof's bytes, with the host trace, advice, public
data, loads and proves inside; the host builds the next segment while the device proves the current one. The
prover setup (arena, tables, a cold prove for the kernel compile) is a separate number.
Every segment proof is verified, and the last digest is checked against the host chain.

    ./bench_sha256_iter [n ...]      default: 1000 10000 150000

Two segment policies per n: 875 hashes (224 x 8064, with 375 and 125 for the rest), the fastest, and 125
hashes (32 x 8064), the only chain grid that the soundness ledger covers (the larger grids split codewords).
One JSON line per (n, policy)."""

from std.time import perf_counter_ns
from std.sys import argv
from max.gpu.host import DeviceContext

from core.params import Params, CLIENT
from core.hash import Blake3
from prover import Prover, load_trace, load_advice, load_public
from relations.statement import advice, Layout
from workloads.sha256 import Sha256Chain, sha256, sha256_chain, chain_hashes
from workload import prove_prepared, verify_workload

comptime G125 = CLIENT.grid(32, 8064)
comptime G375 = CLIENT.grid(96, 8064)
comptime G875 = CLIENT.grid(224, 8064)
# bench/bench_soundness.mojo, case sha256_chain 125, row johnson_dkt26_joint_list_conditional: conditional
# bits, not a certified level. The 375 and 875 grids split codewords, which the ledger does not cover.
comptime INTERACTIVE_BITS = "87.57"
comptime WORK_BITS = "107.2"


def _ms(t0: Int) -> Float64:
    return Float64(perf_counter_ns() - t0) / 1e6


def _hex(b: List[UInt8]) -> String:
    comptime digits = "0123456789abcdef"
    var s = String()
    for v in b:
        s += digits[byte=Int(v >> 4)]
        s += digits[byte=Int(v & 15)]
    return s^


struct Totals(Movable):
    var start: List[UInt8]      # the first message of the next segment
    var hashes: Int
    var segments: Int
    var prove_ms: Float64       # input to the last proof, verification excluded
    var host_ms: Float64        # host work: digest, trace, advice, public data (mostly under the device work)
    var load_ms: Float64        # uploads
    var setup_ms: Float64
    var verify_ms: Float64
    var proof_bytes: Int
    var arena_bytes: Int        # the largest arena of the grids used

    def __init__(out self, var start: List[UInt8]):
        self.start = start^
        self.hashes = 0
        self.segments = 0
        self.prove_ms = 0
        self.host_ms = 0
        self.load_ms = 0
        self.setup_ms = 0
        self.verify_ms = 0
        self.proof_bytes = 0
        self.arena_bytes = 0


struct Segment(Movable):
    """One segment's host data: built on the CPU while the device proves the previous segment."""
    var w: Sha256Chain
    var public: List[UInt8]
    var trace: List[UInt8]
    var idx: List[UInt8]
    var data: List[UInt8]

    def __init__[p: Params](out self, layout: Layout, start: List[UInt8]) raises:
        self.w = Sha256Chain(start.copy())
        self.public = self.w.public_inputs[p]()
        self.trace = self.w.trace[p](layout)
        self.idx = advice[p](layout, self.trace)
        self.data = Sha256Chain.public_data[p](layout, self.public)


def _load[p: Params](ctx: DeviceContext, mut prover: Prover[p, Blake3], s: Segment) raises:
    load_trace[p, Blake3](ctx, prover, s.trace)
    load_advice[p, Blake3](ctx, prover, s.idx)
    load_public[p, Blake3](ctx, prover, s.data)
    ctx.synchronize()


def run_grid[p: Params](ctx: DeviceContext, count: Int, mut t: Totals) raises:
    """`count` segments of chain_hashes[p]() hashes on one prover, pipelined: the host builds segment i + 1
    while the device proves segment i, and the upload of i + 1 follows proof i. The timer runs from the
    first segment's start value to the last proof and stops only for verification."""
    if count == 0:
        return
    var t0 = perf_counter_ns()
    var c = Sha256Chain(t.start.copy()).statement[p]().compile[p]()
    var layout = c.layout.copy()
    var prover = Prover[p, Blake3](ctx, c^)
    var w0 = Sha256Chain(t.start.copy())
    _ = prove_prepared[p, Blake3, Sha256Chain](ctx, prover, w0, layout, w0.public_inputs[p]())   # cold: kernel compile
    t.setup_ms += _ms(t0)
    t.arena_bytes = max(t.arena_bytes, prover.layout.bytes)
    t0 = perf_counter_ns()
    var cur = Segment.__init__[p](layout, t.start)
    t.host_ms += _ms(t0)
    var t1 = perf_counter_ns()
    _load[p](ctx, prover, cur)
    t.load_ms += _ms(t1)
    for i in range(count):
        prover.prove_begin(ctx, cur.public)
        if i + 1 < count:
            t1 = perf_counter_ns()
            var nxt = Segment.__init__[p](layout, List[UInt8](cur.public[32:]))
            t.host_ms += _ms(t1)
            var proof = prover.prove_end(ctx)
            t1 = perf_counter_ns()
            _load[p](ctx, prover, nxt)
            t.load_ms += _ms(t1)
            t.prove_ms += _ms(t0)
            _verify[p](proof^, cur, t)
            cur = nxt^
        else:
            var proof = prover.prove_end(ctx)
            t.prove_ms += _ms(t0)
            _verify[p](proof^, cur, t)
        t0 = perf_counter_ns()


def _verify[p: Params](var proof: List[UInt8], s: Segment, mut t: Totals) raises:
    """Outside the timer: check the segment proof and move the chain on to its last digest."""
    t.proof_bytes += len(proof)
    var tv = perf_counter_ns()
    if not verify_workload[p, Blake3, Sha256Chain](proof^, s.w, s.public):
        raise Error("segment proof rejected")
    t.verify_ms += _ms(tv)
    t.start = List[UInt8](s.public[32:])
    t.hashes += chain_hashes[p]()
    t.segments += 1


def run(ctx: DeviceContext, n: Int, segment: Int) raises:
    if n % 125 != 0:
        raise Error("n is a multiple of 125 (the smallest segment)")
    var t = Totals(sha256(List[UInt8]()))
    var r = n
    if segment == 875:
        run_grid[G875](ctx, r // 875, t)
        r %= 875
        run_grid[G375](ctx, r // 375, t)
        r %= 375
    run_grid[G125](ctx, r // 125, t)
    var digest = sha256_chain(sha256(List[UInt8]()), n)
    for i in range(32):
        if t.start[i] != digest[i]:
            raise Error("last digest differs from the host chain")
    var covered = segment == 125
    print('{"system":"caracal7","workload":"sha256_iter","n":', n, ',"segment_hashes":', segment,
          ',"segments":', t.segments, ',"device":"', ctx.name(), '","total_ms":', t.prove_ms,
          ',"ms_per_hash":', t.prove_ms / Float64(n), ',"hashes_per_s":', Float64(n) * 1e3 / t.prove_ms,
          ',"host_ms":', t.host_ms, ',"load_ms":', t.load_ms, ',"setup_ms":', t.setup_ms, ',"proof_bytes":', t.proof_bytes, ',"verify_ms":', t.verify_ms,
          ',"arena_bytes":', t.arena_bytes, ',"digest":"', _hex(t.start), '"',
          ',"interactive_bits":', ('"' + INTERACTIVE_BITS + '"') if covered else "null",
          ',"work_bits":', ('"' + WORK_BITS + '"') if covered else "null",
          ',"security":"', "conditional ledger, not certified" if covered else "not in the ledger: split codewords", '"}',
          sep="")


def main() raises:
    var ctx = DeviceContext()
    var sizes = List[Int]()
    var args = argv()
    for i in range(1, len(args)):
        sizes.append(Int(String(args[i])))
    if len(sizes) == 0:
        sizes = [1000, 10000, 150000]
    for n in sizes:
        run(ctx, n, 875)
        run(ctx, n, 125)
    _ = ctx   # the context must outlive the buffers of this scope: torn down first, NVIDIA deadlocks
