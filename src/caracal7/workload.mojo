"""A workload is what a frontend provides: the statement, the trace, and the derivation of the public
data from the public inputs. `prove_workload` and `verify_workload` are the one path from a workload to a
proof and back, so the sequence (compile, trace, advice, public data, prove; compile, public data, verify)
is written once and every frontend runs it. Tests that need a wrong trace or a wrong advice call the
pieces directly.

`public_data` is static and runs on both sides: the verifier never sees the trace, so everything it reads
past the proof (the public column values, then the restriction lines, in that order) must follow from
`public_inputs`, which the prefix hashes."""

from max.gpu.host import DeviceContext

from caracal7.core.params import Params
from caracal7.core.hash import Hash
from caracal7.prover import Prover, load_trace, load_advice, load_public
from caracal7.verifier import verify
from caracal7.relations import value_bytes, entry, ENTRY, NONE, NO_BASIS
from caracal7.relations.statement import Statement, Layout, Compiled, advice


trait Workload:
    def statement[p: Params](self) raises -> Statement:
        """The statement; the grid is for periodic public columns and axis-2 shifts (compile binds the rest)."""
        ...

    def trace[p: Params](self, layout: Layout) raises -> List[UInt8]:
        """The W trace, columns_w x N column major, live rows filled and idle rows padded (`pad_trace`)."""
        ...

    def public_inputs[p: Params](self) raises -> List[UInt8]:
        """The bytes the verifier is given; the prefix hashes them."""
        ...

    @staticmethod
    def public_data[p: Params](layout: Layout, public_inputs: List[UInt8]) raises -> List[UInt8]:
        """One period of values per public column ((h2 / m, h1) F bytes), then every restriction line (`restriction_line`), from the
        public inputs alone."""
        ...


def prove_workload[p: Params, H: Hash, W: Workload](ctx: DeviceContext, w: W) raises -> List[UInt8]:
    var c = w.statement[p]().compile[p]()
    var trace = w.trace[p](c.layout)
    var idx = advice[p](c.layout, trace)
    var inputs = w.public_inputs[p]()
    var data = W.public_data[p](c.layout, inputs)
    var n_blocks = value_bytes(c.layout.publics, p.h1(), p.h2())
    var blocks = List[UInt8](capacity=n_blocks)
    for i in range(n_blocks):
        blocks.append(data[i])
    var prover = Prover[p, H](ctx, c^)
    load_trace[p, H](ctx, prover, trace)
    load_advice[p, H](ctx, prover, idx)
    load_public[p, H](ctx, prover, blocks)
    return prover.prove(ctx, inputs)


def verify_workload[p: Params, H: Hash, W: Workload](var proof: List[UInt8], w: W, public_inputs: List[UInt8], profile: Bool = False) raises -> Bool:
    var c = w.statement[p]().compile[p]()
    var data = W.public_data[p](c.layout, public_inputs)
    return verify[p, H](proof^, c, public_inputs, data, profile)


def check_families[p: Params](c: Compiled, trace: List[UInt8], pubs: List[List[UInt8]]) raises:
    """Host check of a compiled statement without accumulators: every entry summed per family is zero on every
    row its gate admits, public reads included. `pubs` holds every public column's values on the whole grid,
    (chain, row) order, in declaration order (a periodic column tiled)."""
    comptime N = p.N()
    comptime h1 = p.h1()
    comptime h2 = p.h2()
    var w = c.layout.columns_w()
    var count = len(c.families) // ENTRY
    var families = 0
    for k in range(count):
        families = max(families, entry(c.families, k).family + 1)
    var sums = List[Int](length=families * N, fill=0)
    for k in range(count):
        var en = entry(c.families, k)
        if en.chal != 0 or en.basis != NO_BASIS:
            raise Error("check_families takes families without challenges or basis factors")
        for x2 in range(h2):
            for x1 in range(h1):
                if (en.mult == 1 and x1 == h1 - 1) or (en.mult == 2 and x2 == h2 - 1):
                    continue
                var v = en.coef * _at[p](trace, pubs, w, en.col_a, en.dj1_a // 2, en.dj2_a // 2, x1, x2)
                if en.col_b != NONE:
                    v *= _at[p](trace, pubs, w, en.col_b, en.dj1_b // 2, en.dj2_b // 2, x1, x2)
                sums[en.family * N + x2 * h1 + x1] = (sums[en.family * N + x2 * h1 + x1] + v) % 127
    for i in range(families * N):
        if sums[i] != 0:
            raise Error("family " + String(i // N) + " fails at row " + String(i % N))


def _at[p: Params](trace: List[UInt8], pubs: List[List[UInt8]], w: Int, col: Int, k1: Int, k2: Int, x1: Int, x2: Int) -> Int:
    """A read at (x1 + k1, x2 + k2); public columns follow the W columns (no Z columns)."""
    comptime N = p.N()
    comptime h1 = p.h1()
    var row = ((x2 + k2) % p.h2()) * h1 + (x1 + k1) % h1
    if col < w:
        return Int(trace[col * N + row])
    return Int(pubs[col - w][row])
