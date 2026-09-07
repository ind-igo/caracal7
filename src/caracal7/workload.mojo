"""A workload is what a frontend provides: the statement, the trace, and the derivation of the public
data from the public inputs. `prove_workload` and `verify_workload` are the one path from a workload to a
proof and back, so the sequence (compile, trace, advice, public data, prove; compile, public data, verify)
is written once and every frontend runs it. Tests that need a wrong trace or a wrong advice call the
pieces directly.

`public_data` is static and runs on both sides: the verifier never sees the trace, so everything it reads
past the proof (the public blocks, then the restriction lines, in that order) must follow from
`public_inputs`, which the prefix hashes."""

from max.gpu.host import DeviceContext

from caracal7.core.params import Params
from caracal7.core.hash import Hash
from caracal7.prover import Prover, load_trace, load_advice, load_public
from caracal7.verifier import verify
from caracal7.relations import block_bytes
from caracal7.relations.statement import Statement, Layout, advice


trait Workload:
    def statement(self) raises -> Statement:
        """The statement, grid-independent (compile binds it to Params)."""
        ...

    def trace[p: Params](self, layout: Layout) raises -> List[UInt8]:
        """The W trace, columns_w x N column major, live rows filled and idle rows padded (`pad_trace`)."""
        ...

    def public_inputs[p: Params](self) raises -> List[UInt8]:
        """The bytes the verifier is given; the prefix hashes them."""
        ...

    @staticmethod
    def public_data[p: Params](layout: Layout, public_inputs: List[UInt8]) raises -> List[UInt8]:
        """Every public block (`public_block`), then every restriction line (`restriction_line`), from the
        public inputs alone."""
        ...


def prove_workload[p: Params, H: Hash, W: Workload](ctx: DeviceContext, w: W) raises -> List[UInt8]:
    var c = w.statement().compile[p]()
    var trace = w.trace[p](c.layout)
    var idx = advice[p](c.layout, trace)
    var inputs = w.public_inputs[p]()
    var data = W.public_data[p](c.layout, inputs)
    var n_blocks = block_bytes(c.layout.publics, p.h1())
    var blocks = List[UInt8](capacity=n_blocks)
    for i in range(n_blocks):
        blocks.append(data[i])
    var families = c.families.copy()
    var prover = Prover[p, H](ctx, c^.take_shape(), families^)
    load_trace[p, H](ctx, prover, trace)
    load_advice[p, H](ctx, prover, idx)
    load_public[p, H](ctx, prover, blocks)
    return prover.prove(ctx, inputs)


def verify_workload[p: Params, H: Hash, W: Workload](var proof: List[UInt8], w: W, public_inputs: List[UInt8]) raises -> Bool:
    var c = w.statement().compile[p]()
    var data = W.public_data[p](c.layout, public_inputs)
    var families = c.families.copy()
    return verify[p, H](proof^, c.shape, public_inputs, families, data)
