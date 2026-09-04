"""Residual grid stages (design section 4, spec 8 and 10.2): coset LDE of committed columns to G,
the fused residual pass over the family tables, and the quotient to E coefficients in plain slots."""

from max.gpu.host import DeviceContext

from caracal7.params import Params


def lde[p: Params](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin],
                   coeff: Int, columns: Int, dst: Int) raises:
    """coeff (column, k2, k1, 2) -> out (column, G2, G1, 2): twist by g^i, forward DFT per axis."""
    raise Error("not implemented: lde")


def residual[p: Params](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin],
                        lde_buf: Int, columns: Int, tables: Int, challenges: Int, dst: Int) raises:
    """(G2, G1, e): the batched residual over the linear and quadratic tables (statement-layer 5).
    Milestone 1 runs synthetic families."""
    raise Error("not implemented: residual")


def quotient[p: Params](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin],
                        residual_buf: Int, quotient_buf: Int, stored_q: Int) raises:
    """residual -> A, B, Q2 in evaluation form -> inverse 2D DFT -> 3 e coordinate columns in the
    plain (x1, x2, r) slots of spec 9.2, written straight into the quotient tree's `stored`."""
    raise Error("not implemented: quotient")
