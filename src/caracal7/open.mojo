"""Openings and the level-1 fold (design section 4, spec 9.1)."""

from max.gpu.host import DeviceContext

from caracal7.params import Params


def build_queries[p: Params](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin],
                             points: Int, count: Int, w_z: Int) raises:
    """w_z (P, slot, e) from the twelve tensor factors of 9.1 per point; quotient columns use the
    single Mon(x) L(r) product (9.2), which is the same buffer read with a different slot map."""
    raise Error("not implemented: build_queries")


def open[p: Params](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin],
                    w_z: Int, points: Int, stored: Int, columns: Int, quotient_slots: Bool, dst: Int) raises:
    """out (P, column, e) = <w_z[p], stored(c)> for every point and column of one tree."""
    raise Error("not implemented: open")


def fold[p: Params](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin],
                    beta: Int, stored_w: Int, columns_w: Int, stored_q: Int, columns_q: Int, y: Int) raises:
    """y (slot, e) = sum_c beta_c stored(c) over both trees: one GEMV."""
    raise Error("not implemented: fold")
