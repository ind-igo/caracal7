"""Tail levels (design section 4, spec 9.3): encode, materialize, three rounds, fold."""

from max.gpu.host import DeviceContext

from caracal7.params import Params


def tail_encode[p: Params](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin],
                           y: Int, rows: Int, L: Int, code: Int) raises:
    """Mat(y) columns (8 of them) -> code (s, 8, e): e/4 independent F4 DFTs, no inverse."""
    raise Error("not implemented: tail_encode")


def expected_symbols[p: Params](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin],
                                y_next: Int, positions: Int, count: Int, v: Int) raises:
    raise Error("not implemented: expected_symbols")


def tail_materialize[p: Params](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin],
                                length: Int, challenges: Int, w_tilde: Int) raises:
    """w~ (slot, e) as the sum of the active claim batch's tensor terms."""
    raise Error("not implemented: tail_materialize")


def tail_round[p: Params](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin],
                          w_tilde: Int, y: Int, length: Int, digit: Int, dst: Int) raises:
    """Three evaluations of the round polynomial over one column digit."""
    raise Error("not implemented: tail_round")


def tail_fold[p: Params](ctx: DeviceContext, base: Pointer[UInt8, MutAnyOrigin],
                         y: Int, rows: Int, r_bar: Int, y_next: Int) raises:
    """y_next = Mat(y) r_bar: GEMV with the 8-column matrix."""
    raise Error("not implemented: tail_fold")
