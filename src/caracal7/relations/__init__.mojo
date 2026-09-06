"""The Caracal relations: the IR the frontend emits, the Z stage, the residual and quotient, the small grid. This surface is what the prover, the verifier, and the proof layout call."""
from caracal7.relations.ir import ENTRY, NONE, POINT, ACC, ACC_W_MAX, CHALS, KIND_PERM, KIND_LOOKUP, entry, shift_points, point_index, point_coord, residual_at, derived_chals, lookup_constant
from caracal7.relations.accumulate import accumulate, derive_chals
from caracal7.relations.residual import lde, residual, quotient, quotient_elems, k_values_to_trace
from caracal7.relations.smallgrid import small_grid_accumulator, small_grid_values, interp_cyclic
from caracal7.relations.sort import counting_sort
