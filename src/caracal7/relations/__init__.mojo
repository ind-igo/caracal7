"""The Caracal relations: the IR the frontend emits, the Z stage, the residual and quotient, the small grid. This surface is what the prover, the verifier, and the proof layout call."""
from caracal7.relations.ir import ENTRY, NONE, NO_BASIS, POINT, ACC, ACC_W_MAX, END, WIRE, PUBF, CHAL, CHAL_ADD, CHAL_MUL, CHAL_ONE, SAMPLED, KIND_PERM, KIND_LOOKUP, KIND_HORNER, PUB, RES, ZERO, FIX_ONE, FIX_E, entry, shift_points, required_points, standard_chals, chal_count, point_index, point_coord, residual_at, derived_chals, lookup_constant, wire_record, public_factor_record, horner_chain_end, eval_values, eval_line, value_bytes, tile_values
from caracal7.relations.accumulate import AccLayout, accumulate, horner, wiring, derive_chals
from caracal7.relations.residual import lde, residual, quotient, quotient_elems, k_values_to_trace, merge_tables
from caracal7.relations.smallgrid import SmallGridLayout, small_grid_product, small_grid_accumulator, small_grid_wiring, small_grid_end, small_grid_values, interp_cyclic
from caracal7.relations.sort import counting_sort
