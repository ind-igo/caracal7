"""The Ligerito PCS: encode, commit, open, tail. This surface is what the prover and the verifier call; kernels and loaders stay behind the module paths."""
from pcs.encode import EncLayout, encode, pack_slot, pack_index, split_index, join_index
from pcs.merkle import merkle, query_gather, root_offset, tree_nodes, multiproof_region, check_multiproof, distinct_sorted
from pcs.open import build_queries, open, open_splits, fold, slot_weight, table_len, factor_len, host_table, expand_beta, compact_openings
from pcs.tail import DOM_BYTES, ROUND_ROWS, domain_bytes, tail_encode, points, running0, tail_materialize, tail_round, tail_fold, power_table_len
from pcs.tail import TAIL_F4, e_mul_f4, host_r3, rbar_at, tail_encode_at, fold8_host, quadratic_at
