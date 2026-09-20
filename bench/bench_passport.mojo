"""One passport in one proof (workloads/passport.mojo): the SOD of bench_sod and the DSC certificate check of
bench_dsc on one grid, the DSC key wired from the certificate body to the SOD's RSA verify, no commitment
groups. Two RSA-2048 verifies (e = 65537) with both signatures, the body and the DSC key witness. Warm prove,
proof size, verify. Against bench_sod plus bench_dsc: one proof on 144 x 4032 for two on 144 x 2688."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from core.params import Params, CLIENT
from core.hash import Blake3
from workloads.bigint import Big
from workloads.passport import Passport, passport_chains
from workloads.mrz import WINDOW_OFFSET
from workload import Session
from bench_sod import N_HEX, S_HEX, M_HEX, DG1_HEX, ECONTENT_HEX, ATTRS_HEX, EMBED_1, EMBED_2, SCOPE_HEX
from bench_dsc import N_CSCA_HEX, S_DSC_HEX, M_DSC_HEX, TBS_HEX, N_OFFSET


def _message(hex: String) raises -> List[UInt8]:
    var v = Big.from_hex(hex).bytes(hex.byte_length() // 2)
    v.reverse()
    return v^


def run[p: Params]() raises:
    var ctx = DeviceContext()
    var msgs: List[List[UInt8]] = [_message(DG1_HEX), _message(ECONTENT_HEX), _message(ATTRS_HEX)]
    var lengths: List[Int] = [len(msgs[0]), len(msgs[1]), len(msgs[2])]
    var tbs = _message(TBS_HEX)
    print("chains ", passport_chains(lengths, len(tbs), 8, 17))
    var w = Passport(8, 17, Big.from_hex(S_HEX), Big.from_hex(N_HEX), Big.from_hex(M_HEX), Big.from_hex(S_DSC_HEX), Big.from_hex(N_CSCA_HEX),
                     Big.from_hex(M_DSC_HEX), lengths^, [EMBED_1, EMBED_2], WINDOW_OFFSET, len(tbs), N_OFFSET, _message(SCOPE_HEX), msgs^, tbs^)
    var t0 = perf_counter_ns()
    var session = Session[p, Blake3, Passport](ctx, w.copy())
    var t_prepare = (perf_counter_ns() - t0) // 1000000
    _ = session.prove()
    _ = session.prove()
    t0 = perf_counter_ns()
    var proof = session.prove()
    var warm = (perf_counter_ns() - t0) // 1000000
    t0 = perf_counter_ns()
    var ok = session.verify(proof.copy())
    var vms = (perf_counter_ns() - t0) // 1000000
    print("passport  grid ", p, "  prepare ", t_prepare, " ms  proof ", len(proof), " B  warm prove ", warm, " ms  verify ", vms, " ms ", ok)
    _ = ctx


def main() raises:
    run[CLIENT.grid(144, 4032)]()
