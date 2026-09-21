"""Stage and byte breakdown of every benchmark workload (docs/profile.md): prove (stage profile), verify
(step profile), and the proof bytes region by region (proof.mojo's layout), on the CLIENT profile. Build
with `-I src -I bench`: the fixtures are the bench files'."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from core.params import Params, CLIENT
from core.hash import Blake3
from prover import Prover, load_trace, load_advice, load_public
from proof import ProofReader, Shape
from relations.statement import advice
from workload import Workload, verify_workload
from workloads.bigint import Big
from workloads.sha256 import Sha256
from workloads.keccak import Keccak
from workloads.poseidon import Poseidon
from workloads.ecdsa import Curve, Ecdsa
from workloads.rsa import RSA
from workloads.sod import SOD
from workloads.dsc import DSC
from workloads.passport import Passport
from workloads.mrz import WINDOW_OFFSET
import bench_rsa
import bench_sod
import bench_dsc


def _message(hex: String) raises -> List[UInt8]:
    var v = Big.from_hex(hex).bytes(hex.byte_length() // 2)
    v.reverse()
    return v^


def sizes[p: Params](proof: List[UInt8], shape: Shape) raises:
    """Walk the proof bytes in the verifier's order without checking anything; print each region's bytes."""
    var r = ProofReader(proof.copy())
    var total = len(proof)
    var pos = 0

    def region(name: String, r: ProofReader, mut pos: Int):
        print("    ", r.pos - pos, " B  ", name)
        pos = r.pos

    _ = r.u32()
    _ = r.prefixed()
    region("header and public inputs", r, pos)
    _ = r.take(Blake3.DIGEST)
    region("W root", r, pos)
    if shape.accumulators() > 0:
        _ = r.take(Blake3.DIGEST)
        if shape.products() > 0:
            _ = r.field_bytes(shape.products() * p.h2() * p.e)
        region("Z root and Z2", r, pos)
    _ = r.take(Blake3.DIGEST)
    if shape.accumulators() > 0:
        _ = r.field_bytes(2 * p.h2() * p.e)
    region("Q root and Q3", r, pos)
    _ = r.field_bytes(shape.points * shape.columns() * p.e)
    region("openings (" + String(shape.points) + " points x " + String(shape.columns()) + " columns)", r, pos)
    for i in range(len(shape.tail)):
        _ = r.take(Blake3.DIGEST)
        if p.grind_bits > 0:
            _ = r.take(8)
        if i == 0:
            _ = r.prefixed()
            if shape.columns_z > 0:
                _ = r.prefixed()
            _ = r.prefixed()
            region("level 1 rows and multiproofs (" + String(p.queries()) + " queries x " + String(4 * p.n_cw() * shape.columns()) + " B rows)", r, pos)
        else:
            _ = r.prefixed()
            region("level " + String(i + 1) + " rows and multiproof (" + String(shape.tail[i - 1].queries) + " queries x " + String(8 * p.e * shape.tail[i - 1].codewords) + " B rows)", r, pos)
        _ = r.field_bytes(9 * p.e)
        region("level " + String(i + 2) + " root and sumcheck rounds", r, pos)
    _ = r.field_bytes(shape.clear_length * p.e)
    region("clear vector (" + String(shape.clear_length) + " E)", r, pos)
    if p.grind_bits > 0:
        _ = r.take(8)
    _ = r.prefixed()
    var last = len(shape.tail)
    if last == 0:
        region("level 1 rows and multiproofs (" + String(p.queries()) + " queries)", r, pos)
    else:
        region("level " + String(last + 1) + " rows and multiproof (" + String(shape.tail[last - 1].queries) + " queries)", r, pos)
    r.done()
    print("    ", total, " B  total")


def breakdown[p: Params, W: Workload](name: String, ctx: DeviceContext, w: W) raises:
    var c = w.statement[p]().compile[p]()
    var trace = w.trace[p](c.layout)
    var inputs = w.public_inputs[p]()
    var data = W.public_data[p](c.layout, inputs)
    var idx = advice[p](c.layout, trace)
    var prover = Prover[p, Blake3](ctx, c^)
    load_trace[p, Blake3](ctx, prover, trace)
    load_advice[p, Blake3](ctx, prover, idx)
    load_public[p, Blake3](ctx, prover, data)
    ctx.synchronize()
    _ = prover.prove(ctx, inputs)
    _ = prover.prove(ctx, inputs)
    var t0 = perf_counter_ns()
    var proof = prover.prove(ctx, inputs)
    var warm = (perf_counter_ns() - t0) // 1000000
    _ = prover.prove(ctx, inputs, profile=True)
    print("\n## ", name, "  grid ", p.h1(), " x ", p.h2(), "  columns W/Z/Q ", prover.shape.columns_w, "/", prover.shape.columns_z, "/", prover.shape.columns_q,
          "  points ", prover.shape.points, "  proof ", len(proof), " B  warm prove ", warm, " ms")
    print("  prove stages:")
    for i in range(len(prover.profile_names)):
        if prover.profile_ms[i] > 0:
            print("    ", prover.profile_ms[i], " ms  ", prover.profile_names[i])
    print("  verify steps:")
    t0 = perf_counter_ns()
    var ok = verify_workload[p, Blake3, W](proof.copy(), w, inputs, profile=True)
    print("    verify total ", (perf_counter_ns() - t0) // 1000000, " ms ", ok)
    print("  proof bytes:")
    sizes[p](proof, prover.shape)


def main() raises:
    var ctx = DeviceContext()
    breakdown[CLIENT.grid(32, 2113)]("sha256 2048 B", ctx, Sha256(List[UInt8](length=2048, fill=7)))
    breakdown[CLIENT.grid(64, 384)]("keccak 2048 B", ctx, Keccak(List[UInt8](length=2048, fill=7)))
    breakdown[CLIENT.grid(64, 720)]("poseidon 16", ctx, Poseidon(List[Int](length=16, fill=3)))
    var c0 = Curve()
    var d = Big.from_hex("1e99423a4ed27608a15a2616a2b0e9e52ced330ac530edcc32c8ffc6a526aedd")
    var k = Big.from_hex("a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90")
    var e = Big.from_hex("4b688df40bcedbe641ddb16ff0a1842d9c67ea1c3bf63f3e0471baa664531d1a")
    var q = c0.mul(c0.g, d)
    var rr = c0.mul(c0.g, k).x.mod(c0.n)
    var sg = k.inv_mod(c0.n).mulmod(e + rr.mulmod(d, c0.n), c0.n)
    breakdown[CLIENT.grid(144, 576)]("ecdsa", ctx, Ecdsa(rr^, sg^, e^, q^))
    breakdown[CLIENT.grid(144, 2016)]("rsa-2048", ctx, RSA(8, 17, Big.from_hex(bench_rsa.S_HEX), Big.from_hex(bench_rsa.N_HEX), Big.from_hex(bench_rsa.M_HEX)))
    var msgs: List[List[UInt8]] = [_message(bench_sod.DG1_HEX), _message(bench_sod.ECONTENT_HEX), _message(bench_sod.ATTRS_HEX)]
    var lengths: List[Int] = [len(msgs[0]), len(msgs[1]), len(msgs[2])]
    breakdown[CLIENT.grid(144, 2688)]("sod", ctx, SOD(8, 17, Big.from_hex(bench_sod.S_HEX), Big.from_hex(bench_sod.N_HEX), Big.from_hex(bench_sod.M_HEX),
                                                       lengths.copy(), [bench_sod.EMBED_1, bench_sod.EMBED_2], WINDOW_OFFSET, _message(bench_sod.SCOPE_HEX), msgs.copy(), _message(bench_sod.R_HEX)))
    var tbs = _message(bench_dsc.TBS_HEX)
    breakdown[CLIENT.grid(144, 2688)]("dsc", ctx, DSC(8, 17, Big.from_hex(bench_dsc.S_DSC_HEX), Big.from_hex(bench_dsc.N_CSCA_HEX), Big.from_hex(bench_dsc.M_DSC_HEX),
                                                       len(tbs), bench_dsc.N_OFFSET, tbs.copy(), Big.from_hex(bench_dsc.N_DSC_HEX), _message(bench_dsc.R_HEX)))
    breakdown[CLIENT.grid(144, 4032)]("passport", ctx, Passport(8, 17, Big.from_hex(bench_sod.S_HEX), Big.from_hex(bench_sod.N_HEX), Big.from_hex(bench_sod.M_HEX),
                                                                 Big.from_hex(bench_dsc.S_DSC_HEX), Big.from_hex(bench_dsc.N_CSCA_HEX), Big.from_hex(bench_dsc.M_DSC_HEX),
                                                                 lengths^, [bench_sod.EMBED_1, bench_sod.EMBED_2], WINDOW_OFFSET, len(tbs), bench_dsc.N_OFFSET,
                                                                 _message(bench_sod.SCOPE_HEX), msgs^, tbs^))
    _ = ctx
