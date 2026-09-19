"""One RSA-2048 verify (e = 65537: 17 modmuls, 1745 chains) on the 144 x 2016 grid, workloads/rsa.mojo. The
key and signature come from OpenSSL (genrsa 2048; dgst -sha256 -sign; m is the PKCS#1 v1.5 encoding of the
digest, checked as pow(s, e, n) in Python). Warm prove, proof size, verify."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from core.params import Params, CLIENT
from core.hash import Blake3
from prover import Prover, load_trace, load_advice, load_public
from relations.statement import advice
from workloads.bigint import Big
from workloads.rsa import RSA
from workload import verify_workload

comptime N_HEX = "9f68f25ceaa7e9a624a218378c25f86a72925a694a9eeceefd83f0a5e6ba16742c236ddf515351f9b9342b14ee1f8a3ab41f3f435f1c6e51616cde968838c357570251f3641c5d9da36eef2571ed944406b37a83e536c191cdccf532482209216eac2ebf6a238b21c7e31bda1e69371811465fee8cd781ccdf638370e8d2ed44bb3404cffeff506935aec1e0492e89d41dc91a4c527f3a0ed9aff60dbfb96122dbe8135425278efa6f505474f64cdd92b48541c454b818a044b1485dd64fffc949ddcb0b176b82d624833eab0a355a01d0c02585497d4a7ae9bb33bdcd8388ea626e180f1c1d0fd2c9136bbe73dd2c6611c0eb66802ce24fd97f72e0d353e13d"
comptime S_HEX = "658efaa66cfaad263866896715cd0533d08fe5cc5429ef998ed23edec4a2106e50cdc80dcd3c4ebcc5d7325ae0afe676295b9104dc9d682ecdde8cb4002fc3aae082b10a7370856d25e15202d1f034780cacedec8445e2b3978673989c1a8bd62fe6abc991891b8a7f3c790196c311005ea0c2946bebdd3df82352953e7bd03e602c48e4b8285a6bb9cf51f43158acc68dd9d99b7d22df2f79003bb256dbcb14fd59b44efc36b0aa1e5e080a65ab9a6f3c2a1e90aedd18bd2c077baa90d0e04478eb38c5bfa8862ffc8f85f4d533c9472a35e52b277e8161bdfc55bfca75c4a3b54828e3b60caaa9f2940ebdbbcdf6306be31ddf3854d565efd5a33dfbadf1ec"
comptime M_HEX = "1ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff003031300d060960864801650304020105000420350b11a18107391fa11e65935917929e189ed0e0d4dd1589a9c60b5dd4f6661e"


def run[p: Params]() raises:
    var ctx = DeviceContext()
    var w = RSA(8, 17, Big.from_hex(S_HEX), Big.from_hex(N_HEX), Big.from_hex(M_HEX))
    var t0 = perf_counter_ns()
    var c = w.statement[p]().compile[p]()
    var t_compile = (perf_counter_ns() - t0) // 1000000
    t0 = perf_counter_ns()
    var trace = w.trace[p](c.layout)
    var t_trace = (perf_counter_ns() - t0) // 1000000
    var inp = w.public_inputs[p]()
    var data = RSA.public_data[p](c.layout, inp)
    var idx = advice[p](c.layout, trace)
    var prover = Prover[p, Blake3](ctx, c^)
    load_trace[p, Blake3](ctx, prover, trace)
    load_advice[p, Blake3](ctx, prover, idx)
    load_public[p, Blake3](ctx, prover, data)
    ctx.synchronize()
    _ = prover.prove(ctx, inp)
    _ = prover.prove(ctx, inp)
    t0 = perf_counter_ns()
    var proof = prover.prove(ctx, inp)
    var warm = (perf_counter_ns() - t0) // 1000000
    t0 = perf_counter_ns()
    var ok = verify_workload[p, Blake3, RSA](proof.copy(), w, inp, profile=True)
    var vms = (perf_counter_ns() - t0) // 1000000
    print("rsa-2048 verify  grid ", p, "  compile ", t_compile, " ms  trace ", t_trace, " ms  proof ", len(proof),
          " B  warm prove ", warm, " ms  verify ", vms, " ms ", ok)
    _ = ctx


def main() raises:
    run[CLIENT.grid(144, 2016)]()
