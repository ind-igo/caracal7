"""One passport SOD (workloads/sod.mojo): SHA-256 of DG1 (93 bytes), of the LDS security object (100 bytes)
and of the signed attributes (73 bytes), the commitment to the DSC key (288 bytes) and the RSA-2048 verify
(e = 65537) with s and n witness, 2068 chains on the 144 x 2688 grid. Fixture from OpenSSL (the DSC key of
bench_dsc; dgst -sha256 -sign on the signed attributes). Warm prove, proof size, verify."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from core.params import Params, CLIENT
from core.hash import Blake3
from workloads.bigint import Big
from workloads.sod import SOD
from workload import Session

comptime N_HEX = "cfee6558e7a464a2c8c92ca9f1fc7bd9640b3ccd2e854e1b322851376f62a7658c2dab943f03c6220e5b87adbdc2f27b58651956e3c7c1685f14ad1a36d4c33ad2e685d0834d884e54ab4a396612e4f3c77599120c02e66e83278bdc982da4bfab5537fbc3cbd87dc5428a429ebaa2a0f0f00881c9725fdab6c5298467c3d7ac1045c63faa7a0704bf0712afb67a204e8bb0c56cc6c84886375ed788ff8579d85c107de14380443a69604a8abb7a10c20c076c8dae998f98c2628afa9815757df6ab902abb89c4fb35eef33f5cc19e123005d3674d3b6fbd515dc652ed130a56a03389af811ef63be368f9f0bcb5379f8e9307f0d37e9ea87d9322976de13bb1"
comptime S_HEX = "213c500ab6561b13def89d97ac3357893ac71147066787a6a4305abd3e2e3eb97579d49a1e3f35a0c27e9982a7578dc3a09acef4a7cd582fa4544b01ce4f04bf3b8e968a92184b47dd42d41112d82d75f77e7ec616e47d0accc033fd17d6e78c681751a9383bd54e93908a8fa82ddcd0a1c19340f181387cf3384b1c91ca4146057c072e36f2800c3af314d908f0a5efa45a9f2efa11ef83b165cf71dbd178fe2f7fd69f14351405b29189347ef8e2df3d3ba256cb14634dd4773fa6ef60222e4df84a15686f2b566ce7dc705b41708903cf18756fb361964ca362bc395ca0183e1327e834c143e5627cb3802b4992224d28c574bcfb5de07b8c99d4664d5e49"
comptime M_HEX = "0001ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff003031300d0609608648016503040201050004209adac772ef5475b6b2624243c606915e185503a5858d3738f3040ca325ab554b"
comptime R_HEX = "ea80c1fe473344b9e0425cc4951209caec70c930eb124edca6147aa50516b2c4"
comptime DG1_HEX = "615b5f1f58503c55544f4552494b53534f4e3c3c414e4e413c4d415249413c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c4c38393839303243333655544f3734303831323246313230343135395a45313834323236423c3c3c3c3c3130"
comptime ECONTENT_HEX = "3062020100300d06096086480165030402010500304e30250201010420432bc07d1c637793f4d77e0b756865f7aec3756f98d6ec6eb767eda371904651302502010204208602578350f117185d85c3255c672a528b88adda60e7b2d4c3cbb6e77b50f40c"
comptime ATTRS_HEX = "3147301506092a864886f70d010903310806066781080101302f06092a864886f70d0109043122042017afa9d3156cc5c65b2c7923340bfe3816681f509cdba0c32ca4c05736a61d73"
comptime EMBED_1 = 29
comptime EMBED_2 = 41


def _message(hex: String) raises -> List[UInt8]:
    var v = Big.from_hex(hex).bytes(hex.byte_length() // 2)
    v.reverse()
    return v^


def run[p: Params]() raises:
    var ctx = DeviceContext()
    var msgs: List[List[UInt8]] = [_message(DG1_HEX), _message(ECONTENT_HEX), _message(ATTRS_HEX)]
    var lengths: List[Int] = [len(msgs[0]), len(msgs[1]), len(msgs[2])]
    var w = SOD(8, 17, Big.from_hex(S_HEX), Big.from_hex(N_HEX), Big.from_hex(M_HEX), lengths^, [EMBED_1, EMBED_2], msgs^, _message(R_HEX))
    var t0 = perf_counter_ns()
    var session = Session[p, Blake3, SOD](ctx, w.copy())
    var t_prepare = (perf_counter_ns() - t0) // 1000000
    _ = session.prove()
    _ = session.prove()
    t0 = perf_counter_ns()
    var proof = session.prove()
    var warm = (perf_counter_ns() - t0) // 1000000
    t0 = perf_counter_ns()
    var ok = session.verify(proof.copy())
    var vms = (perf_counter_ns() - t0) // 1000000
    print("sod  grid ", p, "  prepare ", t_prepare, " ms  proof ", len(proof), " B  warm prove ", warm, " ms  verify ", vms, " ms ", ok)
    _ = ctx


def main() raises:
    run[CLIENT.grid(144, 2688)]()
