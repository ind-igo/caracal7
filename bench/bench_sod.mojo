"""One passport SOD (workloads/sod.mojo): SHA-256 of DG1 (93 bytes), of the LDS security object (100 bytes)
and of the signed attributes (73 bytes), and the RSA-2048 verify (e = 65537), 1987 chains on the 144 x 2016
grid. Fixture from OpenSSL (genrsa 2048; dgst -sha256 -sign on the signed attributes). Warm prove, proof
size, verify."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from caracal7.core.params import Params, CLIENT
from caracal7.core.hash import Blake3
from caracal7.workloads.bigint import Big
from caracal7.workloads.sod import SOD
from caracal7.workload import Session

comptime N_HEX = "a027dc760a245427c068f89fdee6c3d8606c9997dc1e2502cc4993b043f2c52ac65210ed418cbbc241fe060683f207dbcc2c4f80658d7c4d8ce3276d4d1946f40a1c44f9de0ec8001992b081282ecf4dc3e1661080c8e2a6100a28b6492630c5e484edca98b1e700b4acec99d6cc119797e4d9273950498e9b260c6a965e1526dadedebf30c0b06059f8d5b42422ef7e12beddd2f5082ec497913402ce9c788f6ec71df5335288a77377827c9c10ef190a494c675202bb649be63796d18bfe9a160346eb97491401ad7c2b2eb7fe5963595a71eb5d3d38cb3253730f02d8fc4bb42d1a41d01fb691d5a0b3d90ec1b277c9753fbef70340b3a514b958b6495c77"
comptime S_HEX = "93adcfb919cf93ed2ff130917569eb592c72f35f0f074e16ce8c21a052b6fe7dc732f50ab7d818f9ce395745c5f46cc6e76595dbd0668e4e10a88bf4d049b5cc0329bb8190ce31ea464c82fd535c002fde8c806e006c748facc6a32e22100ec647aa6f4d8bdbe25f7ac7eb88c4d469ae0cb728e614cb22ee6d4ed80c0642f9e9389b3f9eadf64df1583453182a29bb43b70b395f6e6b38c82712cbefac28da49d47f0e8e004ab014d8502c1fc51823098b3dafc273a674d47c7d1a6072f4fb9436e0ae80df5f5b846d16d32ff48171e1cd379cf4672451be7eecc2bc04e80df9ebfe076a8059c580580feb999bbc160ff532a8d7544c1dcc9568a28ce644eabf"
comptime M_HEX = "0001ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff003031300d0609608648016503040201050004209adac772ef5475b6b2624243c606915e185503a5858d3738f3040ca325ab554b"
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
    var w = SOD(8, 17, Big.from_hex(S_HEX), Big.from_hex(N_HEX), Big.from_hex(M_HEX), lengths^, [EMBED_1, EMBED_2], msgs^)
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
    run[CLIENT.grid(144, 2016)]()
