"""One DSC certificate check (workloads/dsc.mojo): SHA-256 of the certificate body (650 bytes, the DSC key at
byte 223), the commitment to that key (288 bytes) and the RSA-2048 verify (e = 65537) of the CSCA's signature,
s witness, 2003 chains on the 144 x 2688 grid. Fixture from OpenSSL (a CSCA and a DSC certificate with the
ICAO extensions). Warm prove, proof size, verify. With bench_sod this is one passport."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from core.params import Params, CLIENT
from core.hash import Blake3
from workloads.bigint import Big
from workloads.dsc import DSC
from workload import Session

comptime N_CSCA_HEX = "e033cfbd808b0902536d51580f7e53665c1bf9da78076eee38f95709579ed0a1985729e35da0b0ccb18c936e045c51e21087ed7031c3f94e9cc8b7e400d95cb7cf46bf553fbe439c1cbc7220b2a2afcfdafb13cc8314029da6ff30e82d09451e1857c6a88a8002569e6f0e7405a58dff739f76ea2c12098afa41e26cddd318fcbe1eceed0a4a8089d0c16b4db58b88a20c28a8737a44e3e82fcbe97f5fe0d888032af37b36dfca514d3c1d70b4dc68ceafa9de77fa7a89378f5dde7f8f6536b5f8c919fc09ef65e36a88532f2614e948980f503d968d329162a8d575e7e2218c63408313275562fdcd4ff578fcd267ea2146904c6b52191f5f921bf3dabb3735"
comptime N_DSC_HEX = "cfee6558e7a464a2c8c92ca9f1fc7bd9640b3ccd2e854e1b322851376f62a7658c2dab943f03c6220e5b87adbdc2f27b58651956e3c7c1685f14ad1a36d4c33ad2e685d0834d884e54ab4a396612e4f3c77599120c02e66e83278bdc982da4bfab5537fbc3cbd87dc5428a429ebaa2a0f0f00881c9725fdab6c5298467c3d7ac1045c63faa7a0704bf0712afb67a204e8bb0c56cc6c84886375ed788ff8579d85c107de14380443a69604a8abb7a10c20c076c8dae998f98c2628afa9815757df6ab902abb89c4fb35eef33f5cc19e123005d3674d3b6fbd515dc652ed130a56a03389af811ef63be368f9f0bcb5379f8e9307f0d37e9ea87d9322976de13bb1"
comptime S_DSC_HEX = "2fe94516161933c89e541f1f9e5831071e5ba852efd72b3f01b972c7f5cc4b8d7f5d24f6dc8d3849c6d1227d2d1c2813b1b68b3e788c9ed8440f11534e057bbf4945bc47641675b0a5e3a3038fcf807fa08c4dc99a0cbd362ef5228eae4ba1bd461edc25312828c18f96f00f8f7bdb2e52695b9eedfbb3defa252d5306673997e29372b8380fccff0592aafea3e415c56e6458d485d8719e15ef750cbeaf75fdea2c5ac4520b496f8eb85b192b386fe3ac2ae0b48a634a23f2edaac1c859a1ad8408cd0766002945b6ba75042a7ae7959db1487d8224d03cbfffed2b2c14a1f70065edb8dc173a106e59fb718364a799babe5c2f457530da0497f7ebc55db18d"
comptime M_DSC_HEX = "0001ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff003031300d060960864801650304020105000420d4854880926747257deddd77e4ee7886a68fc3d3d72dbb9c2908b144607adacd"
comptime TBS_HEX = "30820286a0030201020214093ca49c1d905896cfb40999a80420b349892cb3300d06092a864886f70d01010b05003035310b300906035504061302555431143012060355040a0c0b55746f70696120435343413110300e06035504030c07435343412d5554301e170d3236303931393036303334315a170d3239303631353036303334315a3037310b300906035504061302555431133011060355040a0c0a55746f706961204d46413113301106035504030c0a4453432d55542d30303130820122300d06092a864886f70d01010105000382010f003082010a0282010100cfee6558e7a464a2c8c92ca9f1fc7bd9640b3ccd2e854e1b322851376f62a7658c2dab943f03c6220e5b87adbdc2f27b58651956e3c7c1685f14ad1a36d4c33ad2e685d0834d884e54ab4a396612e4f3c77599120c02e66e83278bdc982da4bfab5537fbc3cbd87dc5428a429ebaa2a0f0f00881c9725fdab6c5298467c3d7ac1045c63faa7a0704bf0712afb67a204e8bb0c56cc6c84886375ed788ff8579d85c107de14380443a69604a8abb7a10c20c076c8dae998f98c2628afa9815757df6ab902abb89c4fb35eef33f5cc19e123005d3674d3b6fbd515dc652ed130a56a03389af811ef63be368f9f0bcb5379f8e9307f0d37e9ea87d9322976de13bb10203010001a381a33081a0301d0603551d0e041604145dbc7b2d638e951151afe3a66d04a93e5c615361301f0603551d23041830168014abe73ec74f97e34126cfb193bdb74d9f136fb5ad300e0603551d0f0101ff04040302078030370603551d1f0430302e302ca02aa0288626687474703a2f2f706b692e75746f7069612e6578616d706c652f637363612f63726c2e63726c3015060767810801010602040a30080201003103130150"
comptime N_OFFSET = 223
comptime R_HEX = "ea80c1fe473344b9e0425cc4951209caec70c930eb124edca6147aa50516b2c4"


def _message(hex: String) raises -> List[UInt8]:
    var v = Big.from_hex(hex).bytes(hex.byte_length() // 2)
    v.reverse()
    return v^


def run[p: Params]() raises:
    var ctx = DeviceContext()
    var tbs = _message(TBS_HEX)
    var w = DSC(8, 17, Big.from_hex(S_DSC_HEX), Big.from_hex(N_CSCA_HEX), Big.from_hex(M_DSC_HEX), len(tbs), N_OFFSET, tbs^,
                Big.from_hex(N_DSC_HEX), _message(R_HEX))
    var t0 = perf_counter_ns()
    var session = Session[p, Blake3, DSC](ctx, w.copy())
    var t_prepare = (perf_counter_ns() - t0) // 1000000
    _ = session.prove()
    _ = session.prove()
    t0 = perf_counter_ns()
    var proof = session.prove()
    var warm = (perf_counter_ns() - t0) // 1000000
    t0 = perf_counter_ns()
    var ok = session.verify(proof.copy())
    var vms = (perf_counter_ns() - t0) // 1000000
    print("dsc  grid ", p, "  prepare ", t_prepare, " ms  proof ", len(proof), " B  warm prove ", warm, " ms  verify ", vms, " ms ", ok)
    _ = ctx


def main() raises:
    run[CLIENT.grid(144, 2688)]()
