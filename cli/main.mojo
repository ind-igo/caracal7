"""The csp-benchmarks entry point (docs/csp.md): one binary, one grid per target and input size.

    caracal7 prove  <target> <size> <proof-path> <input...>
    caracal7 verify <target> <size> <proof-path> <input...>

Inputs: sha256 and keccak take the message as hex; poseidon takes Mersenne-31 elements as decimal
strings; ecdsa takes the digest, x_Q, y_Q and the signature r||s as big-endian hex (the utils
generator's lines). Prove writes the proof bytes and prints the timings; verify exits 1 on a bad proof."""
from max.gpu.host import DeviceContext
from std.sys import argv
from std.sys import exit
from std.time import perf_counter_ns

from caracal7.core.params import CLIENT, Params
from caracal7.core.hash import Blake3
from caracal7.workload import Workload, prove_workload, verify_workload
from caracal7.workloads.sha256 import Sha256
from caracal7.workloads.keccak import Keccak
from caracal7.workloads.poseidon import Poseidon
from caracal7.workloads.ecdsa import Ecdsa, Point
from caracal7.workloads.bigint import Big


def _hex(s: String) raises -> List[UInt8]:
    var out = List[UInt8](capacity=s.byte_length() // 2)
    for i in range(0, s.byte_length(), 2):
        out.append(UInt8(atol(String(s[byte=i]) + String(s[byte=i + 1]), 16)))
    return out^


def _run[p: Params, W: Workload](cmd: String, w: W, path: String) raises:
    if cmd == "prove":
        var t0 = perf_counter_ns()
        var ctx = DeviceContext()
        var proof = prove_workload[p, Blake3](ctx, w)
        var ms = (perf_counter_ns() - t0) // 1000000
        var f = open(path, "w")
        f.write_bytes(Span(proof))
        f.close()
        var cells = w.statement[p]().compile[p]().layout.columns_w() * p.N()   # committed witness cells: the circuit size we report
        print("prove_ms", ms, "proof_bytes", len(proof), "cells", cells, "grid", p)
    else:
        var t0 = perf_counter_ns()
        var proof = open(path, "r").read_bytes()
        var ok = verify_workload[p, Blake3](proof^, w, w.public_inputs[p]())
        print("verify_ms", (perf_counter_ns() - t0) // 1000000, "ok", ok)
        if not ok:
            exit(1)


def _sha(cmd: String, size: Int, path: String, var msg: List[UInt8]) raises:
    if size == 128:
        _run[CLIENT.grid(32, 193)](cmd, Sha256(msg^), path)
    elif size == 256:
        _run[CLIENT.grid(32, 321)](cmd, Sha256(msg^), path)
    elif size == 512:
        _run[CLIENT.grid(32, 577)](cmd, Sha256(msg^), path)
    elif size == 1024:
        _run[CLIENT.grid(32, 1089)](cmd, Sha256(msg^), path)
    elif size == 2048:
        _run[CLIENT.grid(32, 2113)](cmd, Sha256(msg^), path)
    else:
        raise Error("sha256 sizes: 128, 256, 512, 1024, 2048")


def _keccak(cmd: String, size: Int, path: String, var msg: List[UInt8]) raises:
    if size == 128:
        _run[CLIENT.grid(64, 24)](cmd, Keccak(msg^), path)
    elif size == 256:
        _run[CLIENT.grid(64, 48)](cmd, Keccak(msg^), path)
    elif size == 512:
        _run[CLIENT.grid(64, 96)](cmd, Keccak(msg^), path)
    elif size == 1024:
        _run[CLIENT.grid(64, 192)](cmd, Keccak(msg^), path)
    elif size == 2048:
        _run[CLIENT.grid(64, 384)](cmd, Keccak(msg^), path)
    else:
        raise Error("keccak sizes: 128, 256, 512, 1024, 2048")


def _poseidon(cmd: String, size: Int, path: String, var elems: List[Int]) raises:
    if size <= 8:
        _run[CLIENT.grid(64, 368)](cmd, Poseidon(elems^), path)
    elif size <= 16:
        _run[CLIENT.grid(64, 720)](cmd, Poseidon(elems^), path)
    else:
        raise Error("poseidon sizes: 1 to 16 elements")


def main() raises:
    var a = argv()
    if len(a) < 5:
        raise Error("usage: caracal7 prove|verify <target> <size> <proof-path> <input...>")
    var cmd = String(a[1])
    if cmd != "prove" and cmd != "verify":
        raise Error("command is prove or verify")
    var target = String(a[2])
    var size = atol(String(a[3]))
    var path = String(a[4])
    if target == "sha256" or target == "keccak":
        var msg = _hex(String(a[5]))
        if len(msg) != size:
            raise Error("message length is not the input size")
        if target == "sha256":
            _sha(cmd, size, path, msg^)
        else:
            _keccak(cmd, size, path, msg^)
    elif target == "poseidon":
        var elems = List[Int]()
        for i in range(5, len(a)):
            elems.append(atol(String(a[i])))
        if len(elems) != size:
            raise Error("element count is not the input size")
        _poseidon(cmd, size, path, elems^)
    elif target == "ecdsa":
        if len(a) != 9:
            raise Error("ecdsa inputs: digest x_Q y_Q signature")
        var sig = String(a[8])
        var w = Ecdsa(Big.from_hex(String(sig[byte=0:64])), Big.from_hex(String(sig[byte=64:128])), Big.from_hex(String(a[5])),
                      Point(Big.from_hex(String(a[6])), Big.from_hex(String(a[7])), False))
        _run[CLIENT.grid(144, 576)](cmd, w, path)
    else:
        raise Error("targets: sha256, keccak, poseidon, ecdsa")
