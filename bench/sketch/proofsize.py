"""Level-1 bytes of the RS and the sketched code: distinct rows x row bytes + 32 B per frontier sibling + u32
per tree, per tree (W, Z, Q), 5 random draws. Other proof regions are taken from docs/profile.md."""
import random, math

def frontier(leaves, pos):
    known = sorted(set(pos)); sib = 0; n = leaves
    while n > 1:
        s = set(known); nxt = set()
        for i in known:
            j = i ^ 1
            if j < n and j not in s: sib += 1
            nxt.add(i // 2)
        known = sorted(nxt); n = (n + 1) // 2
    return sib

def level1(leaves, rows_bytes, draws):        # draws: list of (lo, hi, count)
    tot = 0
    for rb in rows_bytes:
        best = []
        for t in range(5):
            random.seed(t)
            pos = [random.randrange(lo, hi) for lo, hi, c in draws for _ in range(c)]
            best.append(4 + len(set(pos)) * rb + 32 * frontier(leaves, pos))
        tot += sum(best) / len(best)
    return tot

# name, W/Z/Q coordinate columns, L, K, measured level-1 bytes, measured total, RS queries
cases = [("sha256", (43, 0, 60), 161280, 21504, 81856, 194828, 76),
         ("rsa-2048", (172, 120, 60), 322560, 72576, 251348, 580402, 103),
         ("passport", (285, 140, 60), 645120, 145152, 314304, 1151566, 103)]
sk = [("16/128 F", 1/8, 2638), ("16/128 F4", 1/8, 1930), ("24/192 F4", 1/8, 1754)]
q_y = math.ceil(92 / -math.log2(1 - (1/2) / 3))   # the RS sketch part at delta/3, rate 1/2
print("sketch-part queries:", q_y)
for name, cols, L, K, meas, total, q in cases:
    rb = [4 * c for c in cols if c]
    now = level1(L, rb, [(0, L, q)])
    rest = total - meas
    print(f"{name}: level-1 model {now/1e3:.0f} KB (measured {meas/1e3:.0f} KB), total {total/1e6:.2f} MB")
    for lab, alpha, qx in sk:
        Ls = 2 * math.ceil(K * alpha)                # rate 1/2 on the sketch (the domain rounds up a little)
        v = level1(K + Ls, rb, [(0, K, qx), (K, K + Ls, q_y)])
        print(f"   {lab}: level-1 {v/1e6:.2f} MB, total {(rest + v)/1e6:.2f} MB ({(rest + v)/total:.1f}x)")
