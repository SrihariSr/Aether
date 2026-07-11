"""Prove the streaming pipeline produces byte-identical output to the old in-memory one."""
import os
import tempfile

import numpy as np

rng = np.random.default_rng(0)
docs = [rng.integers(257, 50257, rng.integers(50, 400)).tolist() for _ in range(3000)]


def old_way():
    all_tokens = []
    for t in docs:
        all_tokens.extend(t)
    a = np.array(all_tokens, dtype=np.uint16)
    sp = int(len(a) * 0.9)
    return a[:sp].tobytes(), a[sp:].tobytes()


def new_way(tmp):
    all_path = os.path.join(tmp, "all.bin")
    total, buf = 0, []
    with open(all_path, "wb") as f:
        for t in docs:
            buf.extend(t)
            if len(buf) >= 5000:                       # small flush to exercise the path
                np.array(buf, dtype=np.uint16).tofile(f)
                total += len(buf)
                buf.clear()
        if buf:
            np.array(buf, dtype=np.uint16).tofile(f)
            total += len(buf)
            buf.clear()

    data = np.memmap(all_path, dtype=np.uint16, mode="r")
    assert len(data) == total
    sp = int(len(data) * 0.9)
    out = []
    for lo, hi in [(0, sp), (sp, len(data))]:
        p = os.path.join(tmp, f"{lo}.bin")
        with open(p, "wb") as f:
            for st in range(lo, hi, 1000):
                np.asarray(data[st:min(st + 1000, hi)]).tofile(f)
        out.append(open(p, "rb").read())
    del data
    return out[0], out[1]


with tempfile.TemporaryDirectory() as tmp:
    o_tr, o_va = old_way()
    n_tr, n_va = new_way(tmp)

assert o_tr == n_tr, "train.bin differs"
assert o_va == n_va, "val.bin differs"
print(f"PASS: {sum(len(d) for d in docs):,} tokens, train.bin and val.bin byte-identical")
