"""
Regression harness for the Aether correctness pass.

Run on main to capture a baseline, then again on the branch to compare.
    python -m scripts.regression_check --save    # on main
    python -m scripts.regression_check --check   # on the branch

Two checks:
  A (structural, always runs): state_dict keys and shapes are unchanged.
  B (numerical, only if a checkpoint exists): logits from the RESTORED model are bit-identical.

Forced to CPU and float32 so the comparison is exact and not at the mercy of MPS kernels.
"""
import argparse
import hashlib
import json
import os

import torch

from model.config import aether_medium
from model.transformer import GPT

BASELINE = "scripts/_baseline.json"
CKPT = "checkpoints/medium/best.pt"
FIXED_INPUT = [15496, 11, 616, 1438, 318, 220, 32, 2788, 263, 13]  # arbitrary, must not change


def signature():
    torch.manual_seed(0)
    model = GPT(aether_medium())

    # A: structural signature of the state_dict
    sd = model.state_dict()
    keys = "\n".join(f"{k} {tuple(v.shape)}" for k, v in sd.items())
    struct = hashlib.sha256(keys.encode()).hexdigest()

    # C: initialisation stds (these are EXPECTED to change in Task 1; recorded, not asserted)
    stds = {
        pn: round(float(p.std()), 5)
        for pn, p in model.named_parameters()
        if pn.startswith("blocks.0.") and "proj.weight" in pn
    }

    # B: logits from the restored checkpoint
    logits_hash = None
    if os.path.exists(CKPT):
        ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        x = torch.tensor([FIXED_INPUT], dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(x)
        logits_hash = hashlib.sha256(
            logits.float().contiguous().numpy().tobytes()
        ).hexdigest()

    return {
        "n_keys": len(sd),
        "struct_sha256": struct,
        "init_stds": stds,
        "logits_sha256": logits_hash,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    sig = signature()

    if args.save:
        with open(BASELINE, "w") as f:
            json.dump(sig, f, indent=2)
        print("BASELINE SAVED")
        print(json.dumps(sig, indent=2))
        if sig["logits_sha256"] is None:
            print("\nWARNING: no checkpoint at", CKPT)
            print("Check B (the important one) will be skipped. Fetch the checkpoint if you can.")

    if args.check:
        with open(BASELINE) as f:
            base = json.load(f)

        ok = True

        # Check A
        a = sig["struct_sha256"] == base["struct_sha256"] and sig["n_keys"] == base["n_keys"]
        print(f"[A] state_dict structure unchanged : {'PASS' if a else 'FAIL'}")
        if not a:
            print(f"    keys {base['n_keys']} -> {sig['n_keys']}")
            print("    You renamed or added a parameter. This breaks the trained checkpoint.")
            ok = False

        # Check B
        if base["logits_sha256"] is None:
            print("[B] restored-model logits          : SKIPPED (no baseline checkpoint)")
        else:
            b = sig["logits_sha256"] == base["logits_sha256"]
            print(f"[B] restored-model logits identical: {'PASS' if b else 'FAIL'}")
            if not b:
                print("    The forward pass of the TRAINED model changed. Nothing in this pass")
                print("    should do that. Find what you changed and revert it.")
                ok = False

        # Check C (informational)
        print("[C] init stds (expected to change in Task 1):")
        for k in sorted(set(base["init_stds"]) | set(sig["init_stds"])):
            print(f"    {k:38s} {base['init_stds'].get(k)} -> {sig['init_stds'].get(k)}")

        raise SystemExit(0 if ok else 1)
