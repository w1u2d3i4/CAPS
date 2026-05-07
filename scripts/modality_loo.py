#!/usr/bin/env python3
"""E5: Per-modality marginal contribution (leave-one-out from S1).

Derived from existing MMS-Bench results — no compute required.
For each strategy with scaffold MMS results, compute:
    Δ_1H  = top1(S1) - top1(S6)   # S6 = 13C+IR (1H removed)
    Δ_13C = top1(S1) - top1(S5)   # S5 = 1H+IR  (13C removed)
    Δ_IR  = top1(S1) - top1(S4)   # S4 = 1H+13C (IR removed)

Output: results/modality_loo.json
"""
import json
from pathlib import Path
import os

os.chdir(Path(__file__).resolve().parent.parent)

# Sources of MMS-Bench top-1 numbers (must contain S1, S4, S5, S6).
SOURCES = {
    "scaffold_full_caps":       "results/scaffold_full_caps_100ep_mms.json",
    "scaffold_spectre_dropout": "results/scaffold_spectre_dropout_100ep_mms.json",
    "random_full_caps":         "results/ablation_no_caps_100ep_mms.json",
    "random_spectre_dropout":   "results/ablation_spectre_dropout_100ep_mms.json",
    "random_proxy_only":        "results/ablation_proxy_only_100ep_mms.json",
    "random_mmp_only":          "results/ablation_mmp_only_100ep_mms.json",
}


def extract_top1(d):
    """Pull a {scenario: top1} dict from a MMS results JSON.

    The MMS files use keys like 'S1', 'S2', ... each pointing to a dict that
    contains 'top1'.  Some older files use 'S1_full' style keys.
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and "top1" in v:
            out[k] = float(v["top1"])
    return out


def find_scenario(top1_dict, prefix):
    """Find the entry whose key matches the scenario prefix (e.g. 'S1', 'S4')."""
    if prefix in top1_dict:
        return top1_dict[prefix]
    for k, v in top1_dict.items():
        if k.startswith(prefix + "_") or k.lower().startswith(prefix.lower() + "_"):
            return v
    return None


def main():
    out = {}
    for name, path in SOURCES.items():
        if not Path(path).exists():
            print(f"  SKIP {name}: {path} not found")
            continue
        with open(path) as f:
            data = json.load(f)
        t1 = extract_top1(data)
        s1 = find_scenario(t1, "S1")
        s4 = find_scenario(t1, "S4")
        s5 = find_scenario(t1, "S5")
        s6 = find_scenario(t1, "S6")
        if None in (s1, s4, s5, s6):
            print(f"  SKIP {name}: missing scenario(s) — S1={s1} S4={s4} S5={s5} S6={s6}")
            continue
        d_1h  = s1 - s6
        d_13c = s1 - s5
        d_ir  = s1 - s4
        out[name] = {
            "S1_top1":  round(s1, 4),
            "S4_top1":  round(s4, 4),
            "S5_top1":  round(s5, 4),
            "S6_top1":  round(s6, 4),
            "delta_1H_pp":  round(d_1h * 100, 2),
            "delta_13C_pp": round(d_13c * 100, 2),
            "delta_IR_pp":  round(d_ir * 100, 2),
            "ranking_by_marginal":
                sorted([("1H", d_1h), ("13C", d_13c), ("IR", d_ir)],
                       key=lambda x: -x[1]),
        }
        print(f"  {name:>30s}: Δ1H={d_1h*100:+.2f}pp  Δ13C={d_13c*100:+.2f}pp  ΔIR={d_ir*100:+.2f}pp")

    out_path = "results/modality_loo.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
