"""
Segment-order permutations of already-evaluated schedules.

A schedule is a sequence of segments [[method, alpha, fraction], ...].
Everything ConfigGen.py produces varies WHAT is applied and HOW MUCH of
the epoch each method gets, but never touches the ORDER two methods run
in for a fixed set of segments. This module isolates order as the sole
independent variable: given an evaluated schedule, it reorders its
existing segments (same methods, same alphas, same fractions) into new
sequences, so "gridmask then cutout" can be compared directly against
"cutout then gridmask" at matched composition.

Source selection. By default the --top_k highest-accuracy evaluated
schedules with >= 2 segments are used as sources; single-segment and
all-'none' schedules have no order to vary and are skipped. Pass
--source_ids to override with an explicit list of config_ids.

Sampling. A schedule with n segments has n! orderings. n=2 -> 2, n=3 -> 6,
n=4 -> 24. The identity ordering is dropped (that result already exists
in gen_result/). For n<=3 every remaining ordering is emitted; for n=4 a
random sample of --max_per_source is drawn (seeded, deterministic), since
evaluating all 23 non-identity 4-segment orderings per source is more
than the GPU budget for this study needs.

Deduplication. Every reordering is re-validated with policy='reject'
(these are already-legal segments; this is a sanity check, not a repair
pass) and re-hashed with ConfigGen.config_hash so it follows the exact
hashing convention the rest of the corpus uses. Anything that collides
with an already-evaluated config_id, an already-generated config_id in
any config/aug_config_*.json file, or another permutation emitted earlier
in this run, is dropped.

Output. Same shape AugEval.load_configs() already reads:
    {"1": {"config_id": ..., "augment_configs": [...],
           "source_config_id": ..., "source_order": [...],
           "permutation": [...]}, ...}
The extra keys are ignored by load_configs but let later analysis group
permutations by the schedule they came from.
"""

import argparse
import glob
import itertools
import json
import os
import random
from pathlib import Path

from ab.gpt.brute.trans.augment.ScheduleValidate import validate_schedule
from ab.gpt.brute.trans.augment.config.ConfigGen import config_hash
from ab.gpt.util.Const import trans_dir, out_dir



RESULT_DIR = trans_dir / 'augment/gen_result'
CONFIG_DIR = trans_dir / 'augment/config'
RUN_MINUTES_PER_CONFIG = 52.6  # observed median, 30-epoch ImageNette/ResNet


def _method_seq(segments):
    return [m for m, a, f in segments]


def load_evaluated(result_dir: Path):
    """config_id -> {'augment': segments, 'accuracy': float}, from full
    per-run result files in gen_result/ (training_summary files skipped)."""
    out = {}
    for f in glob.glob(str(result_dir / "*.json")):
        if "training_summary" in f or os.path.getsize(f) == 0:
            continue
        try:
            j = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        cid = j.get("config_id")
        if cid and "augment" in j and "accuracy" in j:
            out[cid] = {"augment": j["augment"], "accuracy": j["accuracy"]}
    return out


def known_config_ids(config_dir: Path, evaluated: dict):
    """Every config_id already spoken for: evaluated results plus every
    config ever written to a config/aug_config_*.json file, run or not."""
    ids = set(evaluated)
    for f in glob.glob(str(config_dir / "aug_config_*.json")):
        d = json.load(open(f))
        for entry in d.values():
            cid = entry.get("config_id") if isinstance(entry, dict) else None
            if cid:
                ids.add(cid)
    return ids


def select_sources(evaluated: dict, top_k: int, source_ids=None):
    eligible = {cid: v for cid, v in evaluated.items() if len(v["augment"]) >= 2}
    if source_ids:
        missing = [c for c in source_ids if c not in eligible]
        if missing:
            raise ValueError(f"--source_ids not found (or single-segment): {missing}")
        return [(c, eligible[c]) for c in source_ids]
    ranked = sorted(eligible.items(), key=lambda kv: -kv[1]["accuracy"])
    return ranked[:top_k]


def permute_source(segments, rng, max_per_source):
    """All non-identity orderings for n<=3; a seeded random sample beyond that."""
    n = len(segments)
    identity = tuple(range(n))
    all_perms = [p for p in itertools.permutations(range(n)) if p != identity]
    if len(all_perms) > max_per_source:
        all_perms = rng.sample(all_perms, max_per_source)
    out = []
    for perm in all_perms:
        reordered = [segments[i] for i in perm]
        canonical, _ = validate_schedule(reordered, policy="reject")
        out.append((perm, canonical))
    return out


def generate(result_dir: Path, config_dir: Path, top_k: int,
             max_per_source: int, seed: int, source_ids=None):
    evaluated = load_evaluated(result_dir)
    if not evaluated:
        raise RuntimeError(f"No evaluated schedules with full history found in {result_dir}")
    seen_ids = known_config_ids(config_dir, evaluated)
    rng = random.Random(seed)

    sources = select_sources(evaluated, top_k, source_ids)
    skipped_no_order = sum(1 for v in evaluated.values() if len(v["augment"]) < 2)

    entries = []
    per_source_counts = {}
    for source_id, src in sources:
        segments = src["augment"]
        candidates = permute_source(segments, rng, max_per_source)
        kept = 0
        for perm, canonical in candidates:
            cid = config_hash(canonical)
            if cid == source_id or cid in seen_ids:
                continue
            seen_ids.add(cid)
            entries.append({
                "config_id": cid,
                "augment_configs": canonical,
                "source_config_id": source_id,
                "source_order": _method_seq(segments),
                "permutation": list(perm),
            })
            kept += 1
        per_source_counts[source_id] = (len(candidates), kept)

    return entries, sources, per_source_counts, skipped_no_order


def save(entries, out_dir, seed):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"aug_config_perm_seed{seed}_n{len(entries)}.json")
    output = {str(i + 1): e for i, e in enumerate(entries)}
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Generate segment-order permutations of evaluated schedules")
    ap.add_argument("--result_dir", type=str, default=str(RESULT_DIR))
    ap.add_argument("--config_dir", type=str, default=str(CONFIG_DIR))
    ap.add_argument("--top_k", type=int, default=20,
                    help="highest-accuracy evaluated schedules to permute")
    ap.add_argument("--max_per_source", type=int, default=6,
                    help="cap on new orderings per source; all are kept if n!-1 is smaller")
    ap.add_argument("--source_ids", type=str, default=None,
                    help="comma-separated config_ids, overrides --top_k selection")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    source_ids = args.source_ids.split(",") if args.source_ids else None
    entries, sources, per_source_counts, skipped = generate(
        Path(args.result_dir), Path(args.config_dir),
        args.top_k, args.max_per_source, args.seed, source_ids)

    print(f"Sources considered: {len(sources)}  "
          f"(skipped {skipped} evaluated schedules with < 2 segments)")
    for source_id, (available, kept) in per_source_counts.items():
        print(f"  {source_id}: {available} orderings tried, {kept} new "
              f"({available - kept} duplicated an existing config)")

    if not entries:
        print("No new permutations to emit.")
        return

    out_path = save(entries, args.config_dir, args.seed)
    est_hours = len(entries) * RUN_MINUTES_PER_CONFIG / 60
    print(f"\nTotal new permutations: {len(entries)}")
    print(f"Estimated cost at ~{RUN_MINUTES_PER_CONFIG:.1f} min/run (30-epoch fidelity): "
          f"{est_hours:.1f} GPU-hours")
    print(f"Wrote -> {out_path}")
    print(f"Run with: python -m ab.gpt.brute.trans.augment.AugEval "
          f"--config_file {os.path.basename(out_path)} --seed {args.seed}")


if __name__ == "__main__":
    main()
