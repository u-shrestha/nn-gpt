"""
ab/gpt/brute/trans/augment/stat_utils.py — shared result-pool parser for the
augmentation-schedule track.

AugEval.py, TPE.py and (via sched_gen()) the LLM generation loop all write
one JSON result file per (config_id, seed, fidelity) into directories named
*gen_result* under trans_dir (the flat augment/gen_result/ pool, plus any
per-seed backup copies such as datasets/aug_gen_result_seed101). This module
is the single place that knows that layout, so every caller reads the same
pool the same way instead of re-implementing the glob/parse logic.

load_results() returns the RAW pool (all validity classes, all fidelities).
Callers are responsible for filtering (e.g. AugmentGenPrompt.load_schedule_pool
keeps only validity_class == 'ok' and optionally a single fidelity).
"""

import json
from pathlib import Path

import pandas as pd

from ab.gpt.util.Const import trans_dir

_SKIP_SUFFIXES = ('_training_summary.json',)
_SKIP_PREFIXES = ('INFRA_',)
_SKIP_NAMES = {'gen_stats.json'}


def _result_dirs():
    """Every directory under trans_dir whose name contains 'gen_result' —
    the flat augment/gen_result/ pool plus any per-seed backup copies."""
    seen, dirs = set(), []
    for p in trans_dir.rglob('*gen_result*'):
        if p.is_dir() and p not in seen:
            seen.add(p)
            dirs.append(p)
    return dirs


def _iter_result_files():
    for d in _result_dirs():
        for f in sorted(d.glob('*.json')):
            name = f.name
            if name in _SKIP_NAMES or name.endswith(_SKIP_SUFFIXES) or name.startswith(_SKIP_PREFIXES):
                continue
            yield f


def load_results() -> pd.DataFrame:
    """Load every per-config AugEval/TPE result JSON into one tidy DataFrame,
    one row per (config_id, seed, fidelity).

    A config_id can appear in more than one *gen_result* directory (e.g.
    mirrored into a per-seed backup folder); when that happens the larger
    file wins (a full history is more complete than a bare error stub).
    """
    rows = {}
    for f in _iter_result_files():
        try:
            with open(f) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as e:
            print(f'[stat_utils] Skipping unreadable {f}: {e}')
            continue
        if 'config_id' not in data:
            continue
        key = (data['config_id'], data.get('seed'), data.get('fidelity'))
        size = f.stat().st_size
        prev = rows.get(key)
        if prev is None or size > prev[1]:
            data['_source_file'] = str(f)
            rows[key] = (data, size)

    columns = ['config_id', 'augment', 'accuracy', 'best_accuracy',
              'fidelity', 'seed', 'validity_class', '_source_file']
    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(data for data, _size in rows.values())
