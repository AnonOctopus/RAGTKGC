"""Shared confidence-statistics helpers.

Used by the three confidence-driven statistics that help pick ``--top_k_rules``
and ``--confidence_threshold``:

  * ``naive_history_metadata.py``      — per-rule confidence in the summary
  * ``apply_history_filters.py``       — avg confidence of the filtered history
  * ``rule_confidence_profile.py``     — per-relation top-k confidence decay

Confidence values live in ``common_rule_pool.json`` under
``algorithm_stats[<algorithm>].conf`` (range 0.0-1.0), which holds the most
recent run of that algorithm; the per-run history is under ``runs``.
``load_rule_conf_map``
flattens that pool for a chosen mining algorithm; ``conf_stats`` summarizes any
list of confidences with avg/min/max plus percentiles, which is what lets a
threshold or k be chosen from the distribution rather than guessed.
"""

import json
from typing import Dict, List, Optional, Tuple

_PERCENTILES = (10, 25, 50, 75, 90)


def _percentile(sorted_values: List[float], pct: float) -> float:
    """Linear-interpolation percentile over an already-sorted list.

    ``pct`` is in [0, 100]. Matches numpy's default ('linear') method so the
    numbers are comparable to anything computed elsewhere with numpy.
    """
    if not sorted_values:
        raise ValueError("percentile of empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * frac


def conf_stats(values: List[float], ndigits: int = 4) -> Dict[str, object]:
    """Summary statistics for a list of confidences.

    Returns count/avg/min/max and the p10/p25/p50/p75/p90 percentiles. All
    floats are rounded to ``ndigits`` (matching the existing ``_aggregate``
    style). On an empty input every numeric field is ``None`` so callers can
    serialize the result unconditionally.
    """
    count = len(values)
    if count == 0:
        empty = {"count": 0, "avg": None, "min": None, "max": None}
        empty.update({f"p{p}": None for p in _PERCENTILES})
        return empty

    sorted_values = sorted(values)
    stats: Dict[str, object] = {
        "count": count,
        "avg": round(sum(values) / count, ndigits),
        "min": round(sorted_values[0], ndigits),
        "max": round(sorted_values[-1], ndigits),
    }
    for p in _PERCENTILES:
        stats[f"p{p}"] = round(_percentile(sorted_values, p), ndigits)
    return stats


def load_rule_conf_map(
    rule_pool_path: str, algorithm: str
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Read ``common_rule_pool.json`` into ``{rule_id: conf}`` / ``{rule_id: head_rel}``.

    The pool is a flat dict keyed by rule_id (SHA1); each rule stores its
    confidence per mining algorithm under ``algorithm_stats[<algorithm>].conf``.
    Rules that were not found by ``algorithm`` (no matching ``algorithm_stats``
    entry) are skipped, so the returned maps cover only that algorithm's rules.
    """
    with open(rule_pool_path, "r", encoding="utf-8") as fin:
        pool = json.load(fin)

    rule_to_conf: Dict[str, float] = {}
    rule_to_head_rel: Dict[str, int] = {}
    for rule_id, rule in pool.items():
        algo_stats = rule.get("algorithm_stats", {})
        entry = algo_stats.get(algorithm)
        if not entry or "conf" not in entry:
            continue
        rule_to_conf[rule_id] = float(entry["conf"])
        head_rel = rule.get("head_rel")
        if head_rel is not None:
            rule_to_head_rel[rule_id] = int(head_rel)
    return rule_to_conf, rule_to_head_rel
