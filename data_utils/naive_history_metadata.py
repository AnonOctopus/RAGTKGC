import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional

from tqdm import tqdm


@dataclass
class Fact:
    sub: str
    rel: str
    obj: str
    time_value: float
    time_raw: str
    source_split: str
    source_index: int
    is_indirect: bool = False


HISTORY_LINE_PATTERN = re.compile(
    r"^\s*(?P<time>[^:]+):\s*\[(?P<sub>.*?),\s*(?P<rel>.*?),\s*(?P<obj>.*?)\]\s*$"
)
QUERY_LINE_PATTERN = re.compile(
    r"^\s*(?P<time>[^:]+):\s*\[(?P<sub>.*?),\s*(?P<rel>.*?),\s*$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build naive subject-history metadata for quads and report target-object mention stats."
        )
    )
    parser.add_argument("--dataset", required=False, type=str, help="Dataset name, e.g. icews14")
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test"],
        default="test",
        help="Target split for which metadata is computed",
    )
    parser.add_argument(
        "--include_indirect",
        action="store_true",
        default=True,
        help=(
            "If set, also include indirect facts where the query subject appears as object (X, r, subject, t)."
        ),
    )
    parser.add_argument(
        "--recent_n",
        nargs="+",
        type=int,
        default=[50],
        help="Window sizes n for 'target mentioned in n most recent history facts'",
    )
    parser.add_argument(
        "--test_source",
        choices=["all", "split"],
        default="all",
        help=(
            "When split=test: use all train+valid+test facts as source ('all') or only test ('split')."
        ),
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        default="../data/original",
        help="Base folder containing dataset subfolders with train/valid/test and relation2id",
    )
    parser.add_argument(
        "--max_quads",
        type=int,
        default=None,
        help="Optional cap on number of target quads to process",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help="Output path for per-quad metadata JSONL",
    )
    parser.add_argument(
        "--output_summary",
        type=str,
        default=None,
        help="Output path for aggregate summary JSON",
    )
    parser.add_argument(
        "--precomputed_json",
        type=str,
        default=None,
        help=(
            "Path to precomputed history JSON (list of {context, target}) such as history_modeling_train/*.json. "
            "If provided, statistics are computed from these contexts instead of rebuilding history from raw splits."
        ),
    )
    parser.add_argument(
        "--compare_summary_a",
        type=str,
        default=None,
        help="Path to first summary JSON for rule-usage comparison mode",
    )
    parser.add_argument(
        "--compare_summary_b",
        type=str,
        default=None,
        help="Path to second summary JSON for rule-usage comparison mode",
    )
    parser.add_argument(
        "--compare_label_a",
        type=str,
        default=None,
        help="Optional label for first summary (e.g., mining algorithm name)",
    )
    parser.add_argument(
        "--compare_label_b",
        type=str,
        default=None,
        help="Optional label for second summary (e.g., mining algorithm name)",
    )
    parser.add_argument(
        "--compare_output",
        type=str,
        default=None,
        help="Output path for comparison JSON; defaults next to compare_summary_a",
    )
    parser.add_argument(
        "--compare_results_a",
        type=str,
        default=None,
        help="Path to first results JSONL for correctness overlap comparison",
    )
    parser.add_argument(
        "--compare_results_b",
        type=str,
        default=None,
        help="Path to second results JSONL for correctness overlap comparison",
    )
    parser.add_argument(
        "--compare_results_label_a",
        type=str,
        default=None,
        help="Optional label for first results file",
    )
    parser.add_argument(
        "--compare_results_label_b",
        type=str,
        default=None,
        help="Optional label for second results file",
    )
    parser.add_argument(
        "--compare_results_output",
        type=str,
        default=None,
        help="Output path for results comparison JSON; defaults next to compare_results_a",
    )
    parser.add_argument(
        "--results_jsonl",
        type=str,
        default=None,
        help=(
            "Optional model results JSONL aligned by row order with processed quads; "
            "must contain 'targets' and 'predictions' fields for correctness metrics."
        ),
    )
    return parser.parse_args()


def _summary_label(path: str, explicit_label: Optional[str]) -> str:
    if explicit_label:
        return explicit_label
    return os.path.splitext(os.path.basename(path))[0]


def _read_rule_sets_from_jsonl(path: str) -> List[set]:
    rule_sets: List[set] = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rule_ids = row.get("rule_ids")
            if isinstance(rule_ids, list):
                rule_sets.append(set(str(x) for x in rule_ids))
            else:
                rule_sets.append(set())
    return rule_sets


def _first_item(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _normalize_text(value) -> Optional[str]:
    if value is None:
        return None
    return str(value).strip()


def load_prediction_correctness(path: str, max_rows: Optional[int] = None) -> List[bool]:
    correctness: List[bool] = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            target = _normalize_text(_first_item(row.get("targets")))
            prediction = _normalize_text(_first_item(row.get("predictions")))
            correctness.append(target is not None and prediction is not None and target == prediction)
            if max_rows is not None and len(correctness) >= max_rows:
                break
    return correctness


def _init_prediction_bucket(recent_n: List[int]) -> Dict[str, object]:
    return {
        "count": 0,
        "with_target": 0,
        "total_target_mentions": 0,
        "closest_pool": [],
        "farthest_pool": [],
        "within_n_counts": {n: 0 for n in recent_n},
    }


def _prediction_bucket_summary(bucket: Dict[str, object], recent_n: List[int]) -> Dict[str, object]:
    count = int(bucket["count"])
    with_target = int(bucket["with_target"])
    total_target_mentions = int(bucket["total_target_mentions"])
    closest_pool = bucket["closest_pool"]
    farthest_pool = bucket["farthest_pool"]
    within_n_counts = bucket["within_n_counts"]

    def safe_avg(values: List[float]) -> Optional[float]:
        return (sum(values) / len(values)) if values else None

    return {
        "num_quads": count,
        "num_with_target_in_history": with_target,
        "ratio_with_target_in_history": ((with_target / count) if count else 0.0),
        "target_in_last_n_counts": {str(k): within_n_counts[k] for k in sorted(within_n_counts.keys())},
        "target_in_last_n_ratios": {
            str(k): ((within_n_counts[k] / count) if count else 0.0)
            for k in sorted(within_n_counts.keys())
        },
        "avg_target_mentions_per_quad": ((total_target_mentions / count) if count else 0.0),
        "avg_target_mentions_per_quad_when_present": (
            (total_target_mentions / with_target) if with_target else 0.0
        ),
        "avg_closest_target_mention_index_when_present": safe_avg(closest_pool),
        "avg_farthest_target_mention_index_when_present": safe_avg(farthest_pool),
    }


def compare_rule_usage_summaries(args: argparse.Namespace) -> None:
    with open(args.compare_summary_a, "r", encoding="utf-8") as fa:
        summary_a = json.load(fa)
    with open(args.compare_summary_b, "r", encoding="utf-8") as fb:
        summary_b = json.load(fb)

    label_a = _summary_label(args.compare_summary_a, args.compare_label_a)
    label_b = _summary_label(args.compare_summary_b, args.compare_label_b)

    rules_pct_a = summary_a.get("rules_usage_pct_among_all_facts", {})
    rules_pct_b = summary_b.get("rules_usage_pct_among_all_facts", {})
    if not isinstance(rules_pct_a, dict):
        rules_pct_a = {}
    if not isinstance(rules_pct_b, dict):
        rules_pct_b = {}

    rules_a = set(str(x) for x in rules_pct_a.keys())
    rules_b = set(str(x) for x in rules_pct_b.keys())
    overlap = rules_a & rules_b
    unique_a = rules_a - rules_b
    unique_b = rules_b - rules_a

    nonempty_a = int(summary_a.get("num_quads_with_nonempty_history", 0) or 0)
    nonempty_b = int(summary_b.get("num_quads_with_nonempty_history", 0) or 0)

    overlap_rows = []
    for rid in overlap:
        pct_all_a = float(rules_pct_a.get(rid, 0.0) or 0.0)
        pct_all_b = float(rules_pct_b.get(rid, 0.0) or 0.0)
        overlap_rows.append(
            {
                "rule_id": rid,
                f"pct_among_all_facts_{label_a}": pct_all_a,
                f"pct_among_all_facts_{label_b}": pct_all_b,
            }
        )

    overlap_rows.sort(
        key=lambda x: (
            -max(
                x[f"pct_among_all_facts_{label_a}"],
                x[f"pct_among_all_facts_{label_b}"],
            ),
            x["rule_id"],
        )
    )

    unique_rows_a = [
        {
            "rule_id": rid,
            "pct_among_all_facts": float(rules_pct_a.get(rid, 0.0) or 0.0),
        }
        for rid in unique_a
    ]
    unique_rows_b = [
        {
            "rule_id": rid,
            "pct_among_all_facts": float(rules_pct_b.get(rid, 0.0) or 0.0),
        }
        for rid in unique_b
    ]
    unique_rows_a.sort(key=lambda x: (-x["pct_among_all_facts"], x["rule_id"]))
    unique_rows_b.sort(key=lambda x: (-x["pct_among_all_facts"], x["rule_id"]))

    # Exact coverage percentages are computed from per-quad metadata JSONL if available.
    coverage_info = {
        "available": False,
        "reason": "Missing or unreadable output_jsonl in one or both summaries.",
    }

    jsonl_a = summary_a.get("output_jsonl")
    jsonl_b = summary_b.get("output_jsonl")
    if isinstance(jsonl_a, str) and isinstance(jsonl_b, str) and os.path.exists(jsonl_a) and os.path.exists(jsonl_b):
        rule_sets_a = _read_rule_sets_from_jsonl(jsonl_a)
        rule_sets_b = _read_rule_sets_from_jsonl(jsonl_b)

        total_facts_a = len(rule_sets_a)
        total_facts_b = len(rule_sets_b)

        overlap_covered_a = sum(1 for rset in rule_sets_a if (rset & overlap))
        overlap_covered_b = sum(1 for rset in rule_sets_b if (rset & overlap))
        unique_covered_a = sum(1 for rset in rule_sets_a if (rset & unique_a))
        unique_covered_b = sum(1 for rset in rule_sets_b if (rset & unique_b))

        coverage_info = {
            "available": True,
            "source_jsonl": {
                label_a: jsonl_a,
                label_b: jsonl_b,
            },
            "fact_coverage_pct": {
                f"overlap_{label_a}": ((overlap_covered_a / total_facts_a) * 100.0) if total_facts_a else 0.0,
                f"overlap_{label_b}": ((overlap_covered_b / total_facts_b) * 100.0) if total_facts_b else 0.0,
                f"unique_{label_a}": ((unique_covered_a / total_facts_a) * 100.0) if total_facts_a else 0.0,
                f"unique_{label_b}": ((unique_covered_b / total_facts_b) * 100.0) if total_facts_b else 0.0,
            },
            "fact_coverage_counts": {
                f"overlap_{label_a}": overlap_covered_a,
                f"overlap_{label_b}": overlap_covered_b,
                f"unique_{label_a}": unique_covered_a,
                f"unique_{label_b}": unique_covered_b,
                label_a: total_facts_a,
                label_b: total_facts_b,
            },
        }

    compare_result = {
        "compare_summary_a": args.compare_summary_a,
        "compare_summary_b": args.compare_summary_b,
        "label_a": label_a,
        "label_b": label_b,
        "num_quads_with_nonempty_history": {
            label_a: nonempty_a,
            label_b: nonempty_b,
        },
        "rule_set_sizes": {
            label_a: len(rules_a),
            label_b: len(rules_b),
            "overlap": len(overlap),
            f"unique_{label_a}": len(unique_a),
            f"unique_{label_b}": len(unique_b),
        },
        "coverage_summary": coverage_info,
        "overlap_rules": overlap_rows,
        f"unique_rules_{label_a}": unique_rows_a,
        f"unique_rules_{label_b}": unique_rows_b,
    }

    if args.compare_output:
        output_path = args.compare_output
    else:
        base_a = os.path.splitext(os.path.basename(args.compare_summary_a))[0]
        base_b = os.path.splitext(os.path.basename(args.compare_summary_b))[0]
        output_path = os.path.join(
            os.path.dirname(args.compare_summary_a),
            f"{base_a}_vs_{base_b}_rules_compare.json",
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(compare_result, fout, indent=2)

    print("Saved rules comparison to:", output_path)


def compare_results_correct_predictions(args: argparse.Namespace) -> None:
    correct_a = load_prediction_correctness(args.compare_results_a)
    correct_b = load_prediction_correctness(args.compare_results_b)

    if len(correct_a) != len(correct_b):
        raise ValueError(
            "Results files must have same number of rows for aligned comparison: "
            f"{len(correct_a)} vs {len(correct_b)}"
        )

    label_a = _summary_label(args.compare_results_a, args.compare_results_label_a)
    label_b = _summary_label(args.compare_results_b, args.compare_results_label_b)

    correct_idx_a = {i for i, ok in enumerate(correct_a) if ok}
    correct_idx_b = {i for i, ok in enumerate(correct_b) if ok}

    overlap_idx = correct_idx_a & correct_idx_b
    unique_idx_a = correct_idx_a - correct_idx_b
    unique_idx_b = correct_idx_b - correct_idx_a

    total = len(correct_a)
    result = {
        "compare_results_a": args.compare_results_a,
        "compare_results_b": args.compare_results_b,
        "label_a": label_a,
        "label_b": label_b,
        "num_rows": total,
        "correct_counts": {
            label_a: len(correct_idx_a),
            label_b: len(correct_idx_b),
            "overlap_correct": len(overlap_idx),
            f"unique_correct_{label_a}": len(unique_idx_a),
            f"unique_correct_{label_b}": len(unique_idx_b),
        },
        "correct_ratios": {
            label_a: ((len(correct_idx_a) / total) if total else 0.0),
            label_b: ((len(correct_idx_b) / total) if total else 0.0),
            "overlap_correct": ((len(overlap_idx) / total) if total else 0.0),
            f"unique_correct_{label_a}": ((len(unique_idx_a) / total) if total else 0.0),
            f"unique_correct_{label_b}": ((len(unique_idx_b) / total) if total else 0.0),

            "overlap_correct_a": ((len(overlap_idx) / len(correct_idx_a)) if correct_idx_a else 0.0),
            "overlap_correct_b": ((len(overlap_idx) / len(correct_idx_b)) if correct_idx_b else 0.0),
            f"unique_correct_{label_a}_out_of_correct": ((len(unique_idx_a) / len(correct_idx_a)) if correct_idx_a else 0.0),
            f"unique_correct_{label_b}_out_of_correct": ((len(unique_idx_b) / len(correct_idx_b)) if correct_idx_b else 0.0),
        },
        "overlap_correct_indices": sorted(overlap_idx),
        f"unique_correct_indices_{label_a}": sorted(unique_idx_a),
        f"unique_correct_indices_{label_b}": sorted(unique_idx_b),
    }

    if args.compare_results_output:
        output_path = args.compare_results_output
    else:
        base_a = os.path.splitext(os.path.basename(args.compare_results_a))[0]
        base_b = os.path.splitext(os.path.basename(args.compare_results_b))[0]
        output_path = os.path.join(
            os.path.dirname(args.compare_results_a),
            f"{base_a}_vs_{base_b}_correct_compare.json",
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(result, fout, indent=2)

    print("Saved results correctness comparison to:", output_path)


def parse_time_to_int(raw_time: str) -> float:
    raw_time = raw_time.strip()
    try:
        return float(int(raw_time))
    except ValueError:
        pass

    # Support decimal numeric timestamps from precomputed histories (e.g., 91.0).
    try:
        return float(raw_time)
    except ValueError:
        pass

    # Fallback for date-like formats (e.g., 2014-01-01 from all_facts files).
    try:
        dt = datetime.fromisoformat(raw_time)
        return float(dt.toordinal())
    except ValueError as exc:
        raise ValueError(f"Unsupported timestamp format: {raw_time}") from exc


def read_quads(path: str, split_name: str) -> List[Fact]:
    facts: List[Fact] = []
    with open(path, "r", encoding="utf-8") as f_in:
        for idx, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            sub, rel, obj, raw_time = parts[0], parts[1], parts[2], parts[3]
            facts.append(
                Fact(
                    sub=sub,
                    rel=rel,
                    obj=obj,
                    time_value=parse_time_to_int(raw_time),
                    time_raw=raw_time,
                    source_split=split_name,
                    source_index=idx,
                    is_indirect=False,
                )
            )
    return facts


def parse_precomputed_context(context_text: str) -> Dict[str, object]:
    lines = [line.strip() for line in context_text.split("\n") if line.strip()]
    history: List[Fact] = []
    query_sub: Optional[str] = None
    query_rel: Optional[str] = None
    query_time_raw: Optional[str] = None
    query_time_value: Optional[int] = None

    for idx, line in enumerate(lines):
        history_match = HISTORY_LINE_PATTERN.match(line)
        if history_match:
            t_raw = history_match.group("time").strip()
            history.append(
                Fact(
                    sub=history_match.group("sub").strip(),
                    rel=history_match.group("rel").strip(),
                    obj=history_match.group("obj").strip(),
                    time_value=parse_time_to_int(t_raw),
                    time_raw=t_raw,
                    source_split="precomputed",
                    source_index=idx,
                    is_indirect=False,
                )
            )
            continue

        query_match = QUERY_LINE_PATTERN.match(line)
        if query_match:
            query_time_raw = query_match.group("time").strip()
            query_sub = query_match.group("sub").strip()
            query_rel = query_match.group("rel").strip()
            query_time_value = parse_time_to_int(query_time_raw)

    # Sort descending by time to keep metric semantics identical with raw mode.
    history.sort(key=lambda x: (x.time_value, x.source_index), reverse=True)

    return {
        "history": history,
        "query_sub": query_sub,
        "query_rel": query_rel,
        "query_time_raw": query_time_raw,
        "query_time_value": query_time_value,
    }


def choose_source_splits(split: str, test_source: str) -> List[str]:
    if split == "test" and test_source == "all":
        return ["train", "valid", "test"]
    return [split]


def build_subject_index(
    source_facts: Iterable[Fact], include_indirect: bool
) -> Dict[str, List[Fact]]:
    by_subject: Dict[str, List[Fact]] = defaultdict(list)

    for fact in source_facts:
        # Direct history: subject matches query subject.
        by_subject[fact.sub].append(fact)

        # Indirect history: include the same fact in the object's pool, as-is.
        if include_indirect:
            if fact.obj != fact.sub:
                indirect_fact = Fact(
                    sub=fact.sub,
                    rel=fact.rel,
                    obj=fact.obj,
                    time_value=fact.time_value,
                    time_raw=fact.time_raw,
                    source_split=fact.source_split,
                    source_index=fact.source_index,
                    is_indirect=True,
                )
                by_subject[fact.obj].append(indirect_fact)

    # Sort by time descending (newest first) for easier "closest mention" handling.
    for sub in by_subject:
        by_subject[sub].sort(
            key=lambda x: (x.time_value, x.source_split, x.source_index, x.is_indirect),
            reverse=True,
        )

    return by_subject


def compute_history(history_pool: List[Fact], query_time: int) -> List[Fact]:
    # Retriever-style temporal filtering: strictly earlier only.
    return [fact for fact in history_pool if fact.time_value < query_time]


def to_quad_dict(fact: Fact) -> Dict[str, object]:
    return {
        "sub": fact.sub,
        "rel": fact.rel,
        "obj": fact.obj,
        "time": fact.time_raw,
        "source_split": fact.source_split,
        "source_index": fact.source_index,
        "is_indirect": fact.is_indirect,
    }


def default_output_paths(args: argparse.Namespace) -> Dict[str, str]:
    if args.precomputed_json:
        base_name = os.path.splitext(os.path.basename(args.precomputed_json))[0]
        suffix = "_precomputed"
    else:
        base_name = f"naive_history_{args.split}{'_indirect' if args.include_indirect else ''}"
        suffix = ""

    output_jsonl = (
        args.output_jsonl
        if args.output_jsonl
        else os.path.join(
            "../data/processed_new",
            args.dataset,
            "metadata",
            f"{base_name}{suffix}.jsonl",
        )
    )
    output_summary = (
        args.output_summary
        if args.output_summary
        else os.path.join(
            "../data/processed_new",
            args.dataset,
            "metadata",
            f"{base_name}{suffix}_summary.json",
        )
    )
    return {"jsonl": output_jsonl, "summary": output_summary}


def compute_and_write_stats(
    rows: List[Dict[str, object]],
    args: argparse.Namespace,
    output_jsonl: str,
    output_summary: str,
    source_splits: Optional[List[str]] = None,
    prediction_correctness: Optional[List[bool]] = None,
) -> None:
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    os.makedirs(os.path.dirname(output_summary), exist_ok=True)

    total = 0
    nonempty_history = 0
    with_target = 0
    total_history_len = 0
    total_target_mentions = 0
    closest_mention_index_pool: List[int] = []
    farthest_mention_index_pool: List[int] = []
    within_n_counts = {n: 0 for n in args.recent_n}
    within_n_pos_sums = {n: 0 for n in args.recent_n}
    rule_usage_counts: Dict[str, int] = defaultdict(int)
    rules_stats_available = False
    pred_correct_bucket = _init_prediction_bucket(args.recent_n)
    pred_wrong_bucket = _init_prediction_bucket(args.recent_n)

    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for q_idx, row in enumerate(tqdm(rows, desc="Computing history metadata")):
            total += 1
            history: List[Fact] = row["history"]
            query_obj = row["query_obj"]
            query_rule_ids = row.get("rule_ids")

            if query_rule_ids is not None:
                rules_stats_available = True
                # Count usage once per fact per rule (fact-level percentage semantics).
                for rule_id in set(query_rule_ids):
                    rule_usage_counts[str(rule_id)] += 1

            history_len = len(history)
            total_history_len += history_len
            if history_len > 0:
                nonempty_history += 1

            mention_positions = [
                i for i, fact in enumerate(history) if (fact.obj == query_obj or fact.sub == query_obj)
            ]
            mention_count = len(mention_positions)
            total_target_mentions += mention_count

            has_target = mention_count > 0
            if has_target:
                with_target += 1
                closest_mention_index_pool.append(mention_positions[0])
                farthest_mention_index_pool.append(mention_positions[-1])

            within_n = {}
            for n in args.recent_n:
                hit = has_target and mention_positions[0] < n if history_len > 0 else False
                within_n[str(n)] = hit
                if hit:
                    within_n_counts[n] += 1
                    within_n_pos_sums[n] += mention_positions[0]

            if prediction_correctness is not None:
                is_correct = prediction_correctness[q_idx]
                bucket = pred_correct_bucket if is_correct else pred_wrong_bucket
                bucket["count"] += 1
                bucket["total_target_mentions"] += mention_count
                if has_target:
                    bucket["with_target"] += 1
                    bucket["closest_pool"].append(mention_positions[0])
                    bucket["farthest_pool"].append(mention_positions[-1])
                for n in args.recent_n:
                    if within_n[str(n)]:
                        bucket["within_n_counts"][n] += 1

            meta = {
                "query_index": q_idx,
                "query": {
                    "sub": row.get("query_sub"),
                    "rel": row.get("query_rel"),
                    "obj": query_obj,
                    "time": row.get("query_time_raw"),
                    "split": row.get("query_split", args.split),
                },
                "history_size": history_len,
                "has_target_in_history": has_target,
                "target_mention_count": mention_count,
                "closest_target_mention_index": mention_positions[0] if has_target else None,
                "farthest_target_mention_index": mention_positions[-1] if has_target else None,
                "target_in_last_n": within_n,
            }
            if query_rule_ids is not None:
                meta["rule_ids"] = query_rule_ids
            fout.write(json.dumps(meta) + "\n")

    def safe_avg(values: List[float]) -> Optional[float]:
        return (sum(values) / len(values)) if values else None

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "include_indirect": args.include_indirect,
        "precomputed_json": args.precomputed_json,
        "test_source": args.test_source,
        "source_splits": source_splits,
        "num_target_quads": total,
        "num_quads_with_nonempty_history": nonempty_history,
        "num_quads_with_target_in_history": with_target,
        "ratio_quads_with_target_in_history": (with_target / total) if total else 0.0,
        "avg_history_size": (total_history_len / total) if total else 0.0,
        "avg_target_mentions_per_quad": (total_target_mentions / total) if total else 0.0,
        "avg_target_mentions_per_quad_when_present": (
            (total_target_mentions / with_target) if with_target else 0.0
        ),
        "avg_closest_target_mention_index_when_present": safe_avg(closest_mention_index_pool),
        "avg_farthest_target_mention_index_when_present": safe_avg(farthest_mention_index_pool),
        "target_in_last_n_counts": {str(k): within_n_counts[k] for k in sorted(within_n_counts.keys())},
        "target_in_last_n_avg_closest_index": {
            str(k): ((within_n_pos_sums[k] / within_n_counts[k]) if within_n_counts[k] else None)
            for k in sorted(within_n_counts.keys())
        },
        "target_in_last_n_ratios": {
            str(k): ((within_n_counts[k] / total) if total else 0.0)
            for k in sorted(within_n_counts.keys())
        },
        "output_jsonl": output_jsonl,
    }

    if prediction_correctness is not None:
        num_correct = int(pred_correct_bucket["count"])
        summary["prediction_metrics"] = {
            "results_jsonl": args.results_jsonl,
            "num_predictions": len(prediction_correctness),
            "num_correct": num_correct,
            "num_wrong": int(pred_wrong_bucket["count"]),
            "prediction_accuracy": ((num_correct / len(prediction_correctness)) if prediction_correctness else 0.0),
            "correct_prediction_history_stats": _prediction_bucket_summary(pred_correct_bucket, args.recent_n),
            "wrong_prediction_history_stats": _prediction_bucket_summary(pred_wrong_bucket, args.recent_n),
        }

    if rules_stats_available:
        summary["num_unique_rules_used"] = len(rule_usage_counts)
        usage_pct_items = [
            (rid, ((rule_usage_counts[rid] / total) * 100.0 if total else 0.0))
            for rid in rule_usage_counts.keys()
        ]
        usage_pct_items.sort(key=lambda x: (-x[1], x[0]))
        summary["rules_usage_pct_among_all_facts"] = {rid: pct for rid, pct in usage_pct_items}

    with open(output_summary, "w", encoding="utf-8") as fsum:
        json.dump(summary, fsum, indent=2)

    print("Saved per-quad metadata to:", output_jsonl)
    print("Saved summary to:", output_summary)


def main() -> None:
    args = parse_args()

    if args.compare_results_a or args.compare_results_b:
        if not args.compare_results_a or not args.compare_results_b:
            raise ValueError("Both --compare_results_a and --compare_results_b are required.")
        compare_results_correct_predictions(args)
        return

    if args.compare_summary_a or args.compare_summary_b:
        if not args.compare_summary_a or not args.compare_summary_b:
            raise ValueError("Both --compare_summary_a and --compare_summary_b are required.")
        compare_rule_usage_summaries(args)
        return

    if not args.dataset:
        raise ValueError("--dataset is required unless compare mode is used.")

    outputs = default_output_paths(args)
    output_jsonl = outputs["jsonl"]
    output_summary = outputs["summary"]
    prediction_correctness: Optional[List[bool]] = None

    if args.results_jsonl:
        prediction_correctness = load_prediction_correctness(args.results_jsonl, args.max_quads)

    if args.precomputed_json:
        with open(args.precomputed_json, "r", encoding="utf-8") as fin:
            precomputed_rows = json.load(fin)

        if args.max_quads is not None:
            precomputed_rows = precomputed_rows[: args.max_quads]

        rows: List[Dict[str, object]] = []
        for item in precomputed_rows:
            parsed = parse_precomputed_context(item.get("context", ""))
            rows.append(
                {
                    "history": parsed["history"],
                    "query_sub": parsed["query_sub"],
                    "query_rel": parsed["query_rel"],
                    "query_time_raw": parsed["query_time_raw"],
                    "query_time_value": parsed["query_time_value"],
                    "query_obj": item.get("target"),
                    "rule_ids": item.get("rule_ids", None),
                    "query_split": args.split,
                }
            )

        if prediction_correctness is not None and len(prediction_correctness) != len(rows):
            raise ValueError(
                "Mismatch between rows and predictions: "
                f"{len(rows)} rows vs {len(prediction_correctness)} predictions from {args.results_jsonl}"
            )

        compute_and_write_stats(
            rows=rows,
            args=args,
            output_jsonl=output_jsonl,
            output_summary=output_summary,
            source_splits=None,
            prediction_correctness=prediction_correctness,
        )
        return

    dataset_dir = os.path.join(args.base_data_dir, args.dataset)
    source_splits = choose_source_splits(args.split, args.test_source)

    target_path = os.path.join(dataset_dir, f"{args.split}.txt")
    target_facts = read_quads(target_path, args.split)
    if args.max_quads is not None:
        target_facts = target_facts[: args.max_quads]

    source_facts: List[Fact] = []
    for split_name in source_splits:
        split_path = os.path.join(dataset_dir, f"{split_name}.txt")
        source_facts.extend(read_quads(split_path, split_name))

    by_subject = build_subject_index(source_facts, args.include_indirect)

    rows = []
    for query in target_facts:
        history_pool = by_subject.get(query.sub, [])
        history = compute_history(history_pool, query.time_value)
        rows.append(
            {
                "history": history,
                "query_sub": query.sub,
                "query_rel": query.rel,
                "query_time_raw": query.time_raw,
                "query_time_value": query.time_value,
                "query_obj": query.obj,
                "query_split": args.split,
            }
        )

    if prediction_correctness is not None and len(prediction_correctness) != len(rows):
        raise ValueError(
            "Mismatch between rows and predictions: "
            f"{len(rows)} rows vs {len(prediction_correctness)} predictions from {args.results_jsonl}"
        )

    compute_and_write_stats(
        rows=rows,
        args=args,
        output_jsonl=output_jsonl,
        output_summary=output_summary,
        source_splits=source_splits,
        prediction_correctness=prediction_correctness,
    )


if __name__ == "__main__":
    main()
