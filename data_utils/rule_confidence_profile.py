"""Per-relation top-k confidence profile of a mined rule pool.

Choosing ``--top_k_rules`` for the retrieval filters (apply_history_filters.py)
is otherwise guesswork. ``retrieve.py`` looks up rules by the query's
``head_rel``, so the top-k rules kept for a query are exactly the top-k
highest-confidence rules of that relation. This script therefore reports, for
each relation and each candidate k, the confidence statistics of that
relation's top-k rules — i.e. how confidence decays as k grows — which is the
direct evidence for picking k.

It is a pure analysis of ``common_rule_pool.json``: no history file or split is
involved. Confidence is read for one mining algorithm via
``--rules_algorithm`` (see conf_stats.load_rule_conf_map).

Example
-------
    python rule_confidence_profile.py \\
        --dataset icews14 \\
        --rule_pool ../data/processed_new/icews14/output/icews14/common_rule_pool.json \\
        --rules_algorithm exhaustive \\
        --top_k 5 10 20 50 100
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

from conf_stats import conf_stats, load_rule_conf_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-relation top-k confidence profile of a rule pool (to help choose top_k_rules)."
    )
    parser.add_argument(
        "--rule_pool",
        required=True,
        type=str,
        help="Path to common_rule_pool.json (flat dict keyed by rule_id).",
    )
    parser.add_argument(
        "--rules_algorithm",
        required=True,
        type=str,
        help="Mining algorithm whose conf to use, e.g. exhaustive (selects algorithm_stats[<algo>].conf).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name, used only to locate relation2id when --relation2id is not given.",
    )
    parser.add_argument(
        "--top_k",
        nargs="+",
        type=int,
        default=[5, 10, 20, 50, 100],
        help="Candidate k values to profile (top-k highest-confidence rules per relation).",
    )
    parser.add_argument(
        "--relation2id",
        type=str,
        default=None,
        help="Path to relation2id.json. Defaults to ../data/original/<dataset>/relation2id.json.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Defaults next to the rule pool as <pool>_conf_profile_<algo>.json.",
    )
    return parser.parse_args()


def load_id_to_relation(path: Optional[str]) -> Dict[int, str]:
    """Build an ``{id: relation_name}`` map from a relation2id.json (name->id)."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fin:
        name_to_id = json.load(fin)
    return {int(v): str(k) for k, v in name_to_id.items()}


def profile_for_confs(
    sorted_desc: List[float], top_k: List[int]
) -> Dict[str, Dict[str, object]]:
    """Confidence stats for the top-k slice at each k (input sorted descending)."""
    num_rules = len(sorted_desc)
    profiles: Dict[str, Dict[str, object]] = {}
    for k in top_k:
        slice_confs = sorted_desc[:k]
        entry = conf_stats(slice_confs)
        entry["coverage"] = round(min(k, num_rules) / num_rules, 4) if num_rules else 0.0
        profiles[str(k)] = entry
    return profiles


def main() -> None:
    args = parse_args()

    relation2id_path = args.relation2id
    if relation2id_path is None and args.dataset:
        relation2id_path = os.path.join(
            "../data/original", args.dataset, "relation2id.json"
        )
    id_to_relation = load_id_to_relation(relation2id_path)

    rule_to_conf, rule_to_head_rel = load_rule_conf_map(args.rule_pool, args.rules_algorithm)

    # Group confidences by head relation.
    by_relation: Dict[int, List[float]] = defaultdict(list)
    all_confs: List[float] = []
    for rule_id, conf in rule_to_conf.items():
        head_rel = rule_to_head_rel.get(rule_id)
        if head_rel is None:
            continue
        by_relation[head_rel].append(conf)
        all_confs.append(conf)

    top_k = sorted(set(args.top_k))

    relations_out: Dict[str, object] = {}
    for head_rel in sorted(by_relation.keys()):
        confs_desc = sorted(by_relation[head_rel], reverse=True)
        relations_out[str(head_rel)] = {
            "name": id_to_relation.get(head_rel),
            "num_rules": len(confs_desc),
            "profiles": profile_for_confs(confs_desc, top_k),
        }

    result = {
        "rule_pool": args.rule_pool,
        "rules_algorithm": args.rules_algorithm,
        "top_k": top_k,
        "num_rules_total": len(all_confs),
        "num_relations": len(by_relation),
        "all_relations": {
            "num_rules": len(all_confs),
            "profiles": profile_for_confs(sorted(all_confs, reverse=True), top_k),
        },
        "relations": relations_out,
    }

    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(args.rule_pool)[0]
        output_path = f"{base}_conf_profile_{args.rules_algorithm}.json"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(result, fout, indent=2)

    # Compact stdout table: avg conf at each k, per relation (the artifact for choosing k).
    header = ["rel", "name", "#rules"] + [f"avg@{k}" for k in top_k]
    print("\t".join(header))
    all_profiles = result["all_relations"]["profiles"]
    print(
        "\t".join(
            ["ALL", "all_relations", str(len(all_confs))]
            + [f"{all_profiles[str(k)]['avg']}" for k in top_k]
        )
    )
    for head_rel in sorted(by_relation.keys()):
        rel_entry = relations_out[str(head_rel)]
        name = (rel_entry["name"] or "")[:30]
        profiles = rel_entry["profiles"]
        row = [str(head_rel), name, str(rel_entry["num_rules"])] + [
            f"{profiles[str(k)]['avg']}" for k in top_k
        ]
        print("\t".join(row))

    print("\nSaved confidence profile to:", output_path)


if __name__ == "__main__":
    main()