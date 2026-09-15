"""Build recency-history metadata, the retrieval baseline with no rule bank.

Where `retrieve.py` selects a subject's facts by firing mined rules, this keeps
the subject's facts and lets recency alone decide, which is the ISI-style
`standard` baseline. It writes the same metadata that `retrieve.py
--save_metadata` writes, so `apply_history_filters.py` derives every variant
from it unchanged: the cap, the `id.Name` rendering and the model formats are
all applied there, exactly as for a mined bank.

The metadata format stores facts per rule. There is no rule here, so every
candidate goes in one entry with a null `rule_id` and confidence 1.0. That
entry is what `apply_history_filters.py` sorts newest-first and caps, so the
`--num_facts` cap has the same meaning as on the mined path and the two are
comparable at equal budget. Its `--top_k_rules` and `--confidence_threshold`
select among rules and so do nothing here; a run that passes them is rejected
rather than silently producing a base that looks filtered and is not.

Shares no code path with `TLR.Retriever` and writes under its own mining
directory, so no mined variant can be affected by running this.

Usage, from data_utils/:

    python standard_retrieve.py --dataset icews14 --split test --inverse
"""

import argparse
import json
import logging
import os
from typing import Dict, List

import numpy

from basic import read_json, read_txt_as_list
from id_words import convert_dataset
from TLR import time_period

logger = logging.getLogger(__name__)


def parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["icews14", "icews18"], required=True)
    p.add_argument("--split", choices=["train", "valid", "test"], default="test",
                   help="Split to build queries for. History always comes from "
                        "all_facts.txt, filtered to strictly earlier facts.")
    p.add_argument("--inverse", action="store_true",
                   help="Also keep facts where the query subject is the object. "
                        "Matches --inverse_body_object_match on the mined path, "
                        "and names the output directory '_inv' the same way. The "
                        "fact is stored as it stands, so it renders with the "
                        "other entity as subject.")
    p.add_argument("--base_data_dir", default="../data/original",
                   help="Holds <dataset>/ with the split files, all_facts.txt "
                        "and the id maps.")
    p.add_argument("--path_save", default="../data/processed_new",
                   help="Root for the output; the variant lands in "
                        "<path_save>/<dataset>/standard/<split>[_inv]/.")
    return p.parse_args()


def column_indices(cache: Dict[str, numpy.ndarray], column: numpy.ndarray,
                   value: str) -> numpy.ndarray:
    """Row indices where a fact column equals a value, memoised.

    Args:
        cache: dict memoising results for this column.
        column: the fact column to compare against.
        value: the value to match.

    Returns:
        numpy.ndarray: sorted row indices; empty when the value never occurs.
    """
    # `is None`, not truthiness: a cached empty array is falsy and would be
    # recomputed on every lookup.
    cached = cache.get(value)
    if cached is None:
        cached = numpy.where(column == value)[0]
        cache[value] = cached
    return cached


def main() -> None:
    args = parser()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    workspace = os.path.join(args.base_data_dir, args.dataset) + os.sep
    period = time_period(args.dataset)

    for name in (args.split + ".txt", "all_facts.txt", "ts2id.json",
                 "relation2id.json", "entity2id.json"):
        if not os.path.isfile(workspace + name):
            raise SystemExit(f"Missing source file: {workspace + name}")

    times_id = read_json(workspace + "ts2id.json")
    # The split files hold ids; the fact columns hold names. convert_dataset is
    # what retrieve.py uses for the same purpose, so the query rows here are
    # byte-identical to the mined path's and the two variants stay row-aligned.
    # period=1, deliberately: this converts timestamp ids to the date strings
    # all_facts.txt carries, where the id needs no scaling. time_period's 24 is
    # a different quantity — the divisor that turns an id into a day index when
    # a line is rendered — and applying it here would look up an id that does
    # not exist. retrieve.py makes the same distinction.
    queries = convert_dataset(read_txt_as_list(workspace + args.split + ".txt"),
                              workspace, period=1)

    with open(workspace + "all_facts.txt", encoding="utf-8") as handle:
        all_facts = handle.readlines()
    rows = [row.strip().split("\t") for row in all_facts]
    col_sub = numpy.array([r[0] for r in rows])
    col_obj = numpy.array([r[2] for r in rows])
    col_time = numpy.array([r[3] for r in rows])
    logger.info("%d facts, %d %s queries", len(all_facts), len(queries), args.split)

    suffix = args.split + ("_inv" if args.inverse else "")
    out_dir = os.path.join(args.path_save, args.dataset, "standard", suffix)
    meta_dir = os.path.join(out_dir, "metadata")
    answers_dir = os.path.join(out_dir, "test_answers")
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(answers_dir, exist_ok=True)
    meta_path = os.path.join(meta_dir,
                             f"history_metadata_{args.dataset}_{args.split}.jsonl")

    # convert_dataset returns rows without a trailing newline, so writelines
    # would concatenate the split onto one line and the downstream row count
    # would be 1.
    answers_path = os.path.join(answers_dir, f"test_answers_{args.dataset}.txt")
    with open(answers_path, "w", encoding="utf-8") as handle:
        handle.writelines(row.rstrip("\n") + "\n" for row in queries)

    sub_cache: Dict[str, numpy.ndarray] = {}
    obj_cache: Dict[str, numpy.ndarray] = {}
    sizes: List[int] = []

    with open(meta_path, "w", encoding="utf-8") as out:
        out.write(json.dumps({
            "dataset": args.dataset,
            "inverse_body_object_match": args.inverse,
            "mining": "standard",
            "split": args.split,
            "original_data_dir": os.path.abspath(workspace),
            "base_answers_path": os.path.abspath(answers_path),
        }) + "\n")

        for i, row in enumerate(queries):
            test_sub, test_rel, _, test_time, _ = row.strip().split("\t")
            # all_facts is non-decreasing in the time column, so the rows
            # strictly earlier than the query are exactly [0, cut). The same
            # reasoning and the same strict bound as TLR.tlogic_prepro, so the
            # target quad cannot enter its own history.
            cut = int(numpy.searchsorted(col_time, test_time, side="left"))
            indices = column_indices(sub_cache, col_sub, test_sub)
            keep = indices[indices < cut]
            if args.inverse:
                other = column_indices(obj_cache, col_obj, test_sub)
                keep = numpy.union1d(keep, other[other < cut])

            query_line = (f"{int(times_id[test_time]) // period}: "
                          f"[{test_sub}, {test_rel},\n")
            sizes.append(len(keep))
            out.write(json.dumps({
                "sample_idx": i,
                "query_line": query_line,
                # One entry holding every candidate: there is no rule to
                # attribute facts to, and apply_history_filters takes the union
                # over entries, so a single one leaves recency in sole charge.
                "fired_rules": [{
                    "conf": 1.0,
                    "rule_id": None,
                    "fact_indices": [int(x) for x in keep],
                }] if len(keep) else [],
            }) + "\n")

    sizes.sort()
    logger.info("Candidate facts per query: mean %.1f, median %d, max %d, empty %d",
                sum(sizes) / len(sizes), sizes[len(sizes) // 2], sizes[-1],
                sum(1 for s in sizes if s == 0))
    logger.info("Metadata: %s", meta_path)
    logger.info("Answers : %s", answers_path)


if __name__ == "__main__":
    main()
