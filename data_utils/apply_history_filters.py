"""
Derive a filtered, rendered history dataset from a base retrieval metadata file.

Retrieval decides which facts a query can see; this script decides how many of
them survive and how they are written.  The split is not a convenience — the
metadata stores fact indices rather than rendered text, so every option here is
recoverable from one base, and none of them needs retrieval to run again.

Retrieval-time, and therefore fixed in the base: the rule bank (--mining),
--inverse_body_object_match, --early_stop_at_num_facts.  Each changes which
facts are retrieved at all.  Everything else belongs here.

Because the metadata holds indices, the source facts under
data/original/<dataset>/ must be present when this script runs.

  1. Generate the base once per dataset and rule bank.  --save_metadata
     refuses every filter and rendering flag, and writes no history_facts:

       python retrieve.py -d icews14 -m gtkg --length_1_only \\
           --inverse_body_object_match --save_metadata

  2. Derive as many variants as you like:

       python apply_history_filters.py \\
           --metadata ../data/processed_new/icews14/gtkg/test_inv/metadata/history_metadata_icews14_test.jsonl \\
           --output_dir ../data/processed_new/icews14/gtkg/test_inv_n50_k10_idn/ \\
           --top_k_rules 10 --num_facts 50 --index_target

       python apply_history_filters.py \\
           --metadata ... \\
           --output_dir ... \\
           --confidence_threshold 0.5 --model_type t5 --num_facts 30

     --index_target has to match the flag passed to create_json_train.py, or
     the history and the target it is trained to predict use different formats.

  3. Or simulate a build to see if a filter combo is worth it, writing only
     the stats (no dataset files):

       python apply_history_filters.py \\
           --metadata ... --output_dir ... \\
           --confidence_threshold 0.5 --stats_only
"""

import argparse
import json
import logging
import os
import shutil

from basic import read_json
from conf_stats import conf_stats
from TLR import fact_to_line, time_period


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parser():
    p = argparse.ArgumentParser(
        description="Apply num_facts / top_k_rules / confidence_threshold filters to a "
                    "saved history metadata file without re-running retrieval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--metadata", "-meta", required=True,
        help="Path to history_metadata_<dataset>_<split>.jsonl produced by retrieve.py --save_metadata.",
    )
    p.add_argument(
        "--output_dir", "-o", required=True,
        help="Directory to write filtered history files into (will be created if absent).",
    )
    # Filters (all optional; omit to keep the full unfiltered history)
    p.add_argument(
        "--num_facts", "-n", default=None, type=int,
        help="Maximum number of history facts to keep per sample (most-recent first).",
    )
    p.add_argument(
        "--top_k_rules", "-k", default=None, type=int,
        help="Keep only the top-k highest-confidence fired rules; collect their facts, then apply num_facts.",
    )
    p.add_argument(
        "--confidence_threshold", "-ct", default=None, type=float,
        help="Split facts into high-confidence (>= threshold) and low-confidence (< threshold) groups.",
    )
    p.add_argument(
        "--model_type", "-mt", default="t5", choices=["t5", "llm"],
        help="How to format the confidence split: 't5' concatenates the groups, "
             "'llm' labels them with text headings.",
    )
    p.add_argument(
        "--index_target", "-idn", action="store_true", default=False,
        help="Prefix every history object with its entity id, as '18.Thailand'. "
             "Subjects and relations stay names. Match this to the --index_target "
             "passed to create_json_train.py so context and target agree.",
    )
    p.add_argument(
        "--stats_only", action="store_true", default=False,
        help="Simulate the build: compute and write filter_stats.json only, without "
             "writing the history_facts/test_answers dataset files. Use to gauge whether "
             "a filter combination is worth materializing.",
    )
    return vars(p.parse_args())


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(log_path):
    # force=True: basicConfig is a no-op when the root logger already has
    # handlers, so without it a second run in the same interpreter session keeps
    # logging to the first run's file.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filter logic (mirrors TLR.py build_tl without requiring the original data)
# ---------------------------------------------------------------------------

def _apply_filters(sample, num_facts, top_k_rules, confidence_threshold, model_type, render):
    """
    Reconstruct history text for one sample by applying the requested filters
    to the pre-computed metadata.

    Parameters
    ----------
    sample : dict  — one metadata line: sample_idx, query_line, fired_rules
    num_facts : int or None
    top_k_rules : int or None
    confidence_threshold : float or None
    model_type : "t5" or "llm"
    render : callable(int) -> str  — formats one fact index as a history line

    Returns
    -------
    history_text : str   — formatted history (without the query line)
    surviving_rule_ids : list[int]
    conf_info : dict     — {"above": [...], "below": [...]} confidences of the
                           surviving facts (below is empty unless threshold is set)
    """
    fired_rules = sample["fired_rules"]  # in the order retrieval produced them

    if not fired_rules:
        return "", [], {"above": [], "below": []}

    # ------------------------------------------------------------------
    # top_k_rules: keep only the K highest-confidence rules.
    #
    # This mirrors the filter order in TLR.Retriever.build_tl: top_k_rules,
    # then union the surviving rules' facts, then newest-first, then num_facts.
    # The two are hand-synchronised — keep them in step, or re-filtering saved
    # metadata stops reproducing what retrieval produced.
    #
    # Python's sort is stable, so sorting the stored build order by confidence
    # reproduces build_tl's own sorted slice exactly, while the unfiltered case
    # keeps build order — which is what build_tl uses when top_k_rules is unset.
    # Sorting unconditionally would reorder the rule ids of every unfiltered run.
    # ------------------------------------------------------------------
    rules_in_scope = fired_rules
    if top_k_rules is not None:
        rules_in_scope = sorted(fired_rules, key=lambda r: r["conf"],
                                reverse=True)[:top_k_rules]

    # ------------------------------------------------------------------
    # The fact list is not stored: it is the union of the surviving rules'
    # fact_indices, each fact's confidence the maximum over the rules that
    # retrieved it, and its rule ids those rules in scope order.
    # ------------------------------------------------------------------
    fact_to_conf = {}
    fact_to_rule_ids = {}
    for rule in rules_in_scope:
        c = rule["conf"]
        r = rule["rule_id"]
        for fi in rule["fact_indices"]:
            if fi not in fact_to_conf or c > fact_to_conf[fi]:
                fact_to_conf[fi] = c
            if r is not None:
                fact_to_rule_ids.setdefault(fi, []).append(r)

    # Newest-first: a higher index into all_facts is a later fact.
    facts = sorted(fact_to_conf.keys(), reverse=True)

    # ------------------------------------------------------------------
    # num_facts: keep at most N facts (newest-first, so take the head)
    # ------------------------------------------------------------------
    if num_facts is not None:
        facts = facts[:num_facts]

    # ------------------------------------------------------------------
    # Surviving rule IDs: deduplicated, in order of appearance
    # ------------------------------------------------------------------
    seen_rule_ids: set = set()
    surviving_rule_ids = []
    for fi in facts:
        for rid in fact_to_rule_ids.get(fi, []):
            if rid not in seen_rule_ids:
                surviving_rule_ids.append(rid)
                seen_rule_ids.add(rid)

    if not facts:
        return "", surviving_rule_ids, {"above": [], "below": []}

    # ------------------------------------------------------------------
    # confidence_threshold split
    # ------------------------------------------------------------------
    if confidence_threshold is not None:
        above = [fi for fi in facts if fact_to_conf[fi] >= confidence_threshold]
        below = [fi for fi in facts if fact_to_conf[fi] < confidence_threshold]
    else:
        above = facts
        below = []

    conf_info = {
        "above": [fact_to_conf[fi] for fi in above],
        "below": [fact_to_conf[fi] for fi in below],
    }

    # ------------------------------------------------------------------
    # Format history lines (oldest-first = reversed from newest-first storage)
    # ------------------------------------------------------------------
    def to_lines(fact_list):
        return [render(fi) for fi in reversed(fact_list)]

    if confidence_threshold is not None and below:
        if model_type == "t5":
            # Specific (low-conf) facts first so general (high-conf) facts sit
            # immediately before the query
            lines = to_lines(below) + to_lines(above)
        else:
            lines = []
            if below:
                lines.append("[Specific context - lower confidence rules:]\n")
                lines.extend(to_lines(below))
            if above:
                lines.append("[General context - higher confidence rules:]\n")
                lines.extend(to_lines(above))
    else:
        lines = to_lines(above)

    return "".join(lines), surviving_rule_ids, conf_info


# ---------------------------------------------------------------------------
# Aggregate statistics helper
# ---------------------------------------------------------------------------

def _aggregate(values):
    if not values:
        return {"count": 0, "min": None, "max": None, "avg": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "avg": round(sum(values) / len(values), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parser()

    meta_path = args["metadata"]
    stats_only = args["stats_only"]

    # --- Read the header (first JSONL line) ---
    with open(meta_path, encoding="utf-8") as f:
        header = json.loads(f.readline())
    dataset = header["dataset"]

    model_type = args["model_type"]

    # The metadata stores fact indices, not rendered text, so the history line
    # is built here with the same formatter retrieval uses, over the same files.
    original_dir = header.get("original_data_dir") or f"../data/original/{dataset}/"
    try:
        with open(os.path.join(original_dir, "all_facts.txt"), encoding="utf-8") as f:
            all_facts = f.readlines()
        times_id = read_json(os.path.join(original_dir, "ts2id.json"))
        # Rendering is decided here, not by the retrieval: the metadata holds
        # fact indices, so the same base serves both formats.
        index_target = args["index_target"]
        entities = read_json(os.path.join(original_dir, "entity2id.json")) if index_target else {}
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"Cannot read the source facts under {original_dir} ({exc}). The "
            "metadata holds fact indices into all_facts.txt, so that directory "
            "must be present to render history text."
        )
    period = time_period(dataset)
    _line_cache = {}

    def render(fi):
        """History line for one fact index, memoised across samples.

        Args:
            fi: index into all_facts.

        Returns:
            str: the rendered history line.
        """
        # Facts recur heavily across samples; the cache is bounded by len(all_facts).
        line = _line_cache.get(fi)
        if line is None:
            line = fact_to_line(all_facts[fi], times_id, period, entities,
                                index_target)
            _line_cache[fi] = line
        return line

    # --- Setup output directories ---
    output_dir   = args["output_dir"].rstrip("/\\")
    history_dir  = output_dir + "/history_facts/"
    answers_dir  = output_dir + "/test_answers/"
    # In stats_only mode we still need output_dir for filter_stats.json, but we
    # skip the dataset subdirectories that would hold the materialized history.
    os.makedirs(output_dir, exist_ok=True)
    if not stats_only:
        os.makedirs(history_dir, exist_ok=True)
        os.makedirs(answers_dir, exist_ok=True)
    os.makedirs("../logs/", exist_ok=True)

    split = header.get("split", "unknown")
    log_name = (
        f"apply_filters_{dataset}_{split}"
        f"{'_top' + str(args['top_k_rules']) + 'rules' if args['top_k_rules'] else ''}"
        f"{'_n' + str(args['num_facts']) if args['num_facts'] else ''}"
        f"{'_thresh' + str(args['confidence_threshold']) if args['confidence_threshold'] is not None else ''}"
        f"{'_statsonly' if stats_only else ''}"
        ".log"
    )
    logger = _setup_logging("../logs/" + log_name)

    logger.info("Loaded metadata: %s", meta_path)
    logger.info("Dataset=%s  Split=%s", dataset, split)
    if stats_only:
        logger.info("stats_only mode — simulating build, no dataset files will be written.")
    logger.info(
        "Filters — num_facts=%s  top_k_rules=%s  confidence_threshold=%s  model_type=%s",
        args["num_facts"], args["top_k_rules"], args["confidence_threshold"], model_type,
    )
    logger.info("Rendering — index_target=%s", index_target)

    # --- Stream samples: read one line, filter, write outputs immediately ---
    # Peak RAM stays at ~one sample regardless of metadata file size. The three
    # output files are written line-by-line so nothing is accumulated either.
    # A third file, "<base>_idx_fine_tune_all.txt", used to be written here with
    # exactly the same bytes as the plain .txt. Nothing ever read it, so it is
    # no longer produced.
    base_name  = history_dir + "history_facts_" + dataset
    size_stats = []
    n_samples  = 0
    # Confidence of every surviving fact across all samples, split by the
    # threshold (below stays empty when no confidence_threshold is given).
    above_confs = []
    below_confs = []

    if stats_only:
        # No dataset files; just stream the metadata and accumulate stats.
        with open(meta_path, encoding="utf-8") as mf:
            next(mf, None)  # skip header line
            for line in mf:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)

                history_text, surviving_rule_ids, conf_info = _apply_filters(
                    sample,
                    num_facts=args["num_facts"],
                    top_k_rules=args["top_k_rules"],
                    confidence_threshold=args["confidence_threshold"],
                    model_type=model_type,
                    render=render,
                )
                size_stats.append(history_text.count("\n") if history_text else 0)
                above_confs.extend(conf_info["above"])
                below_confs.extend(conf_info["below"])
                n_samples += 1
    else:
        with open(meta_path, encoding="utf-8") as mf, \
             open(base_name + ".txt", "w", encoding="utf-8") as f_txt, \
             open(base_name + "_rule_ids.txt", "w", encoding="utf-8") as f_rids:
            next(mf, None)  # skip header line
            for line in mf:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)

                history_text, surviving_rule_ids, conf_info = _apply_filters(
                    sample,
                    num_facts=args["num_facts"],
                    top_k_rules=args["top_k_rules"],
                    confidence_threshold=args["confidence_threshold"],
                    model_type=model_type,
                    render=render,
                )
                full_text = history_text + sample["query_line"]
                f_txt.write(full_text + "\n")
                f_rids.write(json.dumps(surviving_rule_ids) + "\n")

                # Count facts in the filtered output (history lines before the query)
                size_stats.append(history_text.count("\n") if history_text else 0)
                above_confs.extend(conf_info["above"])
                below_confs.extend(conf_info["below"])
                n_samples += 1

    logger.info(
        "Filter applied — %d samples.  Filtered history size stats: %s",
        n_samples,
        json.dumps(_aggregate(size_stats), indent=2),
    )

    # ------------------------------------------------------------------
    # Confidence statistics of the final (post-filter) history. In threshold
    # mode the above/below split is reported separately (the two averages);
    # otherwise everything sits in "above".
    # ------------------------------------------------------------------
    overall_confs = above_confs + below_confs
    total_confs = len(overall_confs)
    filter_stats = {
        "dataset": dataset,
        "split": split,
        "metadata": meta_path,
        "stats_only": stats_only,
        "filters": {
            "num_facts": args["num_facts"],
            "top_k_rules": args["top_k_rules"],
            "confidence_threshold": args["confidence_threshold"],
            "model_type": model_type,
        },
        # Not a filter: it changes how each surviving fact is written. Recorded so
        # a derived directory documents its own format instead of relying on the
        # base it came from, whose metadata is format-agnostic.
        "index_target": index_target,
        "n_samples": n_samples,
        "history_size": _aggregate(size_stats),
        "confidence_overall": conf_stats(overall_confs),
    }
    if args["confidence_threshold"] is not None:
        filter_stats["confidence_above"] = conf_stats(above_confs)
        filter_stats["confidence_below"] = conf_stats(below_confs)
        filter_stats["fact_count_by_bucket"] = {
            "above": len(above_confs),
            "below": len(below_confs),
            "above_ratio": round(len(above_confs) / total_confs, 4) if total_confs else 0.0,
            "below_ratio": round(len(below_confs) / total_confs, 4) if total_confs else 0.0,
        }

    filter_stats_path = output_dir + "/filter_stats.json"
    with open(filter_stats_path, "w", encoding="utf-8") as fstats:
        json.dump(filter_stats, fstats, indent=2)
    logger.info("Filtered confidence stats: %s", json.dumps(
        {k: filter_stats[k] for k in filter_stats if k.startswith("confidence")}, indent=2))
    logger.info("Filter stats written to  : %s", filter_stats_path)

    # --- Copy test_answers from base retrieval (skipped in stats_only mode) ---
    if not stats_only:
        # The answers sit beside the metadata, in the base variant's own
        # test_answers/. Prefer that over the header's recorded path, which is
        # absolute and machine-specific: it breaks whenever the repository moves
        # or a base directory is renamed, and the failure is a warning that only
        # surfaces later as a missing file in create_json_train.py.
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(meta_path)))
        local_answers = os.path.join(base_dir, "test_answers",
                                     f"test_answers_{dataset}.txt")
        base_answers = (local_answers if os.path.isfile(local_answers)
                        else header.get("base_answers_path"))
        if base_answers and os.path.isfile(base_answers):
            dest_answers = answers_dir + "test_answers_" + dataset + ".txt"
            shutil.copy(base_answers, dest_answers)
            logger.info("Copied test_answers from : %s", base_answers)
            logger.info("                      to : %s", dest_answers)
        else:
            logger.warning(
                "No test_answers found beside the metadata (%s) or at the path "
                "recorded in its header (%s). Copy test_answers manually before "
                "running create_json_train.py.",
                local_answers, header.get("base_answers_path"),
            )

        logger.info("History text written to  : %s.txt", base_name)
        logger.info("Rule ids written to      : %s_rule_ids.txt", base_name)
    logger.info("Done.")
