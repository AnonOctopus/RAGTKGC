"""Score saved results files, reporting raw and time-aware-filtered Hits@k.

The ranking helpers are shared with run_hf.py so the live figures and the
reported ones cannot drift apart.
"""

import argparse
import json
import os

from utils import HitsMetric, load_true_objects, update_metric


class _ScoringArgs:
    """Minimal stand-in for run_hf's parsed args, for update_metric."""

    def __init__(self, dataset, verbose=False):
        self.dataset = dataset
        self.verbose = verbose


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default = 'icews14',type = str)
    parser.add_argument("--file_name", default = 'all', type = str)
    parser.add_argument(
        "--expected_rows",
        default=None,
        type=int,
        help="Expected rows per results file. Read from <dataset>/test.txt when omitted.",
    )

    args = parser.parse_args()
    return args


def expected_rows(dataset, base_data_dir='./data/original'):
    """Rows a complete results file should have.

    Args:
        dataset: dataset name, e.g. "icews14".
        base_data_dir: folder containing <dataset>/test.txt.

    Returns:
        int: expected row count.

    Raises:
        FileNotFoundError: the split file is missing.
    """
    path = os.path.join(base_data_dir, dataset, 'test.txt')
    with open(path, encoding='utf-8') as f:
        n = sum(1 for line in f if line.strip())
    # run_hf.py evaluates only the first 10,000 ICEWS18 test quads.
    return min(n, 10_000) if dataset == 'icews18' else n


if __name__ == "__main__":
    
    args = parser()
    
    path = f"./results/{args.dataset}"
    dir_list = os.listdir(path)

    

    if args.file_name != 'all':
      results_files = [args.file_name]
    else:
      # _runs.jsonl is the run manifest written by run_hf.py, not a results file.
      results_files = [
          f for f in dir_list if f.endswith('.jsonl') and f != '_runs.jsonl'
      ]

    expected = args.expected_rows if args.expected_rows is not None else expected_rows(args.dataset)
    print(f"Expecting {expected:,} rows per results file")

    # The filtered setting needs every true object of each (subject, relation,
    # day). Ranks fall back to raw if the index cannot be built, and the
    # resolved share below says whether that happened.
    try:
        true_objects = load_true_objects(args.dataset)
        print(f"Fact index: {len(true_objects):,} (subject, relation, day) keys")
    except (OSError, json.JSONDecodeError) as exc:
        true_objects = None
        print(f"WARNING: filtered ranking disabled ({exc}); filtered == raw below")

    scoring_args = _ScoringArgs(args.dataset)
    print()
    header = f"{'file':70s} {'H@1':>7s} {'H@3':>7s} {'H@10':>7s} | "
    header += f"{'fH@1':>7s} {'fH@3':>7s} {'fH@10':>7s} {'resolved':>9s}"
    print(header)
    print("-" * len(header))

    for f in sorted(results_files):

        with open(f'{path}/{f}', 'r', encoding='utf-8') as results:
            lines = [ln for ln in results if ln.strip()]

        # Scoring a short file silently reports a metric over a subset, which
        # looks plausible and is not comparable to anything.
        if len(lines) < expected:
            print(f"SKIPPED {f}: {len(lines):,} rows, expected {expected:,} — "
                  "incomplete run")
            continue
        if len(lines) > expected:
            print(f"WARNING {f}: {len(lines):,} rows, expected {expected:,}")

        metric = HitsMetric()
        for line in lines[:expected]:
            update_metric(json.loads(line), metric, scoring_args, true_objects)

        d = metric.dump()
        resolved = 100 * metric.resolved / metric.total if metric.total else 0.0
        print(f"{f.replace('.jsonl',''):70s} "
              f"{d['hit1']:7.3f} {d['hit3']:7.3f} {d['hit10']:7.3f} | "
              f"{d['f_hit1']:7.3f} {d['f_hit3']:7.3f} {d['f_hit10']:7.3f} "
              f"{resolved:8.1f}%")
