"""Check that two retrieval outputs differ only in the way they were meant to.

Compares an --index_target history file against its bare-name counterpart and
reports every difference that is not an object gaining an "id." prefix. A
clean run means the flag changed exactly one thing.

Usage:
    python diff_variants.py --indexed <history_facts.txt> --bare <history_facts.txt>
    python diff_variants.py --indexed <...> --bare <...> \\
        --answers_indexed <answers.txt> --answers_bare <answers.txt>
"""

import argparse
import re

# A history line: "<day>: [<subject>, <relation>, <object>] " — the object runs
# to the closing bracket, and both subject and relation may contain commas.
LINE = re.compile(r'^(\d+): \[(.*), (.*?)\] $')
# The last line of every sample is the query: "<day>: [<subject>, <relation>,"
# with the object withheld for the model to generate. It has no object, so
# index_target cannot alter it, and the two variants must agree byte for byte.
QUERY = re.compile(r'^(\d+): \[(.*),$')
# An object in indexed form: digits, a dot, then the name.
INDEXED_OBJECT = re.compile(r'^(\d+)\.(.*)$')


def parse(line):
    """Split one history line into day, head text and object.

    Args:
        line: a history line, without its trailing newline.

    Returns:
        tuple[str, str, str] | None: day, the "subject, relation" text, and the
            object; None when the line does not match the expected shape.
    """
    match = LINE.match(line)
    return match.groups() if match else None


def compare(indexed_path, bare_path):
    """Compare an indexed history file against its bare counterpart.

    Args:
        indexed_path: history file produced with --index_target.
        bare_path: history file produced without it.

    Returns:
        dict: counts under the keys "lines" (history facts compared), "blank"
            (sample separators), "queries" (query lines, required to be equal),
            "prefixed" (objects that gained exactly an id prefix),
            "unparsed_indexed", "unparsed_bare", and "problems" holding up to
            twenty descriptions of unexpected differences.

    Raises:
        OSError: either file could not be read.
        SystemExit: the two files have different line counts, which means they
            do not describe the same queries.
    """
    with open(indexed_path, encoding='utf-8') as handle:
        indexed = handle.read().split('\n')
    with open(bare_path, encoding='utf-8') as handle:
        bare = handle.read().split('\n')

    if len(indexed) != len(bare):
        raise SystemExit(
            f"Line counts differ: {len(indexed):,} vs {len(bare):,}. The two "
            "runs did not process the same queries, so the diff is meaningless."
        )

    stats = {"lines": 0, "blank": 0, "queries": 0, "prefixed": 0,
             "unparsed_indexed": 0, "unparsed_bare": 0, "problems": []}

    def note(message):
        if len(stats["problems"]) < 20:
            stats["problems"].append(message)

    for n, (line_i, line_b) in enumerate(zip(indexed, bare), start=1):
        # Blank lines separate samples and must stay aligned.
        if not line_i.strip() or not line_b.strip():
            stats["blank"] += 1
            if line_i.strip() != line_b.strip():
                note(f"line {n}: sample boundaries misaligned")
            continue

        # Query lines carry no object; the only correct outcome is equality.
        is_query_i, is_query_b = bool(QUERY.match(line_i)), bool(QUERY.match(line_b))
        if is_query_i or is_query_b:
            stats["queries"] += 1
            if line_i != line_b:
                note(f"line {n}: query lines differ:\n    {line_i!r}\n    {line_b!r}")
            continue

        stats["lines"] += 1
        parts_i, parts_b = parse(line_i), parse(line_b)
        if parts_i is None:
            stats["unparsed_indexed"] += 1
            note(f"line {n}: indexed line does not match the expected shape: {line_i!r}")
            continue
        if parts_b is None:
            stats["unparsed_bare"] += 1
            note(f"line {n}: bare line does not match the expected shape: {line_b!r}")
            continue

        day_i, head_i, obj_i = parts_i
        day_b, head_b, obj_b = parts_b
        if day_i != day_b:
            note(f"line {n}: day differs, {day_i} vs {day_b}")
        if head_i != head_b:
            note(f"line {n}: subject/relation differ:\n    {head_i!r}\n    {head_b!r}")

        indexed_object = INDEXED_OBJECT.match(obj_i)
        if not indexed_object:
            note(f"line {n}: object carries no id prefix: {obj_i!r}")
        elif indexed_object.group(2) != obj_b:
            note(f"line {n}: object name changed, {indexed_object.group(2)!r} "
                 f"vs {obj_b!r}")
        else:
            stats["prefixed"] += 1

    return stats


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--indexed", required=True,
                   help="history_facts file built with --index_target")
    p.add_argument("--bare", required=True,
                   help="history_facts file built without it")
    p.add_argument("--answers_indexed", default=None,
                   help="answers file from the indexed run; must be identical "
                        "to the bare one, since answers are always names")
    p.add_argument("--answers_bare", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parser()
    stats = compare(args.indexed, args.bare)

    print(f"history lines compared : {stats['lines']:,}")
    print(f"query lines (unchanged): {stats['queries']:,}")
    print(f"sample boundaries      : {stats['blank']:,}")
    print(f"objects correctly prefixed: {stats['prefixed']:,}")
    if stats["unparsed_indexed"] or stats["unparsed_bare"]:
        print(f"unparsed lines         : {stats['unparsed_indexed']:,} indexed, "
              f"{stats['unparsed_bare']:,} bare")

    if args.answers_indexed and args.answers_bare:
        with open(args.answers_indexed, encoding='utf-8') as handle:
            a = handle.read()
        with open(args.answers_bare, encoding='utf-8') as handle:
            b = handle.read()
        print(f"answers files identical: {a == b}")
        if a != b:
            stats["problems"].append(
                "answers files differ; they hold entity names in both variants")

    if stats["problems"]:
        print(f"\n{len(stats['problems'])} problem(s), first few:")
        for message in stats["problems"]:
            print(f"  {message}")
        raise SystemExit(1)

    print("\nOK: the only difference is the object's id prefix.")
