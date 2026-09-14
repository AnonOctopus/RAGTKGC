"""
One-time converter: old monolithic metadata JSON  ->  streaming JSONL.

The original retrieve.py --save_metadata wrote a single
    {"header": {...}, "samples": [ {...}, {...}, ... ]}
object, which forces consumers to json.load() the whole (multi-GB) file into
RAM.  retrieve.py now streams JSONL instead (header on line 1, one sample per
subsequent line).  This script upgrades an existing .json file to that format
*without* ever holding it all in memory, using ijson's incremental parser.

Usage:
    python convert_metadata_to_jsonl.py \\
        --input  .../metadata/history_metadata_icews14_test.json \\
        --output .../metadata/history_metadata_icews14_test.jsonl

If --output is omitted, the input path with a .jsonl extension is used.
"""

import argparse
import json
import os

import ijson


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", "-i", required=True, help="Old monolithic .json metadata file.")
    p.add_argument("--output", "-o", default=None, help="Destination .jsonl (default: input with .jsonl).")
    args = p.parse_args()

    in_path  = args.input
    out_path = args.output or (os.path.splitext(in_path)[0] + ".jsonl")
    if os.path.abspath(in_path) == os.path.abspath(out_path):
        raise SystemExit("Refusing to overwrite the input file; choose a different --output.")

    # Header is a small object — pull it out incrementally and write it first.
    # use_float=True makes ijson emit float/int instead of Decimal (which the
    # stdlib json encoder cannot serialise).
    with open(in_path, "rb") as f:
        header = next(ijson.items(f, "header", use_float=True))

    n = 0
    with open(in_path, "rb") as f, open(out_path, "w", encoding="utf-8") as out:
        out.write(json.dumps(header) + "\n")
        for sample in ijson.items(f, "samples.item", use_float=True):
            out.write(json.dumps(sample) + "\n")
            n += 1
            if n % 1000 == 0:
                print(f"  ...{n} samples", end="\r", flush=True)

    print(f"Converted {n} samples")
    print(f"  from : {in_path}")
    print(f"  to   : {out_path}")


if __name__ == "__main__":
    main()
