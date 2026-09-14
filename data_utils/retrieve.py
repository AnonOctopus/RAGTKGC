from TLR import Retriever
from basic import read_txt_as_list, read_json, write_txt
from id_words import convert_dataset
import os, glob
import argparse
import json
import logging


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", "-d", default="icews14", type=str)
    p.add_argument(
        "--retrieve_type", "-t", default="TLogic", choices=["TLogic", "bs"],
        help=("Retrieval algorithm. Only TLogic is reachable from this script: "
              "retrieve_type is not passed through to the Retriever, so the "
              "entity-only 'bs' baseline cannot be selected here yet."),
    )
    p.add_argument("--name_of_rules_file", "-r", default="", type=str)
    # Negated store_false rather than a --rule_length_all boolean: argparse applies
    # type= to the raw string, and bool("False") is True, so the old form could not
    # be disabled from the command line at all.
    p.add_argument(
        "--length_1_only",
        dest="rule_length_all",
        action="store_false",
        help=(
            "Consult only rules whose body has a single relation. Retrieval of "
            "longer rules is not implemented (only the last body relation is "
            "matched, and it is anchored at the query subject rather than at the "
            "intermediate entity), so a bank containing longer rules is rejected "
            "unless this flag is set."
        ),
    )
    # Names the output folder and selects the rule bank by filename. raw and
    # standard are baseline labels with no bank of their own, so they need an
    # explicit --name_of_rules_file.
    p.add_argument("--mining", "-m", default='ragtkgc', type=str,
                   choices=["gtkg", "exhaustive", "ragtkgc", "ragtkgc_no_walks",
                            "raw", "standard"])
    p.add_argument(
        "--inverse_body_object_match",
        action="store_true",
        help="If set, inverse body relations are matched using fact object == query subject.",
    )
    p.add_argument(
        "--early_stop_at_num_facts",
        action="store_true",
        help="If set, stop iterating rules for a query once the collected fact count reaches num_facts.",
    )
    p.add_argument(
        "--index_target",
        action="store_true",
        help=("If set, prefix every object in the history with its entity id, as "
              "18.Thailand. Subjects and relations stay names. Matches the "
              "format the LLaMA baseline generates. Cannot be combined with "
              "--save_metadata: the metadata stores fact indices, so this is a "
              "rendering choice that belongs to apply_history_filters.py."),
    )
    # --- new arguments ---
    p.add_argument(
        "--num_facts", "-nf",
        default=None, type=int,
        help="Maximum number of history facts to keep per sample. If omitted, all facts are kept.",
    )
    p.add_argument(
        "--confidence_threshold", "-ct",
        default=None, type=float,
        help="Split retrieved facts into above/below this rule-confidence threshold.",
    )
    p.add_argument(
        "--model_type", "-mt",
        default="t5", choices=["t5", "llm"],
        help="Output format for the threshold split: 't5' concatenates groups, 'llm' adds text labels.",
    )
    p.add_argument(
        "--top_k_rules", "-k",
        default=None, type=int,
        help="Keep only the top-k highest-confidence fired rules before applying num_facts trimming.",
    )
    p.add_argument(
        "--save_metadata",
        action="store_true",
        help=(
            "Save per-sample retrieval metadata (all facts + fired rules, before any trimming) "
            "to a JSON file so filters can be applied later via apply_history_filters.py "
            "without re-running the full retrieval."
        ),
    )
    return vars(p.parse_args())


def resolve_rules_file(rules_dir, name_rules, mining, logger):
    """Pick the rule bank to retrieve with.

    Args:
        rules_dir: directory holding the mined banks.
        name_rules: explicit bank filename, or "" to select by algorithm.
        mining: mining algorithm name, matched against the bank filename.
        logger: logger for the failure explanation.

    Returns:
        str | None: path to the bank, or None when it cannot be resolved
            unambiguously.
    """
    if name_rules:
        path = rules_dir + name_rules
        if os.path.isfile(path):
            return path
        logger.error("Rules file not found: %s", path)
        return None

    candidates = sorted(
        p for p in glob.glob(rules_dir + '*rules.json')
        if f"_{mining}_" in os.path.basename(p)
    )
    if len(candidates) == 1:
        return candidates[0]

    available = sorted(os.path.basename(p) for p in glob.glob(rules_dir + '*rules.json'))
    if not candidates:
        logger.error(
            "No rule bank in %s carries '_%s_' in its name. Mine one, or pass "
            "--name_of_rules_file explicitly. Available: %s",
            rules_dir, mining, available or "none",
        )
    else:
        logger.error(
            "%d rule banks match '_%s_'; pass --name_of_rules_file to choose. "
            "Matches: %s",
            len(candidates), mining,
            [os.path.basename(p) for p in candidates],
        )
    return None


def _setup_logging(log_path):
    """Configure a logger that writes to both a file and stdout."""
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


if __name__ == "__main__":
    parsed = parser()

    retrieve_type          = parsed["retrieve_type"]
    type_dataset           = parsed["dataset"]
    name_rules             = parsed["name_of_rules_file"]
    rule_length_all        = parsed["rule_length_all"]
    inverse_body_object_match = parsed["inverse_body_object_match"]
    early_stop_at_num_facts   = parsed["early_stop_at_num_facts"]
    index_target           = parsed["index_target"]
    confidence_threshold   = parsed["confidence_threshold"]
    model_type             = parsed["model_type"]
    top_k_rules            = parsed["top_k_rules"]
    num_facts              = parsed["num_facts"]
    save_metadata          = parsed["save_metadata"]

    # build_bs never opens the metadata file, so the run would finish reporting
    # success and leave apply_history_filters.py with nothing to read. The flag
    # is inert regardless (see the argument's help), but fail loudly rather than
    # produce a base variant whose metadata directory is empty.
    if retrieve_type != "TLogic":
        raise SystemExit(
            f"--retrieve_type {retrieve_type} is not reachable: retrieve.py does "
            "not pass retrieve_type to Retriever, so build_tl always runs. Wire "
            "the parameter through before selecting another algorithm."
        )

    # --save_metadata records the complete pre-filter retrieval state, and that
    # completeness is the whole basis for re-deriving filters later. A filtered
    # run records an already-filtered state, from which every derived dataset
    # would be silently wrong — a second num_facts cap on a capped list, a top-k
    # over rules that were already trimmed. early_stop_at_num_facts belongs in
    # the list too: it truncates retrieval itself, so its metadata is partial
    # even though no post-filter ran. index_target is there for the opposite
    # reason: it renders nothing in this mode, so accepting it would name the
    # output directory _idn for a directory holding no rendered history at all.
    if save_metadata:
        _active = [
            name for name, value in (
                ("--num_facts", num_facts),
                ("--top_k_rules", top_k_rules),
                ("--confidence_threshold", confidence_threshold),
                ("--early_stop_at_num_facts", early_stop_at_num_facts or None),
                ("--index_target", index_target or None),
            ) if value is not None
        ]
        if _active:
            raise SystemExit(
                "--save_metadata writes the base, unfiltered dataset and cannot be "
                f"combined with {', '.join(_active)}. Retrieve once with no filters, "
                "then derive each filtered dataset with apply_history_filters.py."
            )

    path_workspace = "../data/original/" + type_dataset + '/'
    path_out_tl    = "../data/processed_new/" + type_dataset + "/output/" + type_dataset + "/"
    path_save      = "../data/processed_new/" + type_dataset + f"/{parsed['mining']}/"

    os.makedirs(path_save, exist_ok=True)
    logs_dir = "../logs/"
    os.makedirs(logs_dir, exist_ok=True)
    log_file = logs_dir + f"retriever_{type_dataset}.log"
    logger = _setup_logging(log_file)
    logger.info("Starting retrieval — dataset=%s  mining=%s  retrieve_type=%s",
                type_dataset, parsed['mining'], retrieve_type)
    logger.info("Parameters: %s", json.dumps(
        {k: parsed[k] for k in ("inverse_body_object_match", "early_stop_at_num_facts",
                                 "index_target", "confidence_threshold", "model_type",
                                 "top_k_rules", "num_facts", "save_metadata")},
        indent=2))

    period = 1

    # num_relations is derived from the dataset's own relation2id.json in the split
    # loop below, rather than hardcoded per dataset. It is load-bearing twice in
    # TLR.build_tl — `body_rel % num_relations` maps an inverse relation id back to
    # its forward relation, and `body_rel >= num_relations` is the direction test —
    # so a wrong value would retrieve the wrong relation's facts and misclassify
    # rule direction simultaneously, with no error raised.

    li_files = ['train', 'test', 'valid']

    # ------------------------------------------------------------------
    # PRE-RUN VALIDATION — fail fast before any long-running retrieval.
    # ------------------------------------------------------------------
    logger.info("=== Pre-run validation ===")
    _errors = []

    # Rules file. --mining only names the output folder, so without this the
    # first bank in glob order is loaded whatever algorithm it came from, and a
    # gtkg run silently retrieving with the exhaustive bank leaves no trace.
    # An explicit --name_of_rules_file always wins: retrieval types that use no
    # rule bank (the bs/standard path) have no algorithm-tagged bank to match.
    dir_rules = resolve_rules_file(path_out_tl, name_rules, parsed['mining'], logger)
    if dir_rules is None:
        _errors.append("Could not resolve a rule bank — see the error above.")
    else:
        logger.info("  rules file     OK : %s", dir_rules)

    # Per-split source files
    for _split in li_files:
        for _fname in [_split + '.txt', 'relation2id.json', 'entity2id.json',
                       'ts2id.json', 'all_facts.txt']:
            _p = path_workspace + _fname
            if not os.path.isfile(_p):
                _errors.append(f"Missing source file: {_p}")
            else:
                logger.info("  source file    OK : %s", _p)

    # Output path dry-run (create dirs, resolve all variable names)
    for _split in li_files:
        _sfx = _split
        _sfx += "_inv"                           if inverse_body_object_match else ""
        _sfx += f"_n{num_facts}"                 if num_facts is not None else ""
        _sfx += "_es"                            if early_stop_at_num_facts else ""
        _sfx += f"_k{top_k_rules}"               if top_k_rules is not None else ""
        _sfx += f"_ct{confidence_threshold}-{model_type}"     if confidence_threshold is not None else ""
        _sfx += "_idn"                           if index_target else ""
        _history_dir = path_save + _sfx + "/history_facts/"
        _answers_dir = path_save + _sfx + "/test_answers/"
        _base        = _history_dir + "history_facts_" + type_dataset
        _meta_dir    = path_save + _sfx + "/metadata/"
        try:
            # No history_facts/ in metadata mode — it is not written there.
            if not save_metadata:
                os.makedirs(_history_dir, exist_ok=True)
            os.makedirs(_answers_dir, exist_ok=True)
            if save_metadata:
                os.makedirs(_meta_dir, exist_ok=True)
            # Probe that we can write to the output dirs
            for _probe_dir in ([_answers_dir] + ([_meta_dir] if save_metadata else [_history_dir])):
                _probe = os.path.join(_probe_dir, ".write_probe")
                with open(_probe, "w") as _pf:
                    _pf.write("")
                os.remove(_probe)
            logger.info("  output dirs    OK : %s", path_save + _sfx + "/")
        except Exception as _e:
            _errors.append(f"Cannot create/write output dir for split '{_split}': {_e}")

    if _errors:
        for _err in _errors:
            logger.error("VALIDATION FAILED: %s", _err)
        raise SystemExit("Pre-run validation failed — see errors above. Fix them before re-running.")

    logger.info("=== Validation passed — starting retrieval ===")

    for files in li_files:
        # dir_rules was resolved once during validation; re-resolving per split
        # could pick a different bank if the directory changed mid-run.
        logger.info("Processing split: %s  rules_file: %s", files, dir_rules)

        test_ans     = read_txt_as_list(path_workspace + files + '.txt')

        relations = read_json(path_workspace + 'relation2id.json')
        num_relations = len(relations)
        logger.info("num_relations (from relation2id.json): %d", num_relations)
        entities  = read_json(path_workspace + 'entity2id.json')
        times_id  = read_json(path_workspace + 'ts2id.json')
        test_ans  = convert_dataset(test_ans, path_workspace, period=period)

        chains   = read_json(dir_rules)
        rel_keys = list(relations.keys())
        all_facts = []
        with open(path_workspace + "all_facts.txt", "r", encoding='utf-8') as f:
            all_facts = f.readlines()

        rtr = Retriever(
            test_ans,
            all_facts,
            entities,
            relations,
            times_id,
            num_relations,
            chains,
            rel_keys,
            dataset=type_dataset,
            rule_length_all=rule_length_all,
            inverse_body_object_match=inverse_body_object_match,
            early_stop_at_num_facts=early_stop_at_num_facts,
            index_target=index_target,
            confidence_threshold=confidence_threshold,
            model_type=model_type,
            top_k_rules=top_k_rules,
            save_metadata=save_metadata,
            num_facts=num_facts,
        )
        # Build output path suffix from active options — must happen BEFORE
        # retrieval so all paths are defined when we write results.
        out_suffix = files
        out_suffix += "_inv"                           if inverse_body_object_match else ""
        out_suffix += f"_n{num_facts}"                 if num_facts is not None else ""
        out_suffix += "_es"                            if early_stop_at_num_facts else ""
        out_suffix += f"_k{top_k_rules}"               if top_k_rules is not None else ""
        out_suffix += f"_ct{confidence_threshold}-{model_type}"     if confidence_threshold is not None else ""
        out_suffix += "_idn"                           if index_target else ""

        history_dir  = path_save + out_suffix + "/history_facts/"
        answers_dir  = path_save + out_suffix + "/test_answers/"
        base_name    = history_dir + "history_facts_" + type_dataset
        path_txt     = base_name + ".txt"
        path_ruleids = base_name + "_rule_ids.txt"
        path_answer  = answers_dir + "test_answers_" + type_dataset + ".txt"
        stats_path   = logs_dir + f"history_stats_{type_dataset}_{files}.json"
        meta_dir     = path_save + out_suffix + "/metadata/"
        meta_path    = meta_dir + f"history_metadata_{type_dataset}_{files}.jsonl"

        if not save_metadata:
            os.makedirs(history_dir, exist_ok=True)
        os.makedirs(answers_dir,  exist_ok=True)
        if save_metadata:
            os.makedirs(meta_dir, exist_ok=True)

        logger.info("Output paths for split '%s':", files)
        if not save_metadata:
            logger.info("  history text : %s", path_txt)
            logger.info("  rule ids     : %s", path_ruleids)
        logger.info("  answers      : %s", path_answer)
        logger.info("  stats        : %s", stats_path)
        if save_metadata:
            logger.info("  metadata     : %s", meta_path)

        # Per-sample metadata is streamed to JSONL during build (constant RAM).
        # Hand the retriever the destination and the header fields known only here.
        if save_metadata:
            rtr.metadata_path = meta_path
            rtr.metadata_extra_header = {
                "split": files,
                "mining": parsed["mining"],
                "base_answers_path": os.path.abspath(path_answer),
                "original_data_dir": os.path.abspath(path_workspace),
            }

        test_idx, test_text, test_rule_ids = rtr.get_output()

        # These lists become files that downstream code pairs by line number,
        # and the answers come from a different source (the raw split file)
        # than the histories (the retriever). A length mismatch would silently
        # misalign every sample after the divergence, so refuse to write rather
        # than emit a corrupt training set.
        _lengths = {
            "history_text": len(test_text),
            "rule_ids": len(test_rule_ids),
            "answers": len(test_ans),
        }
        if len(set(_lengths.values())) != 1:
            raise RuntimeError(
                f"Line counts disagree for split '{files}', so the history, "
                f"rule-id and answer files would be misaligned: {_lengths}"
            )
        logger.info("Alignment check passed: %d rows per file", len(test_text))

        # History size statistics are already logged inside Retriever;
        # write them to a per-split JSON file as well.
        with open(stats_path, "w", encoding="utf-8") as sf:
            json.dump(rtr.build_stats, sf, indent=2)
        logger.info("History statistics written to: %s", stats_path)

        # A second copy of this file was also written as
        # "<base>_idx_fine_tune_all.txt". It was byte-identical to path_txt and
        # nothing ever read it, so it is no longer produced.
        #
        # With --save_metadata this run is the base, unfiltered dataset, and
        # apply_history_filters.py with no filters regenerates these two files
        # from the metadata. Nothing trains on the uncapped history.
        if save_metadata:
            logger.info("Uncapped history_facts not written — regenerate with: "
                        "apply_history_filters.py --metadata %s", meta_path)
        else:
            with open(path_txt, 'w', encoding='utf-8') as f:
                for i in range(len(test_text)):
                    f.write(test_text[i][0] + '\n')
            with open(path_ruleids, 'w', encoding='utf-8') as f:
                for rule_ids in test_rule_ids:
                    f.write(json.dumps(rule_ids) + '\n')
            logger.info("Saved history text  : %s", path_txt)
            logger.info("Saved rule ids      : %s", path_ruleids)

        # Always names: create_json_train.py adds the id prefix to the target
        # when asked, so the answers file stays one canonical form.
        write_txt(path_answer, test_ans, head='')
        logger.info("Saved answers       : %s", path_answer)

        # Per-sample metadata was streamed to JSONL during get_output() above.
        if save_metadata:
            logger.info("Per-sample metadata saved to : %s", meta_path)

