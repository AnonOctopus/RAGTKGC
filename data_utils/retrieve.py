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
    p.add_argument("--retrieve_type", "-t", default="TLogic-3", type=str)
    p.add_argument("--name_of_rules_file", "-r", default="", type=str)
    p.add_argument("--rule_length_all", "-l", default=True, type=bool)
    p.add_argument("--mining", "-m", default='ragtkgc', type=str)
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
        "--use_ids",
        action="store_true",
        help="If set, output context prompts and answer files using integer IDs instead of natural language names.",
    )
    # --- new arguments ---
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
    return vars(p.parse_args())


def _setup_logging(log_path):
    """Configure a logger that writes to both a file and stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
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
    use_ids                = parsed["use_ids"]
    confidence_threshold   = parsed["confidence_threshold"]
    model_type             = parsed["model_type"]
    top_k_rules            = parsed["top_k_rules"]

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
                                 "use_ids", "confidence_threshold", "model_type", "top_k_rules")},
        indent=2))

    period = 1
    if type_dataset == "icews18":
        num_relations = 256
    elif type_dataset == "icews14":
        num_relations = 230
    elif type_dataset == "GDELT":
        num_relations = 238
    else:
        num_relations = 24

    li_files = ['train', 'test', 'valid']

    for files in li_files:
        existing_rules = glob.glob(path_out_tl + '*rules.json')
        logger.info("Existing rules files: %s", existing_rules)
        dir_rules = existing_rules[0] if name_rules == "" else path_out_tl + name_rules
        logger.info("Processing split: %s  rules_file: %s", files, dir_rules)

        test_ans     = read_txt_as_list(path_workspace + files + '.txt')
        test_ans_ids = test_ans

        relations = read_json(path_workspace + 'relation2id.json')
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
            use_ids=use_ids,
            confidence_threshold=confidence_threshold,
            model_type=model_type,
            top_k_rules=top_k_rules,
        )
        test_idx, test_text, test_rule_ids = rtr.get_output()

        # History size statistics are already logged inside Retriever;
        # write them to a per-split JSON file as well.
        stats_path = logs_dir + f"history_stats_{type_dataset}_{files}.json"
        with open(stats_path, "w", encoding="utf-8") as sf:
            json.dump(rtr.build_stats, sf, indent=2)
        logger.info("History statistics written to: %s", stats_path)

        # Build output path suffix from active options
        out_suffix = files
        out_suffix += "_inverse_included"  if inverse_body_object_match else ""
        out_suffix += "_num_facts"         if not early_stop_at_num_facts else ""
        out_suffix += f"_top{top_k_rules}rules" if top_k_rules is not None else ""
        out_suffix += f"_thresh{confidence_threshold}_{model_type}" if confidence_threshold is not None else ""
        out_suffix += "_ids"               if use_ids else ""

        history_dir = path_save + out_suffix + "/history_facts/"
        os.makedirs(history_dir, exist_ok=True)

        base_name   = history_dir + "history_facts_" + type_dataset
        path_txt    = base_name + ".txt"
        path_idx    = base_name + "_idx_fine_tune_all.txt"
        path_ruleids = base_name + "_rule_ids.txt"

        write_txt(path_idx, test_text)
        with open(path_txt, 'w', encoding='utf-8') as f:
            for i in range(len(test_text)):
                f.write(test_text[i][0] + '\n')
        with open(path_ruleids, 'w', encoding='utf-8') as f:
            for rule_ids in test_rule_ids:
                f.write(json.dumps(rule_ids) + '\n')
        logger.info("Saved history text  : %s", path_txt)
        logger.info("Saved history idx   : %s", path_idx)
        logger.info("Saved rule ids      : %s", path_ruleids)

        answers_dir = path_save + out_suffix + "/test_answers/"
        os.makedirs(answers_dir, exist_ok=True)
        path_answer = answers_dir + "test_answers_" + type_dataset + ".txt"
        write_txt(path_answer, test_ans_ids if use_ids else test_ans, head='')
        logger.info("Saved answers       : %s", path_answer)

