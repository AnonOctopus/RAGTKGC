from basic import just_write_json, just_read_txt, read_txt_as_list, read_json
import random
import argparse
import os
import json

def convert_txt_to_json(inputs, entities, test_ans, rule_ids=None,
                        index_target=False):
    """Pair rendered histories with their targets as training json records.

    Args:
        inputs: rendered prompts, one per query.
        entities: entity name to id. Read only when index_target is set.
        test_ans: answer file lines, tab-separated quads paired by row order.
        rule_ids: per-query rule provenance, or None to omit the field.
        index_target: write the target as "18.Thailand" rather than "Thailand".

    Returns:
        list[dict]: records with context, target and optionally rule_ids.

    Raises:
        ValueError: the history, answer and rule-id files disagree in length.
        KeyError: an object is missing from entities while index_target is set.
    """
    test_ans = [x.strip().split('\t') for x in test_ans if x.strip()]

    # These lists are three files paired by row order. A short list would be
    # padded or would raise deep inside the loop, so check up front. Note that
    # an empty rule_ids entry is legitimate (no rule fired for that query) and
    # is produced by the blank-line branch of the reader above; only a *missing*
    # entry means the files are misaligned.
    if len(inputs) != len(test_ans):
        raise ValueError(
            f"History and answer files disagree: {len(inputs)} contexts vs "
            f"{len(test_ans)} answers"
        )
    if rule_ids is not None and len(rule_ids) != len(test_ans):
        raise ValueError(
            f"Rule-id file is misaligned: {len(rule_ids)} rows vs "
            f"{len(test_ans)} answers"
        )

    data_list = []
    for i in range(len(test_ans)):
        name_obj = test_ans[i][2]
        target = f"{entities[name_obj]}.{name_obj}" if index_target else name_obj
        data = {
            "context": inputs[i],
            "target": target
        }
        if rule_ids is not None:
            data["rule_ids"] = rule_ids[i]
        data_list.append(data)
    return data_list

def sample_data_training(dir_dataset, dir_of_answers, dir_of_entities2id, path_save, name_train, nums_sample = [16, 64, 256, 512, 1024], dir_of_rule_ids = "", index_target=False, seed=None):
    """Write the full training json plus one subset file per requested size.

    Args:
        dir_dataset: history_facts text file, samples separated by a blank line.
        dir_of_answers: answers file, one target quad per line, paired by row order.
        dir_of_entities2id: entity2id json, used to resolve integer targets.
        path_save: directory to write into; created by the caller.
        name_train: stem of the written files, e.g. "icews14_gtkg_inv_n50_train".
            Names the output only; it is not the dataset name.
        nums_sample: subset sizes to draw; a size larger than the available
            samples is skipped rather than raising.
        dir_of_rule_ids: rule-id file; when empty it is inferred from
            dir_dataset by substituting the "_rule_ids.txt" suffix.
        index_target: write the target as "18.Thailand" rather than "Thailand",
            matching the format the LLaMA baseline generates.
        seed: seed for the subset sampler; None draws non-reproducibly.

    Returns:
        None
    """
    content = just_read_txt(dir_dataset)
    inputs = content.split('\n\n')
    # The history file ends with the sample separator, so the split leaves one
    # trailing empty chunk. Dropping it keeps the alignment check below strict.
    # It cannot discard a real sample: one with no retrieved facts is a chunk
    # holding just its query line, never an empty chunk.
    if inputs and not inputs[-1].strip():
        inputs.pop()
    test_ans = read_txt_as_list(dir_of_answers)
    entities = read_json(dir_of_entities2id)
    rule_ids = None
    if dir_of_rule_ids == "":
        inferred_rule_ids = dir_dataset.replace('.txt', '_rule_ids.txt')
        if os.path.exists(inferred_rule_ids):
            dir_of_rule_ids = inferred_rule_ids
    if dir_of_rule_ids != "" and os.path.exists(dir_of_rule_ids):
        rule_ids = []
        with open(dir_of_rule_ids, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == "":
                    rule_ids.append([])
                else:
                    rule_ids.append(json.loads(line))

    data_list = convert_txt_to_json(inputs, entities, test_ans, rule_ids=rule_ids, index_target=index_target)
    output_file_full = path_save+"/"+name_train+'.json'
    just_write_json(data_list, output_file_full, indent=4)
    print("saved as ", output_file_full)
    # Validate every requested size before writing any subset, so an oversize
    # request fails cleanly instead of leaving a half-written set of files.
    oversize = [n for n in nums_sample if n > len(data_list)]
    if oversize:
        raise ValueError(
            f"Requested subset sizes {oversize} exceed the {len(data_list)} "
            f"samples available in {dir_dataset}"
        )

    # A dedicated Random instance rather than the module-level functions: the
    # subsets become reproducible from `seed` without reseeding the global RNG
    # that other imported code may also be drawing from. Random(None) seeds from
    # entropy, so omitting the seed keeps the previous behaviour.
    rng = random.Random(seed)
    for num in nums_sample:
        sampled_data = rng.sample(data_list, num)
        output_file = path_save+"/"+name_train+'_' + str(num) + '.json'
        just_write_json(sampled_data, output_file, indent=4)
        print("saved as ", output_file)

def parse_int_list(string):
    try:
        return [int(x) for x in string.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid list of integers: {}".format(string))
       
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_of_trainset", default="", type=str)
    parser.add_argument("--dir_of_answers", default="", type=str)
    parser.add_argument("--dir_of_entities2id", default="", type=str)
    parser.add_argument("--dir_of_rule_ids", default="", type=str)
    parser.add_argument("--path_save", default="", type=str)
    parser.add_argument('--nums_sample', default="16", type=parse_int_list, 
                        help='The numbers of samples in a list of integers separated by commas')
    # Names the output files, not the dataset: it was called --dataset, so every
    # variant of a dataset wrote the same icews14.json and the results filename
    # that inherits this stem said nothing about which retrieval produced it.
    parser.add_argument(
        "--name_train", required=True, type=str,
        help=("Stem for the written json files, e.g. "
              "icews14_gtkg_inv_n50_train -> icews14_gtkg_inv_n50_train.json "
              "plus one file per --nums_sample entry."),
    )
    parser.add_argument(
        "--index_target",
        action="store_true",
        help=("Prefix the target with its entity id, as 18.Thailand. Must match "
              "the --index_target used during retrieval."),
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Seed for the training-subset sampler. Omit for a non-reproducible draw.",
    )
    parsed = vars(parser.parse_args())
    return parsed

if __name__ == "__main__":
    parsed = parser()
    dir_of_trainset = parsed["dir_of_trainset"]
    print(dir_of_trainset)
    dir_of_answers = parsed["dir_of_answers"]
    print(dir_of_answers)
    dir_of_entities2id = parsed["dir_of_entities2id"]
    print(dir_of_entities2id)
    dir_of_rule_ids = parsed["dir_of_rule_ids"]
    print(dir_of_rule_ids)
    path_save = parsed["path_save"]
    print(path_save)
    nums_sample = parsed["nums_sample"]
    name_train = parsed["name_train"]
    index_target = parsed["index_target"]
    seed = parsed["seed"]

    if not os.path.exists(path_save):
            os.makedirs(path_save)
    sample_data_training(dir_of_trainset, dir_of_answers, dir_of_entities2id, path_save, name_train, nums_sample, dir_of_rule_ids, index_target=index_target, seed=seed)