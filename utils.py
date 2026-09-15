import argparse
from dataclasses import dataclass
import json
import math
import os

MAX_HITS = 10

# The LLaMA baseline's instruction block, verbatim, including its chat markup.
# It describes the id.Name object format, so it belongs with --index_target.
_LLAMA_INSTRUCTION = (
    "<s>[INST] <<SYS>>You must be able to correctly predict the next "
    "{object_label} from a given text consisting of multiple quadruplets in "
    "the form of \"{time}:[{subject}, {relation}, {object_label}.{object}]\" "
    "and the query in the form of \"{time}:[{subject}, {relation},\" in the "
    "end.\nYou must generate {object_label}.{object}\n\n<</SYS>>"
)
_LLAMA_INSTRUCTION_END = "[/INST]"


def apply_prompt_prefix(prompt):
    """Wrap one prompt in the LLaMA baseline's instruction block.

    Args:
        prompt: the rendered history and query.

    Returns:
        str: the wrapped prompt.
    """
    # Applied after any truncation, never baked into the dataset: the block
    # sits at the front, and tail truncation keeps the end of the prompt, so
    # a stored prefix would be the first thing discarded.
    return _LLAMA_INSTRUCTION + prompt + _LLAMA_INSTRUCTION_END

# Same rule as data_utils/TLR.time_period. Duplicated rather than imported
# because TLR imports its siblings by bare name, so it is only importable with
# data_utils as the working directory.
_HOURLY_DATASETS = ("icews14", "icews18")


def time_period(dataset):
    """Divisor applied to a timestamp id when a fact's time is rendered.

    Args:
        dataset: dataset name, e.g. "icews14".

    Returns:
        int: the divisor.
    """
    return 24 if dataset in _HOURLY_DATASETS else 1


def normalize_entity(text):
    """The comparable form of an object string: bare name, casefolded.

    Args:
        text: a predicted or gold object, with or without an "id." prefix.

    Returns:
        str: the entity name, lowercased and stripped of surrounding space.
    """
    # Strip surrounding space first: the digit test below runs on the prefix,
    # and leading space would stop it matching.
    text = text.strip()
    # Strip the index prefix only when it is digits. Entity names contain dots
    # of their own — middle initials such as "Vincent_C._Siew" — and none of
    # them begin with a number, so this is unambiguous in both formats.
    prefix, sep, name = text.partition('.')
    if sep and prefix.isdigit():
        text = name
    return text.casefold().strip()


def load_true_objects(dataset, base_data_dir='./data/original'):
    """Index the true objects of every (subject, relation, day) in the dataset.

    Args:
        dataset: dataset name, e.g. "icews14".
        base_data_dir: folder holding <dataset>/all_facts.txt and ts2id.json.
            all_facts.txt is the union of train, valid and test, which is the
            universe the filtered setting is defined over.

    Returns:
        dict: (subject, relation, day) -> frozenset of object names. The day is
            the integer the history and query lines render, so a results row
            needs no timestamp conversion beyond int(float(...)).

    Raises:
        OSError: all_facts.txt or ts2id.json could not be read.
        json.JSONDecodeError: ts2id.json is not valid JSON.
    """
    period = time_period(dataset)
    with open(os.path.join(base_data_dir, dataset, 'ts2id.json'), encoding='utf-8') as handle:
        ts2id = json.load(handle)

    index = {}
    path = os.path.join(base_data_dir, dataset, 'all_facts.txt')
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 4:
                continue
            subject, relation, obj, date = (p.strip() for p in parts[:4])
            if date not in ts2id:
                continue
            key = (subject, relation, int(ts2id[date]) // period)
            index.setdefault(key, set()).add(normalize_entity(obj))
    return {key: frozenset(objs) for key, objs in index.items()}


def true_objects_for(example, true_objects, dataset):
    """The true objects of one results row's query.

    Args:
        example: results row carrying entity, relation and timestamp.
        true_objects: index from load_true_objects, or None to disable filtering.
        dataset: dataset name. Accepted for symmetry with the other helpers;
            the index is already keyed by rendered day.

    Returns:
        frozenset: normalized object names true for this query, empty when
            filtering is disabled or the query does not resolve against the
            index.
    """
    if not true_objects:
        return frozenset()
    try:
        day = int(float(example['timestamp']))
    except (KeyError, TypeError, ValueError):
        return frozenset()
    return true_objects.get((example['entity'], example['relation'], day), frozenset())


def ranks(predictions, target, other_true):
    """Raw and time-aware-filtered rank of a target among the predictions.

    Args:
        predictions: candidate strings, best first and deduplicated. Compared
            in normalized form, so an "id." prefix and letter case are ignored.
        target: the gold object string, in either format.
        other_true: normalized true objects of the same query other than the
            target. Any of these ranked above the target is not counted
            against it.

    Returns:
        tuple[float, float]: raw and filtered rank, both math.inf when the
            target is not among the predictions.
    """
    candidates = [normalize_entity(p) for p in predictions]
    gold = normalize_entity(target)
    if gold not in candidates:
        return math.inf, math.inf
    index = candidates.index(gold)
    survivors = [p for p in candidates[:index] if p not in other_true]
    return index + 1, len(survivors) + 1


@dataclass
class HitsMetric:

    total: int = 0

    # prediction needs to exactly match the target
    hit1: int = 0
    hit3: int = 0
    hit10: int = 0

    # Time-aware filtered: other objects that are true for the same
    # (subject, relation, day) are removed from the ranking before the target's
    # rank is read, so a model is not penalised for ranking another correct
    # answer first.
    f_hit1: int = 0
    f_hit3: int = 0
    f_hit10: int = 0
    # Queries found in the fact index. Well below total means the filtered
    # figures are really raw ones and must not be reported as filtered.
    resolved: int = 0

    # target needs to just be in the prediction, e.g. prediction = "China]", target = "China".
    # However, this metric is tricky, as prediction = "Chinatown" and target = "China" would count as correct. Therefore, we use the above metrics
    total2: int = 0
    hit1p: int = 0
    hit3p: int = 0
    hit10p: int = 0

    def update(self, rank, filtered_rank=None):
        if rank <= 1:
            self.hit1 += 1
        if rank <= 3:
            self.hit3 += 1
        if rank <= 10:
            self.hit10 += 1

        if filtered_rank is None:
            filtered_rank = rank
        if filtered_rank <= 1:
            self.f_hit1 += 1
        if filtered_rank <= 3:
            self.f_hit3 += 1
        if filtered_rank <= 10:
            self.f_hit10 += 1

    def update2(self, rank):

        if rank <= 1:
            self.hit1p += 1
        if rank <= 3:
            self.hit3p += 1
        if rank <= 10:
            self.hit10p += 1

    def dump(self):
        if not self.total:
            return {"t": 0}
        return {
            "t": self.total,
            "hit1": self.hit1 / self.total,
            "hit3": self.hit3 / self.total,
            "hit10": self.hit10 / self.total,
            "f_hit1": self.f_hit1 / self.total,
            "f_hit3": self.f_hit3 / self.total,
            "f_hit10": self.f_hit10 / self.total,
        }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="google/flan-t5-small", type=str) # the base (foundation) model that the fine tuned version is based on
    parser.add_argument("--finetuned_model", type=str) # name of the folder which holds the fine tuned model version or peft adapter
    parser.add_argument(
        "--dataset",
        choices=["icews14", "icews18"],
        default="icews18",
        type=str,
    )
    # Required, not defaulted: the old default named a path from a directory
    # layout that is no longer produced, and its basename becomes part of the
    # results filename, so a stale value silently mislabels the run.
    parser.add_argument("--dataset_path", required=True, type=str,
                        help="Path to the test json, relative to ./data/processed_new/{dataset}/.")
    parser.add_argument("--dataset_rag_path", type=str) # path to the rag test set
    # Beam width, and therefore how many candidates get ranked. Must be at
    # least 10 for H@10 to carry information beyond H@3. Exposed so the
    # candidate count can be ablated without editing the decoder.
    parser.add_argument("--num_beams", default=10, type=int)
    # Exponent beam search divides the sequence log-probability by the token
    # count with, deciding which of two candidates of different length wins
    # rank 1. 0.6 is the conventional seq2seq length-normalisation value
    # (Wu et al., 2016), taken as a standard rather than tuned here. T5's own
    # published value is for summarization, which rewards long outputs, and is
    # the wrong end of the range for a short entity name.
    parser.add_argument("--length_penalty", default=0.6, type=float)
    # Derived from the longest target in the split when omitted, which is
    # correct for any tokenizer. Set it only to pin the budget deliberately.
    parser.add_argument("--max_new_tokens", default=None, type=int)
    # Off by default. It is LLaMA-2-chat markup, inert text for an
    # encoder-decoder model, and it must match what training used.
    parser.add_argument("--prompt_prefix", default=False, action="store_true",
                        help="Wrap prompts in the LLaMA baseline's instruction "
                             "block. Assumes targets are in id.Name form.")
    # Evaluate only the first N samples. For smoke-testing a change to the
    # decoder without paying for the whole split.
    parser.add_argument("--limit", default=None, type=int)
    # Prompts per generation call. Beam search replicates the KV cache per
    # beam, so memory grows with batch x num_beams x sequence length; raise it
    # while that fits and generation gets several times faster. Defaults to 1,
    # which is what every recorded result was produced with.
    parser.add_argument("--batch_size", default=1, type=int,
                        help="Prompts per generation call. 1 reproduces the "
                             "unbatched behaviour exactly.")
    # Which precision the frozen base is served in. A LoRA adapter is a
    # low-rank correction to a particular copy of the base weights, so serving
    # it over a differently quantised copy changes what it was trained to
    # correct. The default matches training_LLaMA.py's, which keeps the two
    # halves consistent without anyone having to remember. Adapters trained
    # under 4bit need it passed explicitly.
    # LLaMA only: the T5 path is unquantised and ignores this.
    parser.add_argument("--precision", choices=["4bit", "bf16"], default="bf16",
                        type=str,
                        help="Precision of the frozen base model at inference. "
                             "Match the precision the adapter was trained with.")
    parser.add_argument("--verbose", default=False, action="store_true")  # print extra information
    parser.add_argument("--tail_truncate_long_inputs", default=False, action="store_true")  # truncate from the tail (oldest history) instead of head when input exceeds token limit

    args = parser.parse_args()

    return args

def get_filename(dataset, dataset_path = '', model_name = '', tail_truncate_included = False):
    filename_args = "_".join(
        [
            model_name,
            dataset_path.split('/')[-1].split('.')[0],
            "tail_truncate" if tail_truncate_included else "no_tail_truncate"
        ]
    )
    directory = f"./results/{dataset}"
    os.makedirs(directory, exist_ok=True)
    filename = f"{directory}/{filename_args}.jsonl"

    # Never replace an existing results file. Two runs whose model name and
    # dataset basename coincide resolve to the same path, and overwriting would
    # discard the earlier result with nothing left to show it existed. The
    # configuration cannot go in the name — these already reach ~240 of the 260
    # characters Windows allows — so a counter is appended instead and the full
    # invocation is recorded in the per-dataset run manifest.
    if os.path.exists(filename):
        stem, ext = os.path.splitext(filename)
        n = 2
        while os.path.exists(f"{stem}_{n}{ext}"):
            n += 1
        filename = f"{stem}_{n}{ext}"
        print(f"a results file already existed; writing to {filename} instead")

    print(f"output file: {filename}")
    return filename

def write_results(x, predictions, direction, writer, args):

    entity, relation, targets, time = x[0], x[1], x[2], x[3]
    example = {
        "timestamp": time,
        "entity": entity,
        "relation": relation,
        "targets": targets,
        "direction": direction,
        "predictions": list(predictions),
    }
    writer.write(json.dumps(example) + "\n")

    if args.verbose:
        print(f"example:\n{json.dumps(example, indent=2)}")

    return example


def update_metric(example, metric, args, true_objects=None):
    """Score one results row into the running metric.

    Args:
        example: results row with targets and predictions.
        metric: HitsMetric to accumulate into.
        args: parsed arguments; only dataset and verbose are read.
        true_objects: index from load_true_objects, or None to leave the
            filtered figures equal to the raw ones.

    Returns:
        None
    """
    if args.verbose:
        print(f'predictions: {example["predictions"]}')

    # A character-bag cosine over the top-1 prediction used to be accumulated
    # here. It was dropped: it measured letter overlap, so "China" against
    # "Chinatown" scored highly, and it was never reported.
    query_true = true_objects_for(example, true_objects, args.dataset)

    for target in example["targets"]:

        metric.total += 1
        metric.total2 += 1
        if query_true:
            metric.resolved += 1

        # the other approach of calculating metrics
        '''for i, pred in enumerate(example['output_text']):
          if target in pred:
            metric.update2(i+1)'''

        raw_rank, filtered_rank = ranks(
            example["predictions"], target,
            query_true - {normalize_entity(target)},
        )
        if args.verbose and raw_rank != math.inf:
            print(f"target: {target} --> rank: {raw_rank}  filtered: {filtered_rank}")
        metric.update(raw_rank, filtered_rank)
