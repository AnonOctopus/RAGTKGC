import json
import logging
import os
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import logging as tf_logging
from peft import PeftModelForCausalLM
from model_utils import max_target_tokens, predict
from utils import (
    HitsMetric,
    apply_prompt_prefix,
    get_args,
    get_filename,
    load_true_objects,
    update_metric,
    write_results,
)
from datasets import load_dataset

tf_logging.set_verbosity_error()


# ---------------------------------------------------------------------------
# Token-length helpers
# ---------------------------------------------------------------------------

def get_token_limit(tokenizer_obj):
    """Read the max input length in tokens from a tokenizer.

    Args:
        tokenizer_obj: a loaded tokenizer.

    Returns:
        tuple[int, str]: the limit, and which attribute it came from —
            "max_len_single_sentence" or "model_max_length".

    Raises:
        ValueError: neither attribute holds a usable value.
    """
    # Above 1_000_000 means unspecified: transformers stores 1e30 when a
    # tokenizer config omits the field.
    for source in ("max_len_single_sentence", "model_max_length"):
        value = getattr(tokenizer_obj, source, None)
        if value is not None and 0 < value <= 1_000_000:
            return int(value), source
    raise ValueError("No usable max input length on the tokenizer.")


def init_length_stats():
    return {"count": 0, "sum": 0, "min": None, "max": None}


def update_length_stats(stats, value):
    stats["count"] += 1
    stats["sum"] += value
    stats["min"] = value if stats["min"] is None else min(stats["min"], value)
    stats["max"] = value if stats["max"] is None else max(stats["max"], value)


def finalize_length_stats(stats):
    if stats["count"] == 0:
        return {"count": 0, "avg": 0.0, "min": None, "max": None}
    return {
        "count": stats["count"],
        "avg": round(stats["sum"] / stats["count"], 2),
        "min": stats["min"],
        "max": stats["max"],
    }


# ---------------------------------------------------------------------------
# OpenAI prediction helper
# ---------------------------------------------------------------------------

_OPENAI_SYSTEM_PROMPT = (
    "You are an expert in temporal knowledge graph completion (TKGC). "
    "A temporal knowledge graph stores facts as quadruples of the form "
    "(subject, relation, object, timestamp). "
    "Your task is to predict the missing object (tail entity) of a query quadruple "
    "given a set of relevant historical facts retrieved from the knowledge graph. "
    "These historical facts are ordered from oldest to most recent and were selected "
    "because they are likely to provide evidence for the missing entity. "
    "Respond with the predicted entity name only — no explanation, no punctuation, "
    "no surrounding brackets."
)


def predict_openai(client, model_name, prompt):
    """Call the OpenAI Responses API and return a single-element prediction list."""
    response = client.responses.create(
        model=model_name,
        instructions=_OPENAI_SYSTEM_PROMPT,
        input=prompt,
    )
    raw = response.output_text.strip()
    # Strip artefacts that appear in history-completion style outputs
    prediction = raw.replace(']', '').replace('</s>', '').split('\n')[0].strip()
    return [prediction]


# ---------------------------------------------------------------------------
# Logging setup  (called once filename is known)
# ---------------------------------------------------------------------------

def _setup_logging(log_path):
    # force=True is required, not cosmetic: basicConfig is a no-op when the root
    # logger already has handlers. In a notebook session that evaluates several
    # configurations in one kernel, every run after the first would otherwise keep
    # writing to the FIRST run's log file, making log filenames unreliable for
    # identifying which run produced which numbers.
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
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    args = get_args()
    models_path = './models/'

    # -----------------------------------------------------------------------
    # Model / tokenizer initialisation
    # -----------------------------------------------------------------------

    openai_client = None
    openai_model_name = None
    openai_enc = None
    # Set to the reason when the checkpoint's own tokenizer could not be
    # loaded and the base model's was used instead.
    tokenizer_fallback = None

    if args.base_model == 'google/flan-t5-small':
        from transformers import T5TokenizerFast, T5ForConditionalGeneration

        model = T5ForConditionalGeneration.from_pretrained(
            models_path + args.finetuned_model,
            trust_remote_code=True,
            device_map='auto',
        )
        # Fast (Rust) tokenizer, matching training_T5.py. Verified to emit
        # identical ids to the Python T5Tokenizer on this corpus, at roughly an
        # order of magnitude less wall-clock — which the pre-scan pass pays for
        # over the whole test set.
        # TODO: delete this fallback once every T5 model has been retrained.
        # It exists only for checkpoints saved before the fast tokenizer was
        # adopted. Once they are gone, a tokenizer that will not load means the
        # checkpoint is broken and should raise rather than be substituted.
        try:
            tokenizer = T5TokenizerFast.from_pretrained(models_path + args.finetuned_model)
        except (OSError, ValueError, ImportError, TypeError) as exc:
            # Checkpoints saved before the fast tokenizer was adopted ship only
            # spiece.model, and converting it needs protobuf — which raises
            # TypeError, not ImportError, when its generated code is stale.
            # Fine-tuning does not change a T5 tokenizer — the vocabulary and
            # the sentinel tokens are the base model's — so the base one is a
            # substitute.
            # Deferred, not printed: the logger is configured further down,
            # once the filename is known, and which tokenizer produced a
            # result has to be in the run log, not only on the terminal.
            tokenizer_fallback = str(exc).splitlines()[0]
            tokenizer = T5TokenizerFast.from_pretrained(args.base_model)
        # No pad_token override here: T5 has a real <pad> (id 0). Pointing pad
        # at </s> is the LLaMA convention, where no pad token exists. It is
        # inert at batch size 1, but it would pad T5 encoder inputs with </s>
        # once the prediction loop is batched.
        token_limit, token_limit_source = get_token_limit(tokenizer)

        def count_tokens(text):
            return len(tokenizer(text, add_special_tokens=False).input_ids)

    elif args.base_model == 'TheBloke/Llama-2-7B-fp16':
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if args.precision == '4bit':
            bnb4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype='float16',
                bnb_4bit_use_double_quant=True,
            )
            training_model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                device_map="auto",
                quantization_config=bnb4_config,
            )
        else:
            # transformers 5 renamed from_pretrained's `torch_dtype` to
            # `dtype`; both spellings are tried so one script runs on either.
            try:
                training_model = AutoModelForCausalLM.from_pretrained(
                    args.base_model, device_map="auto", dtype=torch.bfloat16)
            except TypeError:
                training_model = AutoModelForCausalLM.from_pretrained(
                    args.base_model, device_map="auto",
                    torch_dtype=torch.bfloat16)
        model = PeftModelForCausalLM.from_pretrained(
            training_model, models_path + args.finetuned_model
        )
        token_limit, token_limit_source = get_token_limit(tokenizer)

        def count_tokens(text):
            return len(tokenizer(text, add_special_tokens=False).input_ids)

    elif args.base_model == 'openai':
        from dotenv import load_dotenv
        from openai import OpenAI as _OpenAIClient
        import tiktoken

        load_dotenv()
        openai_client = _OpenAIClient(api_key=os.environ['OPENAI_API_KEY'])
        openai_model_name = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')
        try:
            openai_enc = tiktoken.encoding_for_model(openai_model_name)
        except KeyError:
            openai_enc = tiktoken.get_encoding("cl100k_base")
        # Context window for common GPT-4 class models
        token_limit = 128_000
        token_limit_source = "hardcoded"

        def count_tokens(text):
            return len(openai_enc.encode(text))

        model = None  # no local model for OpenAI

    else:
        raise ValueError(f"Unsupported base_model: {args.base_model!r}")

    if model is not None:
        model.eval()

    # -----------------------------------------------------------------------
    # Dataset loading
    # -----------------------------------------------------------------------

    dataset_path = f'./data/processed_new/{args.dataset}/' + args.dataset_path
    # force_redownload re-reads the JSON instead of trusting the loader's cache.
    # Test sets are regenerated in place when filters change, and evaluating a
    # stale one would silently report the previous variant's numbers.
    test_set = load_dataset('json', data_files=dataset_path, split='train',
                            download_mode='force_redownload')
    if args.dataset == 'icews18':
        test_set = test_set.select(range(0, 10000))
    if args.limit:
        # Short by construction, so compute_metrics_from_results.py skips the
        # file rather than reporting a metric over a subset as if it were whole.
        test_set = test_set.select(range(0, min(args.limit, len(test_set))))

    test_set_rag = []
    indexes = []
    if args.dataset_rag_path:
        rag_dataset_path = f'./data/processed_new/{args.dataset}/' + args.dataset_rag_path
        test_set_rag = load_dataset('json', data_files=rag_dataset_path, split='train',
                                    download_mode='force_redownload')

        rag_indexes_file = (
            'icews18_gpt_index.txt' if args.dataset == 'icews18' else 'icews14_gpt_index.txt'
        )
        with open(f'./data/processed_new/{args.dataset}/test_rag/{rag_indexes_file}') as f:
            indexes = json.loads(f.readline())

    # -----------------------------------------------------------------------
    # Output filename and logging
    # -----------------------------------------------------------------------

    # On the OpenAI path the local-model flag is irrelevant, and letting it win
    # would name the results file after a model that never ran.
    if args.base_model == 'openai':
        model_name_for_file = openai_model_name or 'unknown-openai'
    else:
        model_name_for_file = args.finetuned_model or 'unknown'

    if args.dataset_rag_path:
        filename = get_filename(
            args.dataset,
            dataset_path=args.dataset_rag_path,
            model_name=model_name_for_file,
            tail_truncate_included=args.tail_truncate_long_inputs,
        )
    else:
        filename = get_filename(
            args.dataset,
            dataset_path=args.dataset_path,
            model_name=model_name_for_file,
            tail_truncate_included=args.tail_truncate_long_inputs,
        )

    logs_dir = './logs/'
    os.makedirs(logs_dir, exist_ok=True)
    log_basename = os.path.splitext(os.path.basename(filename))[0] + '.log'
    log_path = os.path.join(logs_dir, log_basename)
    logger = _setup_logging(log_path)

    # Record the full invocation. Without this a results file cannot be traced
    # back to the flags that produced it: get_filename keeps only the JSON
    # basename and write_results does not store the prompt.
    logger.info("Args: %s", json.dumps(vars(args), indent=2, default=str))

    # Windows resolves to a 260-character path limit unless long paths are
    # enabled, and these names already run to roughly 240. Warn before a longer
    # variant name turns into an opaque file-creation error.
    _abs_out = os.path.abspath(filename)
    if len(_abs_out) > 240:
        logger.warning(
            "Output path is %d characters, close to the 260-character Windows "
            "limit: %s", len(_abs_out), _abs_out,
        )

    # Append the invocation to a per-dataset manifest. The configuration cannot
    # be encoded in the filename without exceeding that path limit, so this is
    # what makes a results file traceable after the fact. Append-only, so a
    # numbered re-run still leaves the earlier record intact.
    manifest_path = os.path.join('./results', args.dataset, '_runs.jsonl')
    try:
        with open(manifest_path, 'a', encoding='utf-8') as mf:
            mf.write(json.dumps({
                "started": datetime.now().isoformat(timespec='seconds'),
                "results_file": os.path.basename(filename),
                "log_file": log_basename,
                "args": vars(args),
            }, default=str) + "\n")
        logger.info("Run recorded in %s", manifest_path)
    except OSError as exc:
        logger.warning("Could not write run manifest %s: %s", manifest_path, exc)

    if model is not None:
        logger.info("Model loaded — base: %s  device: %s", args.base_model, model.device.type)
        if args.base_model == 'TheBloke/Llama-2-7B-fp16':
            logger.info("Base precision: %s (must match the adapter's training)",
                        args.precision)
    else:
        logger.info("OpenAI model: %s", openai_model_name)
    logger.info("Token limit : %d  (source: %s)", token_limit, token_limit_source)
    logger.info("Output file : %s", filename)
    if tokenizer_fallback:
        logger.warning("Tokenizer taken from the base model %s, not the checkpoint: %s",
                       args.base_model, tokenizer_fallback)

    # Generation budget. Beam search truncates at this length, so it has to
    # cover the longest target; the OpenAI path does its own generation and
    # never reads it.
    if args.base_model == 'openai':
        max_new_tokens = None
    elif args.max_new_tokens is not None:
        max_new_tokens = args.max_new_tokens
        logger.info("max_new_tokens: %d (pinned)", max_new_tokens)
    else:
        max_new_tokens = max_target_tokens(test_set, tokenizer)
        logger.info("max_new_tokens: %d (longest target in the split, plus margin)",
                    max_new_tokens)
    if args.base_model != 'openai':
        logger.info("Beam width    : %d", args.num_beams)

    # -----------------------------------------------------------------------
    # Pre-scan: compute input token-length statistics BEFORE predictions
    # -----------------------------------------------------------------------

    logger.info("Pre-scanning dataset for input token length statistics (%d samples)...",
                len(test_set))
    pre_scan_stats = init_length_stats()

    for j, x in enumerate(test_set):
        sample = x
        if test_set_rag and j in indexes:
            sample = test_set_rag[indexes.index(j)]
        update_length_stats(pre_scan_stats, count_tokens(sample['context']))

    logger.info("Input token length statistics (pre-scan):\n%s",
                json.dumps(
                    {
                        "token_limit": token_limit,
                        "tail_truncate_long_inputs": args.tail_truncate_long_inputs,
                        **finalize_length_stats(pre_scan_stats),
                    },
                    indent=2,
                ))

    # -----------------------------------------------------------------------
    # Prediction loop
    # -----------------------------------------------------------------------

    metric = HitsMetric()

    # Time-aware filtered ranking needs every true object of each
    # (subject, relation, day). Without it the filtered figures fall back to the
    # raw ones rather than silently reporting a wrong number.
    try:
        true_objects = load_true_objects(args.dataset)
        logger.info("Fact index for filtered ranking: %d (subject, relation, day) keys",
                    len(true_objects))
    except (OSError, json.JSONDecodeError) as exc:
        true_objects = None
        logger.warning("Filtered ranking disabled — could not build the fact index (%s). "
                       "The f_hit* figures will equal the raw ones.", exc)
    # No overall stat here: the pre-scan above measured exactly this — same
    # count_tokens, same contexts, both before truncation — so recomputing it
    # in the loop would be a second full tokenisation pass for identical
    # numbers. The post-run report reuses pre_scan_stats.
    below_len_stats   = init_length_stats()
    above_len_stats   = init_length_stats()
    # Length of what the model actually receives. Both the pre-scan and
    # input_token_len are measured before truncation, so neither shows this.
    effective_len_stats = init_length_stats()
    # Distinct candidates per query. Beam search returns num_beams sequences,
    # but cleaning collapses some onto the same string, and a mean well below
    # num_beams caps H@10 from below.
    candidate_stats = init_length_stats()
    counter_above_limit = 0

    # The instruction block is prepended after truncation, so its cost has to
    # come out of the budget first or the wrapped prompt exceeds the limit.
    prefix_tokens = count_tokens(apply_prompt_prefix('')) if args.prompt_prefix else 0
    history_limit = token_limit - prefix_tokens
    if args.prompt_prefix:
        logger.info("Prompt prefix costs %d tokens; history truncated to %d",
                    prefix_tokens, history_limit)
        if history_limit <= 0:
            raise SystemExit(
                f"The instruction block alone is {prefix_tokens} tokens, at or "
                f"over the model's {token_limit}-token limit. --prompt_prefix "
                "cannot be used with this model.")

    with (
        torch.no_grad(),
        open(filename, "w", encoding="utf-8") as writer,
        tqdm(test_set) as pbar,
    ):
        for i, x in enumerate(pbar):

            if test_set_rag and i in indexes:
                x = test_set_rag[indexes.index(i)]

            model_input = x['context']
            query_line  = x['context'].split('\n')[-1]

            input_token_len = count_tokens(model_input)

            # Stays equal to input_token_len unless the sample is truncated below.
            effective_token_len = input_token_len

            if input_token_len <= history_limit:
                update_length_stats(below_len_stats, input_token_len)
            else:
                counter_above_limit += 1
                update_length_stats(above_len_stats, input_token_len)
                if args.tail_truncate_long_inputs and model is not None:
                    # Re-tokenise with the underlying HF tokenizer, keep the tail
                    encoded = tokenizer(model_input, add_special_tokens=False)
                    model_input_ids = encoded['input_ids'][-history_limit:]
                    model_input = tokenizer.decode(
                        model_input_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    # Measure the decoded string, not len(model_input_ids).
                    # Decode-then-encode is not round-trip exact — the slice can
                    # cut mid-subword — so the truncated prompt can come back
                    # slightly longer than token_limit. Reporting the sliced id
                    # count would always read as exactly token_limit and hide
                    # that. Same add_special_tokens=False convention as every
                    # other length here; predict() adds one EOS on top.
                    effective_token_len = count_tokens(model_input)

            # After truncation, so the instruction block survives it; its cost
            # was already reserved out of history_limit above.
            if args.prompt_prefix:
                model_input = apply_prompt_prefix(model_input)
                effective_token_len += prefix_tokens

            update_length_stats(effective_len_stats, effective_token_len)

            # Run prediction
            if args.base_model == 'openai':
                predictions = predict_openai(openai_client, openai_model_name, model_input)
                # logger.info(
                #     "OpenAI call [sample %d] — prompt tokens: %d\nPROMPT:\n%s\nPREDICTION: %s",
                #     i, input_token_len, model_input, predictions[0],
                # )
            else:
                predictions = predict(tokenizer, model, model_input, args, max_new_tokens)

            update_length_stats(candidate_stats, len(predictions))

            # Parse the query line to extract entity / relation / time
            if test_set_rag and i in indexes:
                time  = query_line.split(":")[1].strip()
                triple = query_line.split(":")[2].strip()
            else:
                time, triple = query_line.split(':', 1)

            triple = triple.strip()
            obj, rel = triple.split(' ')
            # Strip the delimiters only when present. The query line normally
            # ends the relation with a comma, but not every variant's format
            # does; an unconditional [:-1] eats a real character off the
            # relation name (this is how `standard` recorded a truncated name).
            obj = obj.strip().lstrip('[').rstrip(',')
            rel = rel.strip().rstrip(',')

            quad    = [obj, rel, [x['target']], time.strip()]
            example = write_results(quad, predictions, 'tail', writer, args)
            update_metric(example, metric, args, true_objects)
            pbar.set_postfix(metric.dump())

    # -----------------------------------------------------------------------
    # Post-run statistics
    # -----------------------------------------------------------------------

    logger.info("Samples with input above the history budget (%d of %d tokens): %d",
                history_limit, token_limit, counter_above_limit)
    logger.info("Input token length statistics (runtime):\n%s",
                json.dumps(
                    {
                        "token_limit": token_limit,
                        "history_limit": history_limit,
                        "prompt_prefix_tokens": prefix_tokens,
                        "tail_truncate_long_inputs": args.tail_truncate_long_inputs,
                        "overall": finalize_length_stats(pre_scan_stats),
                        "below_or_equal_limit": finalize_length_stats(below_len_stats),
                        "above_limit": finalize_length_stats(above_len_stats),
                        "effective_after_truncation": finalize_length_stats(effective_len_stats),
                    },
                    indent=2,
                ))
    logger.info("Distinct candidates per query:\n%s",
                json.dumps(
                    {
                        "num_beams": args.num_beams,
                        **finalize_length_stats(candidate_stats),
                    },
                    indent=2,
                ))
    # A query that does not resolve against the fact index contributes a
    # filtered rank equal to its raw rank, so a low resolved share means the
    # f_hit* figures are mostly raw and must not be reported as filtered.
    if metric.total:
        logger.info("Queries resolved against the fact index: %d/%d (%.1f%%)",
                    metric.resolved, metric.total,
                    100 * metric.resolved / metric.total)
    logger.info("Final metrics: %s", json.dumps(metric.dump(), indent=2))
