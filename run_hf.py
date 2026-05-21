import json
import logging
import os

import torch
from tqdm import tqdm
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import logging as tf_logging
from peft import PeftModelForCausalLM
from model_utils import predict
from utils import (
    HitsMetric,
    get_args,
    get_filename,
    update_metric,
    write_results,
)
from datasets import load_dataset

tf_logging.set_verbosity_error()


# ---------------------------------------------------------------------------
# Token-length helpers
# ---------------------------------------------------------------------------

def get_token_limit(tokenizer_obj):
    token_limit = getattr(tokenizer_obj, "max_len_single_sentence", None)
    if token_limit is None or token_limit <= 0 or token_limit > 1_000_000:
        token_limit = getattr(tokenizer_obj, "model_max_length", None)
    if token_limit is None or token_limit <= 0 or token_limit > 1_000_000:
        raise ValueError("Could not infer a valid tokenizer max input length.")
    return int(token_limit)


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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
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

    if args.base_model == 'google/flan-t5-small':
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        model = T5ForConditionalGeneration.from_pretrained(
            models_path + args.finetuned_model,
            trust_remote_code=True,
            device_map='auto',
        )
        tokenizer = T5Tokenizer.from_pretrained(models_path + args.finetuned_model)
        tokenizer.pad_token = tokenizer.eos_token
        token_limit = get_token_limit(tokenizer)

        def count_tokens(text):
            return len(tokenizer(text, add_special_tokens=False).input_ids)

    elif args.base_model == 'TheBloke/Llama-2-7B-fp16':
        bnb4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype='float16',
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        training_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="auto",
            quantization_config=bnb4_config,
        )
        model = PeftModelForCausalLM.from_pretrained(
            training_model, models_path + args.finetuned_model
        )
        token_limit = get_token_limit(tokenizer)

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
    test_set = load_dataset('json', data_files=dataset_path, split='train')
    if args.dataset == 'icews18':
        test_set = test_set.select(range(0, 10000))

    test_set_rag = []
    indexes = []
    if args.dataset_rag_path:
        rag_dataset_path = f'./data/processed_new/{args.dataset}/' + args.dataset_rag_path
        test_set_rag = load_dataset('json', data_files=rag_dataset_path, split='train')

        rag_indexes_file = (
            'icews18_gpt_index.txt' if args.dataset == 'icews18' else 'icews14_gpt_index.txt'
        )
        with open(f'./data/processed_new/{args.dataset}/test_rag/{rag_indexes_file}') as f:
            indexes = eval(f.readline())

    # -----------------------------------------------------------------------
    # Output filename and logging
    # -----------------------------------------------------------------------

    model_name_for_file = args.finetuned_model or openai_model_name or 'unknown'

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

    if model is not None:
        logger.info("Model loaded — base: %s  device: %s", args.base_model, model.device.type)
    else:
        logger.info("OpenAI model: %s", openai_model_name)
    logger.info("Token limit : %d", token_limit)
    logger.info("Output file : %s", filename)

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
    overall_len_stats = init_length_stats()
    below_len_stats   = init_length_stats()
    above_len_stats   = init_length_stats()
    counter_above_limit = 0

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
            update_length_stats(overall_len_stats, input_token_len)

            if input_token_len <= token_limit:
                update_length_stats(below_len_stats, input_token_len)
            else:
                counter_above_limit += 1
                update_length_stats(above_len_stats, input_token_len)
                if args.tail_truncate_long_inputs and model is not None:
                    # Re-tokenise with the underlying HF tokenizer, keep the tail
                    encoded = tokenizer(model_input, add_special_tokens=False)
                    model_input_ids = encoded['input_ids'][-token_limit:]
                    model_input = tokenizer.decode(
                        model_input_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

            # Run prediction
            if args.base_model == 'openai':
                predictions = predict_openai(openai_client, openai_model_name, model_input)
                # logger.info(
                #     "OpenAI call [sample %d] — prompt tokens: %d\nPROMPT:\n%s\nPREDICTION: %s",
                #     i, input_token_len, model_input, predictions[0],
                # )
            else:
                predictions = predict(tokenizer, model, model_input, args)

            # Parse the query line to extract entity / relation / time
            if test_set_rag and i in indexes:
                time  = query_line.split(":")[1].strip()
                triple = query_line.split(":")[2].strip()
            else:
                time, triple = query_line.split(':', 1)

            triple = triple.strip()
            obj, rel = triple.split(' ')
            obj = obj.strip()[1:-1]   # strip leading '[' and trailing ','
            rel = rel.strip()[:-1]    # strip trailing ','

            quad    = [obj, rel, [x['target']], time.strip()]
            example = write_results(quad, predictions, 'tail', writer, args)
            update_metric(example, metric, args)
            pbar.set_postfix(metric.dump())

    # -----------------------------------------------------------------------
    # Post-run statistics
    # -----------------------------------------------------------------------

    logger.info("Samples with input above token limit (%d): %d",
                token_limit, counter_above_limit)
    logger.info("Input token length statistics (runtime):\n%s",
                json.dumps(
                    {
                        "token_limit": token_limit,
                        "tail_truncate_long_inputs": args.tail_truncate_long_inputs,
                        "overall": finalize_length_stats(overall_len_stats),
                        "below_or_equal_limit": finalize_length_stats(below_len_stats),
                        "above_limit": finalize_length_stats(above_len_stats),
                    },
                    indent=2,
                ))
    logger.info("Final metrics: %s", json.dumps(metric.dump(), indent=2))
