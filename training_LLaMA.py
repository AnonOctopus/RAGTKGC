# import the required classes, feel free to add any other PeftConfig type

import argparse
import json
import logging
import os
import time
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, DataCollatorForLanguageModeling, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from typing import Any, Dict, List, Union

from training_controller import TrainingControllerCallback
from utils import apply_prompt_prefix

# Set from args in __main__; read by process_function, which datasets.map
# calls with the batch only.
PROMPT_PREFIX = False

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--trained_model_name",type = str)
    parser.add_argument("--output_dir", default = './models', type = str)
    parser.add_argument("--train_file_path", type = str)
    parser.add_argument(
        "--eval_file_path",
        type=str,
        default=None,
        help=(
            "Optional path (relative to ./data/processed_new/<dataset>/) to a validation "
            "file. When provided, validation loss is computed during training and feeds "
            "both best-checkpoint selection and the TrainingControllerCallback. Without "
            "it there is no signal at all for whether the run under- or overfits."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for training. Controls dropout and the order in which the training set "
            "is shuffled. Defaults to 42, the same value Trainer uses when unset, so omitting it "
            "reproduces earlier runs."
        ),
    )

    # Must match the flag used at evaluation, or the model is fine-tuned on a
    # prompt shape it never sees again.
    parser.add_argument(
        "--prompt_prefix",
        default=False,
        action="store_true",
        help=("Wrap each context in the baseline's instruction block. Assumes "
              "targets are in id.Name form."),
    )

    # --- tuning surface -----------------------------------------------------
    # Exposed so a hyperparameter search is a series of commands rather than a
    # series of edits: nothing about the run can then differ from what the log
    # records, and no sweep can silently drift from the code it was meant to test.
    parser.add_argument(
        "--lora_r", type=int, default=8,
        help="LoRA rank. Higher adapts more but has more to overfit with.",
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=None,
        help=("LoRA scaling numerator; the update is multiplied by alpha/r. "
              "Defaults to 2*lora_r, keeping the effective scale at 2.0 so rank "
              "can be changed without also changing the update magnitude."),
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05,
        help="Dropout on the LoRA input. Raise it only if the eval curve turns up.",
    )
    parser.add_argument(
        "--lora_target_modules", type=str, default="all",
        help=("Comma-separated module names to adapt, or 'all' for every linear "
              "layer in the block (attention and MLP), or 'attention' for the "
              "q/v projections PEFT would pick by default."),
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4,
        help="Peak learning rate. Interacts with the LoRA scale, so change one at a time.",
    )
    parser.add_argument(
        "--num_train_epochs", type=float, default=1,
        help="Passes over the training file. The dominant term in run cost.",
    )
    parser.add_argument(
        "--eval_steps", type=int, default=16,
        help=("Optimiser steps between validations. The default gives about eight "
              "points across the reference protocol's ~128 steps; a screening run "
              "on fewer samples needs a smaller value to produce a curve at all."),
    )
    parser.add_argument(
        "--max_eval_samples", type=int, default=512,
        help=("Cap the validation set. It is evaluated once per eval_steps, so an "
              "uncapped split costs far more than the training it is there to "
              "monitor: the full ICEWS14 validation file is 8,514 samples against "
              "1,024 training samples, and at batch 1 that is most of the run. "
              "The cap is a prefix of a file that is already a seeded draw. Pass 0 "
              "to evaluate on everything."),
    )
    parser.add_argument(
        "--max_train_samples", type=int, default=None,
        help=("Truncate the training set to the first N samples after tokenizing. "
              "For screening hyperparameters at a fraction of the compute before "
              "confirming the winner on the full file. Not a sampling method: the "
              "file is already a seeded random draw, so this takes a prefix of it."),
    )

    args = parser.parse_args()
    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_r
    return args

def process_function(examples):
    """Tokenize a batch and record where each answer begins.

    The training text is the prompt, a space, and the answer. Two things are
    deliberate. No closing "]" is appended: the bracket makes the tokenizer glue
    itself onto the answer's final character for most entities, which leaves no
    token holding only the answer and forces the label to be patched afterwards.
    And EOS is appended by id rather than written into the text, because a
    literal "</s>" parses as the special token only when a space or bracket
    precedes it — glued to the answer it silently becomes '</', 's', '>', and the
    model never learns to stop.

    The answer's token span is taken from the tokenizer's own character offsets
    while both halves of the text are still known here, so the collator does not
    have to recover it by decoding and re-encoding. target_start points at the
    space before the answer, which the model must also generate.

    Args:
        examples: batch with 'context' and 'target' string columns.

    Returns:
        dict: input_ids, attention_mask, special_tokens_mask and target_start,
            one entry per example.
    """
    # PROMPT_PREFIX is set from args in __main__, as `tokenizer` is: datasets.map
    # calls this with the batch only.
    contexts = [apply_prompt_prefix(c) if PROMPT_PREFIX else c
                for c in examples['context']]
    texts = [f"{context} {target}"
             for context, target in zip(contexts, examples['target'])]

    encoded = tokenizer(texts, return_special_tokens_mask=True,
                        return_offsets_mapping=True)

    inputs = {"input_ids": [], "attention_mask": [],
              "special_tokens_mask": [], "target_start": []}
    for i, context in enumerate(contexts):
        offsets = encoded["offset_mapping"][i]
        # First token reaching past the end of the context: the space, then the
        # answer. Empty-span tokens are special tokens and are skipped.
        start = next(j for j, (lo, hi) in enumerate(offsets)
                     if hi > lo and hi > len(context))
        inputs["input_ids"].append(encoded["input_ids"][i] + [tokenizer.eos_token_id])
        inputs["attention_mask"].append(encoded["attention_mask"][i] + [1])
        inputs["special_tokens_mask"].append(encoded["special_tokens_mask"][i] + [1])
        inputs["target_start"].append(start)

    return inputs


# Defined at module level rather than inside __main__ so it can be imported and
# exercised without starting a training run — the supervised span it computes is
# the subject of F60 and cannot be checked otherwise.
class DataCollatorForCompletionLM(DataCollatorForLanguageModeling):
    """Mask every position before the answer out of the loss.

    A causal LM would otherwise be trained on the whole sequence, and the prompt
    is around a hundred times longer than the answer, so almost all of the
    gradient would come from reproducing history the model was handed anyway.

    The span is not searched for here. process_function records where the answer
    starts while it still has the untokenized text, and since the sequence ends
    with the answer followed by EOS, a start is all that is needed.

    Args:
        tokenizer: the tokenizer the batch was encoded with.
        **kwargs: forwarded to DataCollatorForLanguageModeling.
    """

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        starts = [example["target_start"] for example in examples]
        # target_start is ours, not the model's; the base collator would try to
        # pad it into a tensor.
        payload = [{k: v for k, v in example.items() if k != "target_start"}
                   for example in examples]

        batch = super().torch_call(payload)
        labels = batch["labels"].clone()

        for i, start in enumerate(starts):
            labels[i, :start] = -100
            # pad_token is eos_token for this model, so the base collator masked
            # every EOS id as padding — including the real final token, which is
            # the one that teaches the model to stop. Restore it at the last
            # unpadded position, which is correct at any batch size.
            last = int(batch["attention_mask"][i].sum()) - 1
            labels[i, last] = self.tokenizer.eos_token_id

        batch["labels"] = labels

        return batch


def _length_stats(tokenized):
    """Prompt length distribution and how much of it is supervised.

    The supervised share is the signal-to-noise ratio of the objective: with no
    masking it would be 100% and almost all of it history the model was given.

    Args:
        tokenized: mapped dataset carrying input_ids and target_start.

    Returns:
        dict: token-count percentiles and the mean supervised token count.
    """
    lengths = sorted(len(row) for row in tokenized["input_ids"])
    answers = [len(ids) - start for ids, start
               in zip(tokenized["input_ids"], tokenized["target_start"])]
    n = len(lengths)
    return {
        "count": n,
        "mean": round(sum(lengths) / n, 1),
        "min": lengths[0],
        "p50": lengths[n // 2],
        "p90": lengths[int(n * 0.9)],
        "p99": lengths[int(n * 0.99)],
        "max": lengths[-1],
        "supervised_tokens_mean": round(sum(answers) / len(answers), 2),
        "supervised_share": f"{sum(answers) / sum(lengths):.2%}",
    }


def _setup_logging(log_path):
    """Log to a file and to stdout at once.

    The file is what gets carried back from a remote session; the stream is what
    makes the run watchable while it happens.

    Args:
        log_path: file to append the run's log to.

    Returns:
        logging.Logger: configured logger.
    """
    # force=True: basicConfig is a no-op when the root logger already has
    # handlers, so in a notebook kernel that trains more than once every run
    # after the first would keep writing to the first run's file.
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

    args = parser()

    # Fail before training rather than after: the save path is built from
    # trained_model_name, so omitting it would only surface at the very end.
    if not args.trained_model_name:
        raise SystemExit("--trained_model_name is required; it names the output adapter directory.")

    # Feel free to set your own values.

    model = 'TheBloke/Llama-2-7B-fp16' # the model to be finetuned
    trained_model_name = args.trained_model_name # the name of the trained model
    # Per-run checkpoint directory. Trainer names checkpoints by optimiser step,
    # so runs that share a training-set size collide in a shared directory.
    output_dir = os.path.join(
        args.output_dir, trained_model_name.replace("'", "").replace('"', '')
    )

    os.makedirs('./logs/', exist_ok=True)
    logger = _setup_logging(f"./logs/train_llama_{trained_model_name}.log")
    logger.info("=== Run configuration ===")
    logger.info("Arguments: %s", json.dumps(vars(args), indent=2))
    logger.info("Base model: %s", model)
    logger.info("Output directory: %s", output_dir)

    # Which machine produced these numbers — a remote session is not reproducible
    # from the arguments alone.
    try:
        import torch
        import transformers
        logger.info(
            "Environment: torch %s, transformers %s, CUDA %s, device %s",
            torch.__version__, transformers.__version__,
            torch.version.cuda,
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        )
        if torch.cuda.is_available():
            logger.info("GPU memory: %.1f GiB total",
                        torch.cuda.get_device_properties(0).total_memory / 2 ** 30)
    except (ImportError, RuntimeError, AssertionError) as exc:
        logger.warning("Could not read the environment: %s", exc)

    # feel free to add any other parameter

    # Every linear layer in the decoder block, not the q/v pair PEFT defaults to.
    # The default is the original LoRA paper's; adapting the projections it omits
    # — o_proj and the whole MLP, which is most of the parameters — is the change
    # QLoRA identifies as mattering more than rank for approaching full
    # fine-tuning. 'attention' restores the default for a like-for-like check.
    _TARGET_MODULES = {
        "all": ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"],
        "attention": ["q_proj", "v_proj"],
    }
    target_modules = _TARGET_MODULES.get(
        args.lora_target_modules,
        [m.strip() for m in args.lora_target_modules.split(",") if m.strip()],
    )

    lora_config = LoraConfig(r = args.lora_r, #As bigger the R bigger the parameters to train.
                        lora_alpha=args.lora_alpha, # a scaling factor that adjusts the magnitude of the weight matrix.
                        lora_dropout=args.lora_dropout, #Helps to avoid Overfitting.
                        target_modules=target_modules,
                        # 'none', not 'lora_only': Llama-2 builds its attention and
                        # MLP projections without bias terms, so there are no
                        # biases for 'lora_only' to train. It cost nothing, but it
                        # read as if it were doing something. The trainable
                        # parameter count logged below is the check.
                        bias='none',
                        task_type='CAUSAL_LM')
                        # use_rslora is deliberately not set. It divides by sqrt(r)
                        # instead of r, which exists to stop high ranks being
                        # under-scaled; at r=8 with alpha=16 it only raises the
                        # update scale from 2.0 to 5.66, compounding with the
                        # learning rate. It is also absent from the pinned
                        # peft==0.7.1, where passing it raises TypeError.
    
    # Adaptive training controller — mirrors the Flan-T5 arm so the two regimes
    # differ in model and adapter, not in how training is supervised. Active only
    # when --eval_file_path supplies a validation set.
    controller = TrainingControllerCallback(
        min_delta=5e-4,
        patience=2,
        lr_reduction_factor=0.5,
        max_lr_reductions=2,
        ema_alpha=0.4,
        min_grad_norm=1e-2,
        grad_snr_threshold=1.0,
        min_marginal_improvement=1e-5,
    )

    _has_eval = bool(args.eval_file_path)
    # Evaluate on steps, not epochs. The reference protocol is 1024 samples for a
    # single epoch, which at batch 1 x grad-accum 8 is ~128 optimiser steps — so
    # an epoch-based strategy would evaluate exactly once, leaving the controller
    # nothing to compare and best-checkpoint selection nothing to choose between.
    # The default of every 16 steps gives ~8 evaluations across the run; a
    # screening run on fewer samples needs --eval_steps lowered to match. Note
    # this makes the controller's `patience` count evaluations, not epochs.
    _eval_steps = args.eval_steps

    # warm up: nr of steps where the lr starts from 0 and goes to initial set, to prevent oferfitting on early data
    training_args = TrainingArguments(output_dir = output_dir,
                                    # auto_find_batch_size is deliberately not set: it only ever
                                    # halves the batch size on OOM, and 1 is already the floor
                                    # (1 // 2 == 0), so it cannot help. What it does do is replace
                                    # a real OutOfMemoryError with accelerate's much less
                                    # informative "No executable batch size found, reached zero."
                                    per_device_train_batch_size=1,
                                    per_device_eval_batch_size = 1,
                                    learning_rate = args.learning_rate, # Higher learning rate than full fine-tuning.
                                    num_train_epochs = args.num_train_epochs,
                                    # A ratio, not a fixed 20 steps. The reference protocol is
                                    # ~128 optimiser steps, where 20 is a plausible 16%; a
                                    # screening run on 256 samples is 32 steps, where the same 20
                                    # is 62% of the run spent ramping up, and the learning rate
                                    # being compared would barely be reached. 0.06 matches the
                                    # Flan-T5 arm and scales with whatever the run turns out to be.
                                    warmup_ratio = 0.06,
                                    # Train loss logged wherever validation is measured, so the two
                                    # curves line up; at the screening default of 20 a short run
                                    # produced a single point.
                                    logging_steps  = _eval_steps,
                                    gradient_accumulation_steps=8,
                                    # Explicitly 0.0, not commented out. Flan-T5 uses 0.1 because
                                    # every one of its parameters is updated; here only the LoRA
                                    # adapter trains, for a few hundred steps, with its own dropout
                                    # already regularising it. L2 on that is close to inert, and
                                    # leaving the line disabled made a deliberate difference
                                    # between the two arms look like an oversight.
                                    weight_decay = 0.0,
                                    # target_start is carried on the dataset for the collator.
                                    # Trainer drops any column the model's forward() does not name,
                                    # which would remove it before the collator ever sees it.
                                    remove_unused_columns = False,
                                    # --- evaluation & checkpointing ---
                                    # Without an eval file nothing is saved during training; the
                                    # adapter is written once at the end (see the save block below).
                                    eval_strategy = 'steps' if _has_eval else 'no',
                                    eval_steps = _eval_steps if _has_eval else None,
                                    save_strategy = 'steps' if _has_eval else 'no',
                                    save_steps = _eval_steps if _has_eval else None,
                                    save_total_limit = 3,
                                    load_best_model_at_end = _has_eval,
                                    metric_for_best_model = 'eval_loss',
                                    greater_is_better = False,
                                    # data_seed is left to follow seed. This does not make a run
                                    # bit-exact: 4-bit quantisation and nondeterministic GPU
                                    # kernels still vary between runs at the same seed.
                                    seed = args.seed,
                                    report_to = 'none') # the list of integrations to report the results and logs to.
    logger.info("LoRA: %s", lora_config.to_dict() if hasattr(lora_config, "to_dict") else lora_config)
    logger.info("Training arguments: %s", json.dumps({
        k: training_args.to_dict()[k] for k in (
            "per_device_train_batch_size", "per_device_eval_batch_size",
            "gradient_accumulation_steps", "learning_rate", "weight_decay",
            "num_train_epochs", "warmup_steps", "lr_scheduler_type",
            "eval_strategy", "eval_steps", "save_strategy", "save_steps",
            "load_best_model_at_end", "metric_for_best_model", "seed",
        ) if k in training_args.to_dict()
    }, indent=2, default=str))
    
    # load the model in a 4bit configuration to use QLora; this is how we trained the models, but you can also load them in 8bit or full if enough resources are available

    bnb4_config =  BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type='nf4', # precision of the stored weights
                                bnb_4bit_compute_dtype='bfloat16', # precision of computations
                                bnb_4bit_use_double_quant=True
                                )
    
    # set your own quantization_config if desired

    training_model = AutoModelForCausalLM.from_pretrained(model,
                                                            trust_remote_code = True,
                                                            quantization_config = bnb4_config
                                                            ) # for training, device_map does not have to be set. check -> https://huggingface.co/docs/transformers/v4.35.0/main_classes/quantization#bitsandbytes-integration
    PROMPT_PREFIX = args.prompt_prefix
    logger.info("Prompt prefix: %s (must match the flag used at evaluation)",
                "on" if PROMPT_PREFIX else "off")

    tokenizer = AutoTokenizer.from_pretrained(model)

    tokenizer.pad_token = tokenizer.eos_token
    # Llama's tokenizer pads on the left, which is what batched generation wants:
    # it keeps the prompts flush against the tokens being generated. Training
    # wants the opposite. target_start and the EOS position are both indices into
    # the unpadded sequence, and left padding shifts every real token right by
    # the pad count, so the mask would land on the prompt instead of the answer.
    # With batch size 1 nothing is padded and the difference never shows.
    tokenizer.padding_side = 'right'

    # If quantization is applied, enable the line, else disable it.
    training_model = prepare_model_for_kbit_training(training_model) # This method wraps the entire protocol for preparing a model before running a training.

    # If PEFT is desired, then get the peft version of the model, else disable it.
    training_model = get_peft_model(training_model, lora_config, low_cpu_mem_usage = False) # feel free to put any config file from above. low_cpu_mem_usage — Create empty adapter weights on meta device. Useful to speed up the loading process. Leave this setting as False if you intend on training the model -> https://huggingface.co/docs/peft/package_reference/peft_model

    trainable = sum(p.numel() for p in training_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in training_model.parameters())
    logger.info("Trainable parameters: %s of %s (%.4f%%)",
                f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    # load a dataset from the data files
    # IMPORTANT: for ICEWS18, there are two training folders: 1024 and full. Use 1024 for finetuning LLaMA2-7B and full for Flan-T5-Small

    # force_redownload re-reads the JSON instead of trusting the loader's cache.
    # The pipeline regenerates these files in place, and older `datasets`
    # versions have served the previous contents for an unchanged path.
    dataset = load_dataset('json', data_files=f'./data/processed_new/{args.dataset}/{args.train_file_path}', split = 'train',
                           download_mode='force_redownload')

    tokenized_input = dataset.map(process_function, batched = True, remove_columns=dataset.column_names)

    # LLaMA-2's context is 4096. Unlike T5's relative position buckets, RoPE is
    # only trained to that length and extrapolates poorly past it, so this is a
    # hard limit rather than a loss of resolution.
    #
    # The count is reported rather than silently applied. Dropping samples for
    # length biases whatever is trained on what survives — toward short
    # histories — so a non-zero count here means the sampling below is no longer
    # drawing from the full distribution, and the fix is to truncate rather than
    # discard. Measured at num_facts=50 with the instruction prefix, the longest
    # prompt is well inside the limit and nothing is dropped.
    _before_filter = len(tokenized_input)
    tokenized_input = tokenized_input.filter(lambda x: len(x['input_ids']) < 4095)
    _dropped = _before_filter - len(tokenized_input)
    if _dropped:
        logger.warning(
            "%d of %d training samples (%.2f%%) exceed 4095 tokens and were "
            "discarded. The remaining sample is biased toward short histories; "
            "truncate instead of filtering before trusting this run.",
            _dropped, _before_filter, 100 * _dropped / _before_filter)
    else:
        logger.info("Length filter: 0 of %d training samples over 4095 tokens.",
                    _before_filter)

    logger.info("Token length statistics:\n%s",
                json.dumps(_length_stats(tokenized_input), indent=2))

    # The training subset is drawn by create_json_train.py, which takes a seeded
    # random sample and writes it as <name>_1024.json. Pass that file directly.
    # This used to be done here with `select(range(0, n, round(n / 1024)))`, a
    # stride that collapsed to "the first 1024 rows in file order" whenever n was
    # under ~1536, and could return fewer than 1024 rows without saying so.
    # Reading the pre-drawn file also means this arm and the Flan-T5 arm can
    # train on exactly the same subset.
    if args.max_train_samples and args.max_train_samples < len(tokenized_input):
        logger.info("Screening on the first %d of %d samples.",
                    args.max_train_samples, len(tokenized_input))
        tokenized_input = tokenized_input.select(range(args.max_train_samples))

    logger.info("Training samples: %d", len(tokenized_input))
    if len(tokenized_input) > 1024:
        logger.info("Training on all %d samples. The reference protocol uses "
                    "1024 — pass the _1024.json file to --train_file_path for that.",
                    len(tokenized_input))

    # One decoded sample, so the log shows exactly what the model was trained on
    # and which part of it carried a gradient.
    _probe = DataCollatorForCompletionLM(tokenizer, mlm=False)([tokenized_input[0]])
    _labels = _probe["labels"][0]
    logger.info("Sample prompt tail : %r",
                tokenizer.decode(_probe["input_ids"][0][-60:]))
    logger.info("Sample supervised  : %r",
                tokenizer.decode(_labels[_labels != -100]))

    eval_input = None
    if _has_eval:
        eval_dataset = load_dataset('json', data_files=f'./data/processed_new/{args.dataset}/{args.eval_file_path}',
                                    split = 'train', download_mode='force_redownload')
        eval_input = eval_dataset.map(process_function, batched = True,
                                      remove_columns=eval_dataset.column_names)
        _eval_before = len(eval_input)
        eval_input = eval_input.filter(lambda x: len(x['input_ids']) < 4095)
        if _eval_before != len(eval_input):
            logger.warning("Evaluation: %d of %d samples dropped for length.",
                           _eval_before - len(eval_input), _eval_before)
        if args.max_eval_samples and args.max_eval_samples < len(eval_input):
            logger.info("Capping validation at %d of %d samples.",
                        args.max_eval_samples, len(eval_input))
            eval_input = eval_input.select(range(args.max_eval_samples))
        logger.info("Evaluation samples: %d (evaluated every %d optimiser steps)",
                    len(eval_input), _eval_steps)

    trainer = Trainer(model = training_model, # We pass in the PEFT version of the foundation model or the standard one if full finetuning is desired
                args = training_args, #The args for the training.
                train_dataset = tokenized_input, #The dataset used to to train the model.
                eval_dataset = eval_input,
                data_collator=DataCollatorForCompletionLM(tokenizer, mlm=False), # mlm=False indicates not to use masked language modeling
                callbacks=[controller] if _has_eval else None,
                )
    if _has_eval:
        controller.set_trainer(trainer)

    logger.info("=== Training starts ===")
    _t0 = time.perf_counter()
    trainer.train()
    logger.info("=== Training finished in %.1f min ===", (time.perf_counter() - _t0) / 60)

    # The full loss curve, so a remote run can be judged from its log alone
    # rather than from whatever scrolled past in the notebook.
    for entry in trainer.state.log_history:
        logger.info("  %s", json.dumps({k: v for k, v in entry.items()
                                        if k != "total_flos"}, default=str))
    if _has_eval and trainer.state.best_model_checkpoint:
        logger.info("Best checkpoint: %s (%s = %s)",
                    trainer.state.best_model_checkpoint,
                    training_args.metric_for_best_model,
                    trainer.state.best_metric)

    # Persist the trained LoRA adapter. Required whether or not an eval set was
    # given: without one nothing is checkpointed during training at all, and with
    # one the checkpoints are pruned by save_total_limit, so neither case leaves a
    # dependable final artifact behind.
    # trainer.save_model on a PEFT-wrapped model writes adapter_config.json and
    # adapter_model.safetensors — the layout PeftModelForCausalLM.from_pretrained
    # expects in run_hf.py. No tie_weights() call here: that is specific to T5's
    # shared encoder/decoder embeddings.
    # Same directory the checkpoints went to, so a run's adapter and its
    # checkpoints stay together.
    save_path = output_dir
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    trainer.state.save_to_json(os.path.join(save_path, "trainer_state.json"))
    logger.info("Model saved to %s", save_path)

    # After the save, never before: the summary is cosmetic, and on the Flan-T5
    # side a single unencodable character in it once raised between the end of
    # training and the model reaching disk, losing hours of GPU time.
    if _has_eval:
        logger.info("Controller summary:\n%s", controller.summary())
    logger.info("Log written to ./logs/train_llama_%s.log", trained_model_name)
