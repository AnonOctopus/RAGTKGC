# import the required classes

import argparse
import glob
import json
import os

import numpy
import torch
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from datasets import load_dataset
from training_controller import TrainingControllerCallback


TOKEN_LIMIT = None
TAIL_TRUNCATE_LONG_INPUTS = False

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--trained_model_name",type = str)
    parser.add_argument("--output_dir", default = './models', type = str)
    parser.add_argument("--train_file_path", type = str)
    parser.add_argument(
      "--tail_truncate_long_inputs",
      action="store_true",
      help=(
        "If set, keep samples above max input length by truncating to the last tokens up to the limit "
        "instead of discarding them."
      ),
    )
    parser.add_argument(
      "--eval_file_path",
      type=str,
      default=None,
      help=(
        "Optional path (relative to ./data/processed_new/<dataset>/) to a JSON Lines file used as the "
        "evaluation set. When provided, validation loss is computed each epoch and fed to the "
        "TrainingControllerCallback to make CONTINUE / REDUCE_LR / STOP decisions."
      ),
    )
    parser.add_argument(
      "--resume_from_checkpoint",
      default=None,
      help=(
        "Continue an interrupted run instead of starting over. Pass a checkpoint "
        "directory, or 'auto' to pick the latest one inside this run's output "
        "directory. Optimiser state, learning-rate schedule and epoch counter are "
        "restored, so the result matches an uninterrupted run. Every other "
        "argument must match the original invocation, the training file above all: "
        "Trainer resumes at a step number, and a different dataset makes that step "
        "point somewhere else entirely."
      ),
    )
    parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help=(
        "Random seed for training. Controls dropout and the order in which the training set is "
        "shuffled. Defaults to 42, the same value Trainer uses when unset, so omitting it "
        "reproduces earlier runs. Vary it to measure run-to-run spread."
      ),
    )

    args = parser.parse_args()
    return args
    
# tokenize the input prompt and target entity (as labels)


def get_token_limit(tokenizer_obj):
    token_limit = getattr(tokenizer_obj, "max_len_single_sentence", None)
    if token_limit is None or token_limit <= 0 or token_limit > 1_000_000:
        token_limit = getattr(tokenizer_obj, "model_max_length", None)
    if token_limit is None or token_limit <= 0 or token_limit > 1_000_000:
        raise ValueError("Could not infer a valid tokenizer max input length.")
    return int(token_limit)


def summarize_lengths(lengths, token_limit):
    def stats(values):
        if not values:
            return {"count": 0, "avg": 0.0, "min": None, "max": None}
        return {
            "count": len(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    below_or_equal = [x for x in lengths if x <= token_limit]
    above = [x for x in lengths if x > token_limit]
    return {
        "token_limit": token_limit,
        "overall": stats(lengths),
        "below_or_equal_limit": stats(below_or_equal),
        "above_limit": stats(above),
    }

UNTRUNCATED_LENGTH_COLUMN = "untruncated_input_length"


def process_function(examples):

  inputs = tokenizer(examples['context'], return_special_tokens_mask=True)
  # Carry the pre-truncation length out as a column so length statistics can be
  # derived from this single tokenization pass. Measuring them afterwards would
  # under-report long inputs, because tail-truncation has already shortened them.
  untruncated_lengths = [len(ids) for ids in inputs['input_ids']]
  if TAIL_TRUNCATE_LONG_INPUTS:
    for i, ids in enumerate(inputs['input_ids']):
      if len(ids) > TOKEN_LIMIT:
        start = len(ids) - TOKEN_LIMIT
        inputs['input_ids'][i] = ids[start:]
        if 'attention_mask' in inputs:
          inputs['attention_mask'][i] = inputs['attention_mask'][i][start:]
        if 'special_tokens_mask' in inputs:
          inputs['special_tokens_mask'][i] = inputs['special_tokens_mask'][i][start:]
        if 'token_type_ids' in inputs:
          inputs['token_type_ids'][i] = inputs['token_type_ids'][i][start:]

  labels = tokenizer(examples['target'], return_special_tokens_mask=True)
  inputs['labels'] = labels['input_ids']
  inputs[UNTRUNCATED_LENGTH_COLUMN] = untruncated_lengths

  return inputs


if __name__ == "__main__":

    args = parser()

    # Fail before training rather than after: the save path is built from
    # trained_model_name, so omitting it would only surface at the very end.
    if not args.trained_model_name:
        raise SystemExit("--trained_model_name is required; it names the output model directory.")

    # Feel free to set your own values.

    model = 'google/flan-t5-small' # the model to be finetuned
    trained_model_name = args.trained_model_name # the name of the trained model
    # Per-run checkpoint directory. Trainer names checkpoints by optimiser step,
    # so runs that share a training-set size — the seed repeats of one
    # configuration — collide in a shared directory, and load_best_model_at_end
    # could then restore another run's checkpoint.
    output_dir = os.path.join(
        args.output_dir, trained_model_name.replace("'", "").replace('"', '')
    )

    # Adaptive training controller — monitors validation loss, gradient norms,
    # and compute efficiency to decide CONTINUE / REDUCE_LR / STOP each epoch.
    # Activated only when --eval_file_path is supplied (provides a validation set).
    controller = TrainingControllerCallback(
        min_delta=5e-4,          # tighter threshold — cosine keeps improving slowly
        patience=2,              # tolerate 2 non-improving epochs before acting
        lr_reduction_factor=0.5, # halve the LR on each plateau (backup if cosine isn't enough)
        max_lr_reductions=2,     # allow 2 manual reductions; mostly cosine does the work
        ema_alpha=0.4,           # reactive to recent epochs
        min_grad_norm=1e-2,
        grad_snr_threshold=1.0,
        min_marginal_improvement=1e-5,
    )

    _has_eval = bool(args.eval_file_path)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        # --- batch / accumulation ---
        # 8, not 2: measured on this laptop's 6 GiB card, a worst-case batch of
        # 8 (every sample at the token limit) peaks at 3.25 GiB, and throughput
        # saturates here — 16 fits but is no faster and leaves no headroom.
        # Note that raising this changes the effective batch, since no gradient
        # accumulation is configured.
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,        # match train batch to keep VRAM usage predictable
        # Batches drawn at random pad to their longest member, and inputs range
        # from a few tokens to the limit, so over half of every batch was
        # padding. Grouping by length removes almost all of it.
        group_by_length=True,
        # --- optimiser ---
        # This is full fine-tuning — every parameter of the model is updated and
        # there is no adapter. 3e-4 is above the Trainer's AdamW default, which
        # suits a model of this size; the cosine schedule below anneals it away.
        learning_rate=3e-4,
        weight_decay=0.1,                    # L2 regularisation to reduce overfitting
        num_train_epochs=3,                  # same as reference; best-checkpoint selection via eval set recovers quality
        # --- schedule ---
        lr_scheduler_type="cosine",          # full anneal from 3e-4 to ~0; proven recipe for this dataset
        warmup_ratio=0.06,                   # 6% of total steps; scales automatically with dataset size
        # --- evaluation & checkpointing ---
        eval_strategy="epoch" if _has_eval else "no",
        save_strategy="epoch",               # must match eval_strategy for load_best_model_at_end
        save_total_limit=3,                  # keep 3 checkpoints: 2 recent + best (prevents best being evicted)
        load_best_model_at_end=_has_eval,    # restore the best checkpoint after training (requires eval set)
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # --- logging ---
        logging_strategy="steps",
        logging_steps=500,                   # log every 500 steps for fine-grained visibility
        # --- reproducibility ---
        # data_seed is left to follow seed. This does not make a run bit-exact:
        # bf16 reduction order and nondeterministic GPU kernels still vary, so
        # two runs at the same seed can differ slightly.
        seed=args.seed,
        # --- precision & reporting ---
        bf16=True,                           # fixed — bfloat16 mixed precision
        report_to="none",
    )
    print(f"Training seed: {args.seed}")
    
    # load the model and its tokenizer

    training_model = T5ForConditionalGeneration.from_pretrained(model, trust_remote_code = True,
                                                                device_map = 'auto',
                                                                )
    # Fast (Rust) tokenizer. Verified to emit identical input_ids,
    # attention_mask, special_tokens_mask and labels as the Python T5Tokenizer
    # on this corpus, at roughly an order of magnitude less wall-clock.
    tokenizer = T5TokenizerFast.from_pretrained(model)
    TOKEN_LIMIT = get_token_limit(tokenizer)
    TAIL_TRUNCATE_LONG_INPUTS = args.tail_truncate_long_inputs

    # set the model to train mode
    training_model.train()

    # load a dataset from the data files 
    # IMPORTANT: for ICEWS18, there are two training folders: 1024 and full. Use 1024 for finetuning LLaMA2-7B and full for Flan-T5-Small

    # force_redownload re-reads the JSON instead of trusting the loader's cache.
    # The pipeline regenerates these files in place, and older `datasets`
    # versions than the one developed against have served the previous contents
    # for an unchanged path. Costs one re-parse per run, which is negligible
    # beside training, and removes a silent wrong-data failure mode.
    dataset = load_dataset('json', data_files=f'./data/processed_new/{args.dataset}/{args.train_file_path}', split = 'train',
                           download_mode='force_redownload')

    # process_function reads `tokenizer`, `TOKEN_LIMIT` and
    # TAIL_TRUNCATE_LONG_INPUTS from module scope, all assigned above. They are
    # part of the map's cache fingerprint, so changing one does invalidate the
    # cache, but it also means the function cannot be imported or called
    # standalone before those names exist.
    tokenized_input = dataset.map(process_function, batched = True, remove_columns=dataset.column_names)

    # Length statistics come from the column recorded during the map above.
    # Tokenizing the corpus a second time here would double the cost and hold
    # every context's ids in memory at once.
    input_lengths = tokenized_input[UNTRUNCATED_LENGTH_COLUMN]
    length_stats = summarize_lengths(input_lengths, TOKEN_LIMIT)
    length_stats['tail_truncate_long_inputs'] = TAIL_TRUNCATE_LONG_INPUTS
    print('Input token length stats:')
    print(json.dumps(length_stats, indent=2))

    # The collator pads every remaining column, so the bookkeeping one must go.
    tokenized_input = tokenized_input.remove_columns([UNTRUNCATED_LENGTH_COLUMN])

    # filter out any input prompt that is longer than the models context size

    if not TAIL_TRUNCATE_LONG_INPUTS:
      tokenized_input = tokenized_input.filter(lambda x: len(x['input_ids']) <= TOKEN_LIMIT)

    # Optionally load and tokenize the evaluation set for the controller
    tokenized_eval = None
    if args.eval_file_path:
        eval_dataset = load_dataset(
            'json',
            data_files=f'./data/processed_new/{args.dataset}/{args.eval_file_path}',
            split='train',
            download_mode='force_redownload',
        )
        tokenized_eval = eval_dataset.map(process_function, batched=True, remove_columns=eval_dataset.column_names)
        tokenized_eval = tokenized_eval.remove_columns([UNTRUNCATED_LENGTH_COLUMN])
        if not TAIL_TRUNCATE_LONG_INPUTS:
            tokenized_eval = tokenized_eval.filter(lambda x: len(x['input_ids']) <= TOKEN_LIMIT)

    trainer = Seq2SeqTrainer(model = training_model,
                args = training_args,
                train_dataset = tokenized_input,
                eval_dataset = tokenized_eval,
                data_collator=DataCollatorForSeq2Seq(tokenizer, model = training_model),
                callbacks=[controller],
                )
    # Give the controller a reference to the trainer so it can access the
    # optimizer directly — Trainer doesn't pass it through on_evaluate kwargs.
    controller.set_trainer(trainer)
    
    # Resolve 'auto' to the highest-numbered checkpoint in this run's directory.
    # Trainer accepts a bool for the same purpose, but its True means "search and
    # raise if nothing is there", which turns a fresh run into a failure; being
    # explicit lets an absent checkpoint simply start from scratch.
    resume = args.resume_from_checkpoint
    if resume == "auto":
        checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        resume = max(checkpoints, key=lambda p: int(p.rsplit("-", 1)[1])) if checkpoints else None
        print(f"Resuming from {resume}" if resume
              else f"No checkpoint in {output_dir}; starting from scratch.")
    elif resume:
        if not os.path.isdir(resume):
            raise SystemExit(f"--resume_from_checkpoint: no such directory {resume!r}")
        print(f"Resuming from {resume}")

    if resume:
        # Trainer restores rng_state.pth with weights_only=True, which refuses any
        # class outside torch's allowlist. The file holds numpy's RandomState, so
        # resuming fails on the four classes its pickle names. They are data
        # containers, and the checkpoint was written by this script, so
        # allowlisting them adds no exposure that loading the checkpoint at all
        # does not already carry.
        torch.serialization.add_safe_globals([
            numpy.core.multiarray._reconstruct,
            numpy.ndarray,
            numpy.dtype,
            type(numpy.dtype(numpy.uint32)),   # numpy.dtype[uint32]
        ])

    trainer.train(resume_from_checkpoint=resume)

    # Save before anything else runs. This used to sit after the controller
    # summary was printed, and a single unencodable character in that summary
    # raised, losing a finished model that had cost hours of GPU time. Nothing
    # cosmetic belongs between the end of training and the model reaching disk.
    #
    # The best model is already loaded into trainer.model by
    # load_best_model_at_end. tie_weights() first, so encoder.embed_tokens and
    # decoder.embed_tokens are re-tied before serialisation and every later
    # load is free of the 'missing keys' warning. Same directory as the
    # checkpoints, so a run's model and its checkpoints stay together.
    save_path = output_dir
    trainer.model.tie_weights()
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    trainer.state.save_to_json(os.path.join(save_path, "trainer_state.json"))
    print(f"Model saved to {save_path}")

    print(controller.summary())