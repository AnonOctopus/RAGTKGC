# import the required classes

import argparse
import json
import os
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers import T5Tokenizer, T5ForConditionalGeneration
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

def process_function(examples):

  inputs = tokenizer(examples['context'], return_special_tokens_mask=True)
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

  return inputs


if __name__ == "__main__":

    args = parser()

    # Feel free to set your own values.

    model = 'google/flan-t5-small' # the model to be finetuned
    trained_model_name = args.trained_model_name # the name of the trained model
    output_dir = args.output_dir #The output directory where checkpoints will be written.

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
        per_device_train_batch_size=2,       # fixed — hardware limit on this laptop
        per_device_eval_batch_size=2,        # match train batch to keep VRAM usage predictable
        # --- optimiser ---
        learning_rate=3e-4,                  # higher than full fine-tuning; fine for adapter-scale updates
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
        # --- precision & reporting ---
        bf16=True,                           # fixed — bfloat16 mixed precision
        report_to="none",
    )
    
    # load the model and its tokenizer

    training_model = T5ForConditionalGeneration.from_pretrained(model, trust_remote_code = True,
                                                                device_map = 'auto',
                                                                )
    tokenizer = T5Tokenizer.from_pretrained(model)
    TOKEN_LIMIT = get_token_limit(tokenizer)
    TAIL_TRUNCATE_LONG_INPUTS = args.tail_truncate_long_inputs

    # set the model to train mode
    training_model.train()

    # load a dataset from the data files 
    # IMPORTANT: for ICEWS18, there are two training folders: 1024 and full. Use 1024 for finetuning LLaMA2-7B and full for Flan-T5-Small

    dataset = load_dataset('json', data_files=f'./data/processed_new/{args.dataset}/{args.train_file_path}', split = 'train')

    context_tokenized = tokenizer(dataset['context'], add_special_tokens=True)
    input_lengths = [len(x) for x in context_tokenized['input_ids']]
    length_stats = summarize_lengths(input_lengths, TOKEN_LIMIT)
    length_stats['tail_truncate_long_inputs'] = TAIL_TRUNCATE_LONG_INPUTS
    print('Input token length stats:')
    print(json.dumps(length_stats, indent=2))

    tokenized_input = dataset.map(process_function, batched = True, remove_columns=dataset.column_names)

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
        )
        tokenized_eval = eval_dataset.map(process_function, batched=True, remove_columns=eval_dataset.column_names)
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
    
    trainer.train()
    print(controller.summary())

    # Save the best model (already loaded into trainer.model by load_best_model_at_end)
    # under output_dir/trained_model_name.  tie_weights() is called first to ensure
    # encoder.embed_tokens / decoder.embed_tokens are re-tied before serialisation,
    # which prevents the 'missing keys' warning on every subsequent load.
    save_path = os.path.join(args.output_dir, trained_model_name.replace("'","").replace('"', ''))
    trainer.model.tie_weights()
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    trainer.state.save_to_json(os.path.join(save_path, "trainer_state.json"))
    print(f"Model saved to {save_path}")