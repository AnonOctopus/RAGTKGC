"""Decoding helpers: turn one query prompt into a ranked candidate list."""

from utils import normalize_entity


def max_target_tokens(test_set, tokenizer, margin=8):
    """Token budget for generation, from the longest target in this run.

    Args:
        test_set: the loaded split; each row carries a 'target' string.
        tokenizer: the tokenizer the model will decode with.
        margin: slack added to the longest target. Beam search stops at EOS,
            so an over-generous budget is nearly free, while too small a one
            truncates candidates and can cut a long name down onto a shorter
            true one, scoring a hit the model never proposed.

    Returns:
        int: max_new_tokens to pass to generate.

    Raises:
        ValueError: the split is empty.
        KeyError: a row has no 'target' field.
    """
    if len(test_set) == 0:
        raise ValueError("Cannot size the generation budget from an empty split.")
    longest = max(len(tokenizer(row['target']).input_ids) for row in test_set)
    return longest + margin


def _clean(text):
    """Normalise one decoded candidate to a bare entity string.

    Args:
        text: the decoded continuation, special tokens already skipped.

    Returns:
        str: the candidate entity name, possibly empty.
    """
    # Only the first line: a decoder-only model given a history-completion
    # prompt predicts the object and then continues with the next fact line.
    # Targets never contain a newline, so this is inert elsewhere.
    return text.split('\n')[0].replace(']', '').replace('</s>', '').strip()


def _dedup(texts):
    """Candidates best first, empties and duplicates removed.

    Args:
        texts: decoded candidates for one query, best-scoring first.

    Returns:
        list[str]: the surviving candidates, order preserved.
    """
    # Deduplicate on the form scoring compares, not on the raw string: two
    # spellings that score as the same entity would otherwise both survive and
    # one would consume a rank above the target. The first — best-scoring —
    # spelling of each is the one kept.
    seen = set()
    candidates = []
    for text in texts:
        key = normalize_entity(text)
        if not key or key in seen:
            continue
        seen.add(key)
        candidates.append(text)
    return candidates


def beam_candidates(tokenizer, model, inputs, max_new_tokens, num_beams,
                    prompt_len=0, length_penalty=1.0):
    """Decode a ranked candidate list per prompt with beam search.

    Args:
        tokenizer: tokenizer matching the model.
        model: a generation-capable model in eval mode.
        inputs: tokenizer output for one or more prompts, already moved to the
            model's device. Batched input must be padded, and for a
            decoder-only model padded on the left, so that one prompt_len
            describes every row.
        max_new_tokens: cap on generated tokens.
        num_beams: beam width; also how many sequences are returned per prompt.
        prompt_len: tokens to drop from the front of every output sequence.
            0 for encoder-decoder models, whose output holds only the
            continuation; the padded input width for decoder-only ones, whose
            output repeats the prompt it was given.
        length_penalty: exponent the sequence log-probability is divided by the
            token count with, deciding how candidates of different length
            compare. 1.0 normalises fully; 0.0 not at all; above 1.0 rewards
            longer candidates.

    Returns:
        list[list[str]]: one candidate list per input prompt, in input order.
    """
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        length_penalty=length_penalty,
        early_stopping=True,
    )
    # generate returns num_beams rows per prompt, flattened and grouped by
    # prompt. Reading the batch size off the input rather than dividing by
    # num_beams keeps this honest if a future generation config returns a
    # different number of sequences.
    batch_size = inputs["input_ids"].shape[0]
    per_prompt = outputs.shape[0] // batch_size
    return [
        _dedup(
            _clean(tokenizer.decode(seq[prompt_len:], skip_special_tokens=True))
            for seq in outputs[i * per_prompt:(i + 1) * per_prompt]
        )
        for i in range(batch_size)
    ]


def predict_batch(tokenizer, model, prompts, args, max_new_tokens):
    """Rank candidate objects for several query prompts at once.

    Args:
        tokenizer: tokenizer matching the model.
        model: the fine-tuned model in eval mode.
        prompts: full input texts, already truncated by the caller.
        args: parsed arguments; base_model, num_beams, length_penalty and
            verbose are read.
        max_new_tokens: generation budget, from max_target_tokens.

    Returns:
        list[list[str]]: candidates best first per prompt, in prompt order.

    Raises:
        ValueError: args.base_model is not a supported local model.
    """
    # Only when the tokenizer has no pad token of its own. LLaMA has none and
    # needs eos standing in; T5 has <pad> and must keep it, or batched encoder
    # inputs get padded with </s> and the encoder reads an end-of-sequence
    # marker where there is only padding.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.base_model == 'google/flan-t5-small':
        # Encoder-decoder: the output holds the continuation alone, so nothing
        # is stripped, and the encoder reads whichever side is padded.
        prompt_len = 0
    elif args.base_model == 'TheBloke/Llama-2-7B-fp16':
        # Decoder-only: the output repeats the prompt, so it has to be sliced
        # off. Padding on the left makes the continuation start at the same
        # index in every row of the batch, which is what lets one prompt_len
        # describe them all; padding on the right would put a different number
        # of pad tokens before each continuation.
        tokenizer.padding_side = 'left'
    else:
        raise ValueError(f"predict_batch() does not support base_model {args.base_model!r}")

    inputs = tokenizer(list(prompts), return_tensors="pt",
                       padding=True).to(model.device)
    if args.base_model != 'google/flan-t5-small':
        prompt_len = inputs['input_ids'].shape[1]

    predictions = beam_candidates(
        tokenizer, model, inputs, max_new_tokens, args.num_beams, prompt_len,
        args.length_penalty,
    )

    if args.verbose:
        for row in predictions:
            for i, p in enumerate(row):
                print(f"  {i + 1}: {p}")

    return predictions


def predict(tokenizer, model, prompt, args, max_new_tokens):
    """Rank candidate objects for one query prompt.

    Args:
        tokenizer: tokenizer matching the model.
        model: the fine-tuned model in eval mode.
        prompt: the full input text, already truncated by the caller.
        args: parsed arguments; base_model, num_beams, length_penalty and
            verbose are read.
        max_new_tokens: generation budget, from max_target_tokens.

    Returns:
        list[str]: candidates best first, deduplicated.

    Raises:
        ValueError: args.base_model is not a supported local model.
    """
    return predict_batch(tokenizer, model, [prompt], args, max_new_tokens)[0]
