"""Check a machine can run training_LLaMA.py before any GPU time is spent.

Exercises everything except loading the 7B model: the installed library
versions, the arguments TrainingArguments accepts, the collator API, the
tokenizer's padding side, and — the part worth the most — that the span left in
the loss is exactly the answer, on this machine's tokenizer and data.

A remote session rarely runs the pinned versions, and the failures that follow
are not always loud: a renamed argument raises, but a changed padding default
silently trains the model on its own prompt. Running this first turns both into
a few seconds at the terminal.

Usage, from the repository root:

    python preflight_llama.py --dataset icews14 \\
        --train_file_path gtkg/train_inv_n50_idn/json/icews14_gtkg_inv_n50_idn_train_1024.json
"""

import argparse
import inspect
import json
import sys

FAILURES = []


def check(name, ok, detail=""):
    """Record one result and print it.

    Args:
        name: what was checked.
        ok: whether it passed.
        detail: optional context shown after the verdict.

    Returns:
        bool: the value of ok, so callers can branch on it.
    """
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


def parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="icews14")
    p.add_argument("--train_file_path", required=True,
                   help="Path relative to ./data/processed_new/<dataset>/.")
    p.add_argument("--prompt_prefix", action="store_true",
                   help="Check the instruction-prefix path instead of the bare one.")
    p.add_argument("--samples", type=int, default=64,
                   help="How many rows to verify the supervised span on.")
    return p.parse_args()


if __name__ == "__main__":
    args = parser()

    print("\n1. environment")
    import torch
    import transformers
    import peft
    from transformers import TrainingArguments, AutoTokenizer
    print(f"     transformers {transformers.__version__}, peft {peft.__version__}, "
          f"torch {torch.__version__}")
    check("CUDA available", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
    if torch.cuda.is_available():
        print(f"     GPU memory {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")

    print("\n2. TrainingArguments accepts what the script passes")
    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    for name in ("warmup_steps", "logging_steps", "remove_unused_columns",
                 "load_best_model_at_end", "metric_for_best_model",
                 "save_total_limit", "gradient_accumulation_steps"):
        check(f"accepts {name}", name in accepted)
    check("accepts an eval strategy",
          "eval_strategy" in accepted or "evaluation_strategy" in accepted,
          "eval_strategy" if "eval_strategy" in accepted else "evaluation_strategy")

    print("\n3. the script imports and its LoRA config builds")
    sys.argv = ["training_LLaMA.py", "--dataset", args.dataset,
                "--trained_model_name", "preflight"]
    import training_LLaMA
    from training_LLaMA import (process_function, DataCollatorForCompletionLM,
                                _build_training_args, _length_stats)
    from peft import LoraConfig
    parsed = training_LLaMA.parser()
    check("lora_alpha follows 2*r", parsed.lora_alpha == 2 * parsed.lora_r,
          f"r={parsed.lora_r} alpha={parsed.lora_alpha}")
    modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    config = LoraConfig(r=parsed.lora_r, lora_alpha=parsed.lora_alpha,
                        lora_dropout=parsed.lora_dropout, target_modules=modules,
                        bias="none", task_type="CAUSAL_LM")
    check("LoraConfig builds with all linear layers", config is not None)

    # Actually inject adapters, on a stand-in with the same layer names. PEFT
    # picks an implementation per layer by walking a chain of dispatchers, and a
    # broken optional dependency anywhere in that chain raises rather than being
    # skipped — which only happens once the layers are real. Doing it on four
    # tiny Linears costs nothing and fails here instead of after a 13 GB
    # download. bfloat16 because the dispatch taken depends on the layer type,
    # and a quantised model reaches a different branch than an unquantised one.
    from peft import get_peft_model

    class _Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(self, name, torch.nn.Linear(8, 8, bias=False))

        def forward(self, x):
            return self.o_proj(self.v_proj(self.k_proj(self.q_proj(x))))

        def prepare_inputs_for_generation(self, *a, **kw):
            # PeftModelForCausalLM binds this at wrap time. Never called here.
            raise NotImplementedError

    try:
        stub = get_peft_model(_Stub().to(torch.bfloat16), config)
        adapters = sum(p.numel() for p in stub.parameters() if p.requires_grad)
        check("adapters inject into bf16 Linear layers", adapters > 0,
              f"{adapters} trainable adapter parameters")
    except (ImportError, RuntimeError, TypeError, ValueError, AttributeError) as exc:
        check("adapters inject into bf16 Linear layers", False,
              f"{type(exc).__name__}: {exc}")

    print("\n4. tokenizer")
    tok = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-fp16")
    tok.pad_token = tok.eos_token
    print(f"     padding_side defaults to {tok.padding_side!r}")
    tok.padding_side = "right"
    check("padding_side can be set to right", tok.padding_side == "right")
    # Informational, not a verdict. On transformers 4 a literal "</s>" glued to a
    # word tokenizes as '</', 's', '>' rather than EOS; on 5 it is parsed. The
    # script depends on neither, appending the id instead, so this only records
    # which behaviour this machine has.
    glued = tok("France</s>", add_special_tokens=False)["input_ids"]
    print(f"     a literal </s> glued to a word "
          f"{'IS' if tok.eos_token_id in glued else 'is NOT'} parsed as EOS here")
    training_LLaMA.tokenizer = tok
    training_LLaMA.PROMPT_PREFIX = args.prompt_prefix

    print("\n5. collator API")
    collator = DataCollatorForCompletionLM(tok, mlm=False)
    check("collator constructs", collator is not None)
    check("torch_call is still the entry point", hasattr(collator, "torch_call"))

    print("\n6. data and the supervised span"
          f"  (prefix {'on' if args.prompt_prefix else 'off'})")
    from datasets import load_dataset
    path = f"./data/processed_new/{args.dataset}/{args.train_file_path}"
    raw = load_dataset("json", data_files=path, split="train")
    check("training file loads", len(raw) > 0, f"{len(raw)} rows")
    mapped = raw.map(process_function, batched=True, remove_columns=raw.column_names)
    check("target_start survives map", "target_start" in mapped.column_names)
    mapped = mapped.filter(lambda x: len(x["input_ids"]) < 4095)
    check("nothing dropped for length", len(mapped) == len(raw),
          f"{len(raw) - len(mapped)} over 4095 tokens")

    # The invariant that matters, whatever the tokenizer does with a literal
    # "</s>": every sequence ends with exactly one EOS, and carries no stray one.
    probe = [mapped[i] for i in range(min(32, len(mapped)))]
    ends_with_eos = sum(row["input_ids"][-1] == tok.eos_token_id for row in probe)
    single_eos = sum(row["input_ids"].count(tok.eos_token_id) == 1 for row in probe)
    check("every sequence ends with EOS", ends_with_eos == len(probe),
          f"{ends_with_eos}/{len(probe)}")
    check("exactly one EOS per sequence", single_eos == len(probe),
          f"{single_eos}/{len(probe)}")

    n = min(args.samples, len(mapped))
    wrong = []
    for i in range(0, n, 8):
        rows = list(range(i, min(i + 8, n)))
        batch = collator([mapped[j] for j in rows])
        if "target_start" in batch:
            check("target_start stripped before the model", False)
            break
        for k, j in enumerate(rows):
            labels = batch["labels"][k]
            kept = tok.decode(labels[labels != -100]).replace("</s>", "").strip()
            if kept != raw[j]["target"]:
                wrong.append((raw[j]["target"], kept))
    else:
        check("target_start stripped before the model", True)
    check(f"supervised span equals the target ({n} rows, batch 8)", not wrong,
          f"{len(wrong)} wrong" if wrong else "")
    for target, kept in wrong[:3]:
        print(f"       target={target!r}  supervised={kept!r}")

    print("\n6b. batched generation setup (no model needed)")
    # The three things batching changes for a decoder-only model, each of which
    # fails silently rather than raising: the padding side, the slice that
    # removes the echoed prompt, and the split of generate's flattened output
    # back into one list per prompt. A wrong answer here still looks like a
    # plausible entity list, so it is checked against the tokenizer directly.
    import torch as _torch
    from model_utils import beam_candidates
    tok.padding_side = "left"
    two = tok(["short prompt", "a considerably longer prompt than the other one"],
              return_tensors="pt", padding=True)
    widths = [int(m.sum()) for m in two["attention_mask"]]
    check("left padding puts the pad tokens first",
          int(two["attention_mask"][0][0]) == 0 and int(two["attention_mask"][0][-1]) == 1,
          f"real tokens per row: {widths}")
    check("both rows share one prompt width",
          two["input_ids"].shape[1] == max(widths),
          f"padded width {two['input_ids'].shape[1]}")

    # Each beam continues with a different, known word. That makes the check
    # able to fail: a wrong slice point leaks prompt text into the candidate, a
    # wrong reshape hands row 0 row 1's words, and an off-by-one in either
    # changes which words come back. Asserting on empty output would not.
    WORDS = ["Thailand", "Malaysia", "Vietnam"]

    class _Gen:
        """Stands in for generate: echoes each prompt, then one known word."""
        def __init__(self, beams, word_ids):
            self.beams, self.word_ids = beams, word_ids

        def generate(self, **kw):
            ids = kw["input_ids"]
            repeated = ids.repeat_interleave(self.beams, dim=0)
            tails = [self.word_ids[i % self.beams] for i in range(repeated.shape[0])]
            width = max(len(t) for t in tails)
            padded = [t + [tok.eos_token_id] * (width - len(t)) for t in tails]
            return _torch.cat([repeated, _torch.tensor(padded)], dim=1)

    beams = len(WORDS)
    word_ids = [tok(w, add_special_tokens=False)["input_ids"] for w in WORDS]
    rows = beam_candidates(tok, _Gen(beams, word_ids), two, 8, beams,
                           prompt_len=two["input_ids"].shape[1])
    check("one candidate list per prompt", len(rows) == 2, f"{len(rows)} lists")
    check("each prompt gets its own beams back",
          all(r == WORDS for r in rows), f"{rows}")

    print("\n7. training arguments build on this version")
    ta = _build_training_args({
        "output_dir": "./_preflight", "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1, "gradient_accumulation_steps": 8,
        "learning_rate": 3e-4, "num_train_epochs": 1, "warmup_steps": 8,
        "logging_steps": 4, "weight_decay": 0.0, "remove_unused_columns": False,
        "eval_strategy": "steps", "eval_steps": 4, "save_strategy": "steps",
        "save_steps": 4, "save_total_limit": 3, "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss", "greater_is_better": False,
        "seed": 1, "report_to": "none",
    }, __import__("logging").getLogger("preflight"))
    check("TrainingArguments built", ta is not None)

    print("\n8. objective shape")
    print("     " + json.dumps(_length_stats(mapped.select(range(n))), indent=5))

    # ASCII only. This prints to a console whose encoding is not ours to choose;
    # on Windows it is cp1252, where an em-dash raises UnicodeEncodeError.
    print("\n" + ("PREFLIGHT PASSED - safe to spend GPU time."
                  if not FAILURES else f"PREFLIGHT FAILED: {FAILURES}"))
    sys.exit(1 if FAILURES else 0)
