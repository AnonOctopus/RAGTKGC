# RAGTKGC — codebase investigation notes

Working notes for the paper update. Records what the code actually does, how it
diverges from the GenTKG baseline it is built on, and what must be decided or
fixed before the full re-run.

Scope of the update: ICEWS14 + ICEWS18, Flan-T5-Small (+ LLaMA2-7B if resources
allow), one best variant per model, GPT-4.1 RAG applied only to the single best
model.

---

## 0. Status

**Phase 0 and Phase 1 are complete.** Sections 3 and 9 below were written during
the audit and describe defects in the past tense of that audit; this section is
authoritative where they disagree.

| Phase | State | Items |
|---|---|---|
| 0 — blockers | **done** | F5, F57, F71, F75 |
| 1 — output-neutral hygiene | **done** | 28 items, listed in §10 |
| 2 — make it fast | **F43, F33a, F55b/c/d done**, 1 left | batch `run_hf.py` |
| 3 — re-mine | **DONE** | F48, F86, F37a, F37c, F38b, F33b, rename, naming sync, F37b rebuild — six banks mined 2026-09-09 |
| 4 — protocol change | **Groups A, B and C done**, D pending; **E dropped** | filtered H@k + `cs` removed; F67 beam search; F53/F47 GenTKG answer format; then LLaMA parity |
| 5 — experiment grid | pending | the runs |

`.369` on the reference file still reproduces after every Phase 1 edit, which is
the standing regression check.

### Findings withdrawn or reframed by measurement

Four plan items did not survive being measured. They are **not** to be applied
as originally written.

| F | Filed as | Actually |
|---|---|---|
| **F62** | `datasets.map` cache goes stale on a changed global | **Refuted.** The fingerprint invalidates correctly, the tokenizer hashes stably, and a rewritten JSON is re-read — verified on `datasets` 3.2.0 *and* 3.5.0. The Colab symptom remains unexplained; capture `datasets.__version__` there if it recurs. `download_mode='force_redownload'` was added to all six `load_dataset` calls as version-insurance only. |
| **F68** | Variant missing from `get_filename`, so runs overwrite each other | **Wrong premise** — the variant is already in the JSON basename *and* the model name, twice. The real hazard is length: the longest results path is 240 of the 260 characters Windows allows (`LongPathsEnabled=0`). Fixed by auto-numbering (`_2`, +2 chars), a run manifest at `results/{ds}/_runs.jsonl`, and a warning above 240. |
| **F69** | Delete the pre-scan; the runtime block reports the same thing | **Redundant half was the other one.** The pre-scan is cheap and gives length stats before GPU time is spent; `overall_len_stats` inside the loop was the duplicate. Removed that instead, and added `effective_after_truncation` — the length actually fed to the model, which neither statistic previously reported. |
| **F72** | Give each model an explicit token limit instead of inferring | **Rejected.** The tokenizer is authoritative, hardcoding breaks silently on any model change, and the `ValueError` is correct behaviour (it refuses to guess) rather than a defect. Replaced by returning *which* attribute the limit came from, and logging it. |

### New findings from the Phase 1 pass

| ID | Finding |
|---|---|
| **F81** | The legacy `test` vs `test_num_facts` directories are **not** "uncapped vs capped". Both are capped at 50 facts; they differ in *which end* survives, because `early_stop_at_num_facts` stops consulting rules once 50 facts exist rather than collecting all and taking the newest 50. `test` keeps the oldest (ts 91→326), `test_num_facts` the newest (254→331). Legacy data only — every set will be regenerated. Current `out_suffix` does record `_early_stop`, so the trap is closed. |
| **F82** | T5 has **no absolute position embeddings**; `relative_attention_max_distance = 128`. Past 128 tokens of distance every relative position collapses into one bucket, and effective inputs average 456 tokens. So the model has no positional resolution over most of a history and can only perceive recency by reading the literal timestamp strings. This is a stronger justification for the `num_facts` cap than "it fits the context", and it is paper-relevant. |
| **F83** | A results file was silently incomplete: `llama-2-7B-icews14-gtkg_icews14_gpt_given_relations.jsonl` has 7,362 of 7,371 rows and was being scored as complete. F79 now skips it. At least one LLaMA run terminated early with no trace. |
| **F84** | **CLOSED by F67.** H@1 == H@3 on all 50 ICEWS14 results files, because `permute` yielded one distinct candidate. Beam search at k=10 separates them. Every one of those 50 files is now obsolete and must be re-run before H@3 or H@10 is reported. |
| **F86** | **FIXED 2026-09-08. `transition_distr="exp"` never produced an exponential distribution over time.** `sample_next_edge` computes `np.exp(tss - cur_ts)` with timestamps in steps of 24, so the decay constant is 24× too steep: a one-day gap already costs a factor `exp(-24) = 3.8e-11`, and any gap of 32 days or more underflows to exactly 0. Measured on `gtkg` / ICEWS14 / 60 relations × 200 walks (7,235 steps with more than one candidate): **87.4% sample uniformly among the candidates at the single newest admissible timestamp** — verified that 100% of the probability mass sits on the newest timestamp in every case that is not already a hard argmax — and **12.6% underflow entirely**, hit the `except ValueError` fallback at `temporal_walk.py:82` and sample **uniformly over all candidates, ignoring recency completely**. So the walk is "newest-only", degrading to "time-blind" an eighth of the time; it is never graded. `np.exp(tss - cur_ts)` is only meaningful when consecutive timestamps differ by ~1, so this is an artifact of this repo's ts2id encoding (day × 24) rather than TLogic's design — dividing the exponent by the period would restore the intended decay. **Affects `gtkg`, which Phase 3 re-mines.** Provenance established from public code: `temporal_walk.py` is TLogic's file verbatim (only `sample_start_edge2` added), TLogic's own `ts2id.json` is day-indexed (`2014-01-02: 1`, max 364), and GenTKG — from which this repo's code and data derive — divides by 24 in its own `TLR.py`, proving its timestamps are hour-scaled too. So the algorithm assumes day indices while the data supplies hours; the defect is inherited, not introduced here, and GenTKG's baseline carries it as well. Fixed by normalising the gap by the smallest timestamp spacing present in `learn_data`, derived rather than hardcoded so it is **provably a no-op on day-indexed input** (verified: granularity 24 here, 1 on the same data re-indexed by day, 1 for a single-timestamp input). After the fix: underflow fallback **12.6% → 0.0%**, genuinely graded steps **0.0% → 65.3%**. `unif` is unaffected. Consequence for the paper: `gtkg` numbers will not match GenTKG's published TLR numbers, and the reason is a corrected transition. |
| **F88** | **CLOSED.** `run_hf.py` could not evaluate any T5 model trained before F61: it loads the tokenizer from the fine-tuned directory, and pre-F61 models saved only `spiece.model`, so `T5TokenizerFast` tried to convert the slow SentencePiece file and died with `TypeError: Descriptors cannot be created directly` from a protobuf clash. Now falls back to the base model's tokenizer, which is identical after fine-tuning, and logs the substitution. |
| **F89** | **10 of 7,371 rows in a legacy results file carry raw timestamps — MINOR, unexplained.** In `llama-2-7B-icews14-ragtkgc_icews14_ragtkgc_test.jsonl`, rows at scattered positions record `timestamp` as `8136`, `8208`, … where every other row records a day such as `334.0`. Every formatter in the current code divides by the period, so this cannot be reproduced from the present source; it is presumably an artifact of whatever produced that file. It costs 0.14% of queries their filtered rank and every results file is being regenerated, so it was not chased further. |
| **F85** | **RESOLVED 2026-09-08: `--confidence_threshold` selects nothing; in `t5` mode it only reorders.** Full ICEWS14 test split, `--num_facts 50` with and without `--confidence_threshold 0.5`: the two files hold the **same multiset of history lines** (0 lines unique to either), and 1,772 of 7,371 samples (24.0%) differ — **all 1,772 by reordering alone, 0 by a different fact set**. This is what `model_type="t5"` does by construction: `to_lines(below) + to_lines(above)` partitions and concatenates, never drops. The earlier "byte-identical" reading was wrong (24% of samples do differ), but the substantive point is stronger than suspected. **Paper-relevant:** the threshold must not be described as filtering low-confidence evidence in the T5 experiments — its only effect is placing low-confidence facts first and high-confidence facts adjacent to the query, which interacts with F82. The `llm` branch additionally inserts two label lines, so there it does change content. |

### Verified numbers added

| What | Value |
|---|---|
| BERTScore, reference file (F20, was unverified) | **raw 0.9007 → rescaled 0.4116** |
| Truncated inputs still above `token_limit` | 3,395 of 6,087 (55.8%), all by exactly 1 token |
| Effective input length after truncation | avg 456.2, max 512 (model receives +1 EOS = 513) |
| `T5TokenizerFast` vs `T5Tokenizer` | 0 id mismatches / 5.1 M tokens; **9.3×** faster in `gtkgt` |

### Dependencies between remaining items

- ~~**F80's top-k extension blocks on F67.**~~ Unblocked: beam search produces a
  real candidate list. BERTScore still scores top-1 only, by decision, not
  because the candidates are artefacts.
- ~~**F55b and F55d change the metadata format together.**~~ Resolved: applied
  together on 2026-09-08. F56's remaining exposure is the filter *order*; the
  rendering half is now one shared function.

---

## 1. How to run things

| Fact | Value |
|---|---|
| **Project python** | `C:\Users\Ionut\anaconda3\envs\gtkgt\python.exe` — python 3.9.21, transformers 4.51.3, datasets 3.5.0, torch 2.5.1+cu121, plus `evaluate`, `bert_score`, `peft`, `ijson`. **Use this for every run and every verification.** |
| Base env | `C:\Users\Ionut\anaconda3\python.exe` — transformers 4.48.1, datasets 3.2.0, has pypdf / PyMuPDF but **no `evaluate`, `bert_score` or `peft`**. Also needs `USE_TF=0` to import transformers at all (Keras 3 vs tf-keras). Use only for PDF extraction. |
| `learn.py` | must run from `data_utils/rules_learning` (flat imports, `../../data/...`) |
| `retrieve.py`, `apply_history_filters.py`, `naive_history_metadata.py` | must run from `data_utils` |
| `training_T5.py`, `run_hf.py`, `compute_metrics_from_results.py`, `bertscore.py` | run from repo root |
| `paper.pdf` | not encrypted; the Read tool misreports it. Extract with PyMuPDF instead |

Two data roots, same content, different encoding:

- `data/original/{ds}/` — `train/valid/test.txt` are **integer-encoded** (5 cols);
  `all_facts.txt` is **string-encoded** (4 cols)
- `data/processed_new/{ds}/` — string-encoded

Nothing enforces that they stay in sync.

---

## 2. Verified reference numbers (ICEWS14)

Rule banks in `data/processed_new/icews14/output/icews14/`, identified by the
`found_by` field stamped into each rule:

| File | Algorithm | Relations with rules | Rules | Forward body | Inverse body |
|---|---|---|---|---|---|
| `080525134642_r[1]_n200_exp_s1` | gtkg | 413 | 8027 | 4007 (49.9%) | 4020 (50.1%) |
| `080525131706_r[1]_n200_exp_s1` | ragtkgc | 421 | 33972 | 16999 | 16973 |
| `170426145844_r[1]_n200_exp_s1` | ragtkgc_no_mining | 434 | 41068 | 20534 | 20534 |
| `060426144613_r[1]_n200_unif_s1` | unstamped, `unif` | 424 | 11707 | — | — |
| `060426150204_r[1]_n200_unif_s1` | unstamped, `unif` | 425 | 34348 | — | — |

ICEWS18: `050525143021` = gtkg (465 rels / 12553 rules), `050525174831` =
ragtkgc (475 / 68709). Both unstamped, matched by count against the paper.
**No `ragtkgc_no_mining` bank exists for ICEWS18 yet.**

Split sizes: train 74845, valid 8514, test 7371, `all_facts` 90730 (= sum).
`relation2id`: 230 relations (ICEWS14), 256 (ICEWS18) — matches the hardcoded
`num_relations` in `retrieve.py`. `rel_keys[i]` correctly names relation `i`
(0 positional mismatches).

Timestamp domains — **both roots are already hour-scaled** (step 24):

| Source | step | max (ICEWS14) |
|---|---|---|
| `data/original/train.txt` col 3 | 24 | 7272 (day 303) |
| `processed_new/ts2id.json` | 24 | 8736 (day 364) |

`all_facts.txt` is chronologically sorted (0 out-of-order adjacent pairs in
90729) — the recency-by-index assumption in retrieval holds. **Not yet verified
for ICEWS18.**

### Old paper's published numbers, for comparison

Table 3 (mining): gtkg ICEWS14 16.593 s / 413 / 7996; ragtkgc ICEWS14
1013.062 s / 421 / 33972; gtkg ICEWS18 30.343 s / 465 / 12553; ragtkgc ICEWS18
6327.188 s / 475 / 68709.

The gtkg rule count differs from the file (7996 vs 8027) because gtkg was
averaged over several seeds while ragtkgc was a single run. Counting convention
also varies: `learn.py` prints `// 2`, `rules_statistics` differs again. **Pick
one convention and state it.**

Table 4 (H@1 / H@3 / CS / BERTScore), ICEWS14 | ICEWS18:

| Model | ICEWS14 | ICEWS18 |
|---|---|---|
| Llama2-7B-raw | .171 / .277 / .611 / .905 | .083 / .122 / .631 / .885 |
| Llama2-7B-standard | .313 / .436 / .692 / .913 | .121 / .196 / .641 / .890 |
| Llama2-7B-gtkg | .317 / .383 / .685 / .909 | .172 / .242 / .625 / .895 |
| Llama2-7B-ragtkgc | .338 / .468 / .698 / .919 | .181 / .338 / .649 / .896 |
| Flan-T5-Small-raw | .287 / .287 / .655 / .915 | .197 / .197 / .630 / .907 |
| Flan-T5-Small-standard | .335 / .335 / .692 / .922 | .201 / .201 / .634 / .906 |
| Flan-T5-Small-gtkg | .350 / .350 / .699 / .923 | .227 / .227 / .649 / .910 |
| Flan-T5-Small-ragtkgc | .354 / .354 / .699 / .923 | .226 / .226 / .650 / .910 |

H@1 == H@3 for Flan-T5-Small in the published table too — it emits one
prediction, so H@3/H@10 carry no information for that model.

### Current results recomputed from `results/icews14/`

Via `compute_metrics_from_results.py`. Only Flan-T5-Small was re-run this round;
LLaMA and all GPT-RAG files are from the previous paper.

| Miner | Retrieval config | H@1 | CS |
|---|---|---|---|
| raw | — | .287 | — |
| standard | — | .335 | — |
| gtkg | `test` | .350 | .699 |
| ragtkgc | `test` | .354 | .699 |
| ragtkgc_no_mining | `test`, no truncate | .338 | .689 |
| ragtkgc_no_mining | `test`, tail_truncate | .344 | .692 |
| **gtkg** | **`inverse_included_num_facts`, fancy_half** | **.369** | — |
| ragtkgc | `inverse_included_num_facts`, fancy_half, tail_truncate | .365 | .703 |
| ragtkgc_no_mining | `inverse_included_num_facts`, fancy_half | .363 | — |
| ragtkgc_no_mining | `..._50_top10rules`, fancy_half, tail_truncate | .363 | .700 |

Caveats: the `.369` / `.363` files carry no `tail_truncate` suffix, and that
suffix only exists after commit `19ac0ad`, so the test-time flag for those runs
cannot be recovered from artifacts. `write_results` does not store the prompt.
Also `ragtkgc_no_mining/test_inverse_included_num_facts/` no longer exists on
disk, so the `.363` matched run is not currently reproducible.

Token lengths (train split, 4000-row samples, seed 0, limit 511):

| Variant | avg tokens | max | discarded without `--tail_truncate_long_inputs` |
|---|---|---|---|
| `gtkg/train_inverse_included_num_facts` | 988.6 | 2952 | 67.2% |
| `ragtkgc/train_inverse_included_num_facts` | 1060.4 | 2952 | 70.7% |
| `ragtkgc_no_mining/..._50_top10rules` | 903.4 | 3175 | 68.2% |

History size per test sample:

| Variant | avg facts | max |
|---|---|---|
| `gtkg/test` | 36.35 | 50 |
| `gtkg/test_inverse_included` | 39.15 | 50 |
| `gtkg/test_inverse_included_num_facts` | 39.15 | 50 |
| `ragtkgc/test_inverse_included_num_facts` | 40.60 | 50 |
| `ragtkgc_no_mining/test_inverse_included` | **831.49** | **6049** |
| `ragtkgc_no_mining/..._50_top10rules` | 34.90 | 50 |

---

## 3. Findings register

Severity: **H** blocks or invalidates results · **M** worth fixing · **L** cosmetic.

### F1 — `period=24` double-scaling in mining · H · OPEN

`learn.py` calls `get_unique_quads_per_rels(dataset, path)` without `period`, so
the default 24 applies and `basic.py:67` computes `quad[3] * 24` on a timestamp
that is **already** hour-scaled. Start-quad timestamps reach 174528 vs the
graph's 7272 — exactly 24×. 95.6% of non-zero-ts start quads exceed the entire
graph's maximum timestamp.

Two consequences:

1. `temporal_walk.py:110` filters `next_edges[:,3] < cur_ts`. With `cur_ts`
   inflated past every graph timestamp the filter excludes nothing, so for
   length-1 rules only the cyclic constraint survives. Bodies occurring at or
   after the head qualify.
2. `sample_next_edge` computes `np.exp(tss - cur_ts)`. float64 `exp(-x)`
   underflows to 0.0 at x ≥ 746. For inflated timestamps that holds from day
   ~16.5 onward (~95% of the timeline), so the sum is 0, `np.random.choice`
   raises `ValueError`, and the bare `except` falls back to **uniform**. The
   `exp` in the rule-bank filenames never took effect for `ragtkgc`.

Affects `ragtkgc` and `ragtkgc_no_walks`. **Not** `gtkg` (start edges come from
`Grapher`). **Not** `ragtkgc_no_mining` (never calls `sample_walk`; reads only
relation ids from `unique_quads`, ignores timestamps; `estimate_confidence`
works in the `Grapher` domain).

`retrieve.py` uses `period = 1`, and `TLR._time_period()` correctly **divides**
by 24 to produce day floats. The same constant is used in opposite directions on
the two sides of the pipeline.

Not present in the GenTKG baseline — this one is local.

### F2 — inverse body relations were subject-anchored · H · FIXED by flag, INHERITED

Retrieving `_b(s,o)` from a forward-only `all_facts` needs two transformations:
map the relation (`b % num_relations`) **and** move the anchor to the object
column. `TLR_gentkg.py:123-126` does only the first; `s_0` is built from
`col_sub == test_sub` alone. So an inverse rule silently impersonated its
forward twin.

Measured on ICEWS14 test, `gtkg/test` vs `gtkg/test_inverse_included`
(5190 of 7371 samples changed):

| | query subject is fact OBJECT | query subject is fact SUBJECT |
|---|---|---|
| removed (old only) | 0 | 68181 |
| added (new only) | 77732 | 7289 |

Twin collapse under the old code — `b` and `_b` for the same head retrieved
identical fact sets, differing only in `conf` and `rule_id`:

| Bank | rules | twin pairs | rules collapsing to duplicates | inverse rules with no forward twin |
|---|---|---|---|---|
| gtkg | 8027 | 1647 | 3294 (41.0%) | 2373 |
| ragtkgc | 33972 | 11196 | 22392 (65.9%) | 5777 |
| ragtkgc_no_mining | 41068 | 14604 | 29208 (71.1%) | 5930 |

Effective distinct filters under the old code: gtkg 6380, ragtkgc 22776,
no_mining 26464. Mean |conf(b) − conf(_b)| among twins is 0.030, so twins rank
adjacently — worst case for `top_k_rules`, which could spend consecutive slots
on duplicates.

Consequence for the narrative: the old paper's implied chain "more rules →
better retrieval" could not cash out, since ~66% of ragtkgc's rules collapsed.
`--inverse_body_object_match` is a **correction to the baseline method**, and
`ragtkgc_no_mining` has the most to gain from it (highest twin rate). Apply it
uniformly to all miners so the comparison stays fair.

### F3 — training discards long samples · H · CONFIG DECISION

`training_T5.py:176-177` (and `:188-189` for eval): without
`--tail_truncate_long_inputs` the dataset is **filtered**, not truncated —
every sample above 511 tokens is deleted. That removes 67–71% of the train
split, leaving only atypically short histories. With the flag, all samples are
kept and long ones are clipped to the last 512 tokens.

This, not the input format, is why non-`fancy` models behave differently: they
were trained on a different (and much smaller) dataset.

### F4 — `num_facts` default flipped from 50 to None · M · UNDERSTOOD

Old `retrieve.py` hardcoded `num_facts = 50` and appended the suffix
`_num_facts` when `--early_stop_at_num_facts` was **absent**. New `retrieve.py`
takes `--num_facts` defaulting to `None` (= uncapped) and names the suffix
`_n{N}`. Result: the `ragtkgc_no_mining` base retrieval ran uncapped at 831
facts/sample average.

### F5 — `data/original/icews18/all_facts.txt` missing · H · **FIXED 2026-09-03**

`retrieve.py` reads the candidate fact pool from
`data/original/{ds}/all_facts.txt`. ICEWS18 had only `train/valid/test.txt` and
the id maps, so retrieval failed at pre-run validation.

Copied from `data/processed_new/icews18/all_facts.txt` (32,851,054 bytes) after
verification. Note that `wc -l` undercounts both datasets' `all_facts.txt` by one
because neither file ends in a newline — count records, not lines.

| Check | Result |
|---|---|
| records | 468,558 = 373,018 train + 45,995 valid + 49,545 test |
| chronological sort | 0 out-of-order adjacent pairs in 468,557 |
| format | 5 tab-fields (4 + trailing empty), identical shape to ICEWS14 |
| date range | 2018-01-01 .. 2018-10-31, 304 distinct dates |
| dates present in `ts2id` | 304 of 304 |
| `ts2id` scale | ids 0..7272, step 24 — hour-scaled, same as ICEWS14 |
| entities | 23,033 in file, 23,033 in `entity2id`, 0 unknown |
| relations | 256 in file, 256 in `relation2id`, 0 unknown |

Two side results: the 256 relations confirm the hardcoded `num_relations` for
ICEWS18 (F6), and this closes the "chronological sort not yet verified for
ICEWS18" gap left open in §2.

### F6 — `num_relations` hardcoded · L · **FIXED 2026-09-03**

`retrieve.py` hardcoded 230/256/238/24 per dataset name. It duplicated
`len(relation2id)` and was load-bearing twice in `TLR.build_tl`: the modulo maps
an inverse relation id back to its forward relation, and `>= num_relations` is
the direction test. A wrong value would have retrieved the wrong relation's facts
*and* misclassified rule direction at once, with no error raised.

Replaced with `num_relations = len(relations)` immediately after
`relation2id.json` is loaded in the split loop (so no extra file read), plus a
log line recording the value per split. Placed after pre-run validation, so a
missing `relation2id.json` still produces the proper validation error rather than
an unhandled exception.

Behaviour-neutral, verified against the constants it replaced:

| dataset | hardcoded | derived |
|---|---|---|
| icews14 | 230 | 230 |
| icews18 | 256 | 256 |

GDELT (238) and the fallback (24) could not be checked — no data locally — but
they are now correct by construction for any dataset.

### F7 — `idx_test` arithmetic wrong for train/valid · L · **FIXED 2026-09-03 (deleted)**

`TLR.py:215` assumes the split being processed sits at the end of `all_facts`:

| split | formula(i=0) | true row | |
|---|---|---|---|
| train | 15885 | 0 | wrong |
| valid | 82216 | 74845 | wrong |
| test | 83359 | 83359 | correct |

The guard existed to prevent the target quadruple leaking into its own history.
Inherited from `TLR_gentkg.py:82-87`.

**Deleted, and the deletion is provably output-neutral.** For every split,
`idx_test` points at or after the query's own row in `all_facts`:

| split | `idx_test` | query's own row | |
|---|---|---|---|
| train | `15885 + i` | `i` | `idx_test > own row` |
| valid | `82216 + i` | `74845 + i` | `idx_test > own row` |
| test | `83359 + i` | `83359 + i` | equal |

`all_facts` is chronologically sorted, so `col_time[idx_test] >=
col_time[own row] == test_time`, while `s_t` requires strictly `< test_time`. The
guard therefore cannot fire on any dataset whose `all_facts` is sorted and whose
splits are concatenated in order — which F5 verified for both.

Measured by replicating `tlogic_prepro`'s arithmetic over every sample:

| dataset | train | valid | test |
|---|---|---|---|
| icews14 | 0 / 74,845 | 0 / 8,514 | 0 / 7,371 |
| icews18 | 0 / 373,018 | 0 / 45,995 | 0 / 49,545 |

Zero removals in 975,298 samples. Replaced with a comment recording why no guard
is needed, and noting that if the time filter is ever relaxed to `<=` a guard
must be reinstated — indexed by the split's offset into `all_facts`, not by
arithmetic from the end.

### F8 — no same-timestamp facts · L · BY DESIGN, DOCUMENT IT

`col_time < test_time` is strict, so at day granularity the history can never
contain same-day facts. Defensible (prevents co-occurrence leakage) but it
bounds what the history can hold and should be stated in the paper.

### F9 — `ti.sleep(0.001)` per sample · L · DELETE

`TLR.py:449`, inherited from `TLR_gentkg.py:157`. ~75 s of pure sleep on the
74845-sample train split.

### F10 — `--use_ids` on `create_json_train.py` is a no-op · L

`convert_txt_to_json` accepts `use_ids` but never reads it; the `entities`
argument is also unused. Whether targets are ids or names is decided entirely by
what `retrieve.py` wrote into `test_answers`.

### F11 — `_ids` answers files are 50% blank lines · L · HARMLESS

`retrieve.py:188-189` passes the raw `readlines()` of the id-encoded split
straight to `write_txt`, which adds another newline. Verified: 14742 lines for
7371 records. `convert_txt_to_json` filters blanks uniformly, so alignment with
the history chunks is preserved (7371 == 7371).

### F12 — LLaMA2-7B trains on 1024 samples by design · M · REPORT, DO NOT REMOVE

`training_LLaMA.py:183-188` subsamples any training file larger than 1024 rows
down to exactly 1024 evenly-spaced examples, unconditionally and with no flag.

This is **deliberate**: it replicates GenTKG's data-efficiency setup (see the
comment at `:172`). The cap stays — removing it would break comparability with
the baseline. LLaMA also stays in the paper.

The length filter on this path is `< 4095` (LLaMA context), so almost nothing is
dropped for length; the 1024 cap is the binding constraint.

What does need addressing is that the paper compares across regimes without
saying so:

| Model | training examples | epochs |
|---|---|---|
| Flan-T5-Small | 74845 (≈24k without the truncation flag, per F3) | 3 |
| LLaMA2-7B | 1024 | 1 |

State this in the setup section. The claim "Flan-T5-Small beats LLaMA2-7B on
H@1" is not a model-capacity result at these budgets.

### F12b — GenTKG reports better LLaMA results at an equal or smaller budget · OPEN INVESTIGATION

The anomaly worth chasing. Candidate causes, cheapest first:

| Candidate | Where | Cost to test |
|---|---|---|
| Object rendered `18.Thailand` (id + `.` + name) vs bare `Thailand` | `TLR_gentkg.py:174-177` vs `TLR.py:122-127` | retrieval re-run, no GPU |
| Timestamp `292` vs `292.0` | `build_history_query`, both files | same run |
| `max_new_tokens` (26 for LLaMA/icews14) | `model_utils.py:68` | none |
| Epochs (1 here) | `training_LLaMA.py:138` | needs their config |
| LoRA config (r=8, alpha=16, `bias='lora_only'`, rslora) | `training_LLaMA.py:125-130` | needs their config |
| Early-stop retrieval (their default) vs all-rules-then-trim | F4 | retrieval re-run |
| Prediction extraction and ranking | `model_utils.py`, `utils.py` | none |

Leading hypothesis: the id-prefixed object gives a causal LM a short
unambiguous token to copy, which would matter more for LLaMA than for a
seq2seq model — consistent with the gap appearing on LLaMA and not Flan-T5.
Untested.

### F13 — adaptive training controller is inert at 3 epochs · M · DESCRIBE HONESTLY

Settings in `training_T5.py`: `patience=2`, `max_lr_reductions=2`,
`min_delta=5e-4`, `ema_alpha=0.4`. Eval runs once per epoch → 3 evaluation
points. Trace: eval 1 sets the baseline (CONTINUE); eval 2 gives
`patience_counter=1 < 2` (CONTINUE); eval 3 gives `patience_counter=2` →
REDUCE_LR, applied with zero training steps remaining.

`STOP` is unreachable: it needs `_lr_reduction_count >= 2` (impossible from one
firing) and `_marginal_improvement_negligible()`, which returns `False` whenever
`len(_eval_history) < 4` (`training_controller.py:307-308`).

So the callback never alters training. The real effect of `--eval_file_path` is
`load_best_model_at_end`, which is plain `Trainer` behaviour. Do not describe
this as adaptive LR scheduling in the paper unless the epoch count is raised.

Also `grad_window` is accepted in `__init__` but never stored; `_reset_state`
hardcodes `deque(maxlen=20)`.

### F14 — checkpoint selection differs with/without `--eval_file_path` · M · CONFOUND

`load_best_model_at_end = bool(args.eval_file_path)` and
`metric_for_best_model = "eval_loss"`. With an eval file the best epoch is
restored; without one, the **last** epoch is kept. Nothing in the model name
records which. A second confound between `fancy` and non-`fancy` models,
independent of F3.

### F15 — the LLaMA path was never updated to the new pipeline · H · BLOCKS LLAMA RE-RUN

`training_LLaMA.py` accepts only `--dataset`, `--trained_model_name`,
`--output_dir`, `--train_file_path`. No `--tail_truncate_long_inputs`, no
`--eval_file_path`, no controller, no best-checkpoint selection. Re-running
LLaMA today would put it on the old regime while Flan-T5 is on the new one.
Both options must be added first if LLaMA appears in the new paper.

### F16 — completion collator is tokenizer-brittle · M

In `DataCollatorForCompletionLM` (`training_LLaMA.py`):

1. Hardcoded LLaMA-2 token ids for `")]"→")"` (4638→29897), `"))]"→"))"`
   (28166→876), `".]"→"."` (5586→29889).
2. Unexplained dataset-specific off-by-one: masks from `idx+1` for `icews14`,
   from `idx` otherwise (`:105-108`).
3. `np.where(...)[0][-1]` raises `IndexError` when the target's first token is
   absent, so the `if ... is None: raise RuntimeError` guard below it is dead.

### F17 — Flan-T5 emits exactly one candidate, so H@3/H@10 are undefined · H · DECISION NEEDED

`model_utils.py:84-93` calls `permute(..., dec_cand=1, ...)`. With `dec_cand=1`
only the top-1 token is followed at each step, so exactly one sequence is
produced. Verified: `n_preds == 1` for all 7371 samples in both Flan-T5 result
files. `rank` can therefore only be 1 or "not found", making
H@1 ≡ H@3 ≡ H@10 by construction — a decoder property, not a model property.

`permute` branches `dec_cand^depth`, depth bounded by `maxl` (42 for
Flan-T5/ICEWS14), so raising `dec_cand` risks 3^42 in the worst case even though
it usually terminates early on `]`/`</s>`. The safe fix is beam search with
`num_beams=k, num_return_sequences=k`, which is what the LLaMA branch already
does.

**Deliberate tradeoff, not an oversight**: `dec_cand > 1` and beam search were
both rejected as too slow at this scale. So Flan-T5 reports **H@1 only** and its
H@3/H@10 columns should be dropped rather than printed as duplicates.

Consequence for the baseline comparison: if GenTKG obtains its H@3/H@10 by
**ranking the whole entity vocabulary by likelihood** rather than by generating
candidates — a common and much cheaper approach in TKG-ICL work — then their
H@3/H@10 are not comparable to these numbers at all, and that would explain both
their stronger results and their lower cost. Check this first in the original
repo (F12b).

### F18 — LLaMA emits exactly three candidates, so H@10 == H@3 · M

`model_utils.py:130-136`: `num_beams=3, num_return_sequences=3`. Verified:
`n_preds == 3` for all samples. H@10 is not reportable for either model as the
code stands; the published table showing only H@1/H@3 was correct.

### F19 — `cs` is character-frequency cosine, not semantic similarity · H · PAPER CLAIM

`utils.cosine_similarity` builds `Counter(target)` over a **string**, so it
counts characters. Measured with the verbatim function:

| target | prediction | cs |
|---|---|---|
| China | China | 1.0000 |
| China | Chian (anagram) | **1.0000** |
| China | Chinatown | 0.8090 |
| Malaysia | Australia | 0.7252 |
| Thailand | Japan | 0.5976 |
| Thailand | Netherlands | 0.5262 |
| Barack_Obama | Citizen_(India) | 0.2128 |

The published CS range is 61.1%–69.9%; arbitrary unrelated country names score
0.53–0.73. The reported values sit in the metric's noise floor and an anagram
scores 1.0. The paper's claim that responses were "semantically close … as CS@1
values range between 61.1% and 69.9%" is not supported by this metric.

Fix: drop CS, or replace with an embedding cosine (`sentence-transformers` is
already a dependency via `--use_llm_similarity`). Recomputable from existing
`results/*.jsonl` with no GPU run.

### F20 — BERTScore is un-rescaled and top-1 only · M · UNVERIFIED

`bertscore.py` uses `predictions[0]` / `targets[0]` and `lang="en"` without
`rescale_with_baseline`. Un-rescaled BERTScore is high in absolute terms for
almost any short English string pair, so the published 0.905–0.923 range may
also be near a floor. Hypothesis only — recompute with
`rescale_with_baseline=True` before relying on those numbers.

### F21 — `'gtkg' in finetuned_model` also matches `ragtkgc` · L

`model_utils.py:143` is a substring test, and `"gtkg"` is a substring of
`"ragtkgc"`. So every rule-based LLaMA model gets `.split('\n')[0]` post-
processing while `raw` and `standard` do not. Measured impact is small
(17/22113 `standard` predictions contain a newline, 0 for `raw`), so it does not
explain the baseline gap — but it is an inconsistent post-processing path
between baselines and treatment. Make it explicit.

### F22 — partial o4-mini results file · L

`o4-mini-2025-04-16_icews14_gtkg_test_ids_inverse_included_num_facts_tail_truncate.jsonl`
has 1291 of 7371 samples. `compute_metrics_from_results.py` divides by the
actual count, so its H@1 of .259 is over a 17% subset and is not comparable.

### F23 — exact-match scoring · L · DOCUMENT

`update_metric` uses exact string equality. A substring variant
(`hit1p/hit3p`) exists in `HitsMetric` but is commented out of `dump()`.

### F24 — the RAG test set changes three things at once · H · CONFOUND

`rag_with_gpt_4_1.py:61` hardcodes the history source to the **ragtkgc** test
file regardless of which model will be evaluated. A substituted sample therefore
differs from its original in history-modeling algorithm, added GPT facts, **and**
added instruction text.

H@1 split by substituted (3000) vs untouched (4371) indices:

| model | file | substituted | untouched | overall |
|---|---|---|---|---|
| standard | base | 0.304 | 0.356 | 0.335 |
| standard | +RAG | **0.018** | 0.356 | **0.218** |
| gtkg | base | 0.000 | 0.590 | 0.350 |
| gtkg | +RAG | 0.012 | 0.590 | 0.355 |
| ragtkgc | base | 0.010 | 0.591 | 0.354 |
| ragtkgc | +RAG | 0.013 | 0.590 | 0.355 |

Untouched columns are identical between base and RAG, so substitution is the
only difference. The whole `standard` drop from .335 to .218 comes from the 3000
substituted samples — an out-of-distribution artifact, not a RAG result.

### F25 — the selection is circular · H · FRAMING

Samples were chosen because the evaluated models failed on them, so `gtkg`
scores 0.000 and `ragtkgc` 0.010 on the selected subset *before* RAG. Post-RAG
numbers on that subset start from a floor of zero and cannot be read as general
improvement.

Real overall gains: gtkg .350 → .355 (+0.005), ragtkgc .354 → .355 (+0.001) —
12 and 3 extra correct samples out of 3000.

### F26 — selection criterion differs from the reported metric · M

The selector uses `d['targets'][0] not in pred[0]` (substring); reporting uses
exact match.

### F27 — the GPT index cannot be reproduced from the files on disk · M

On the 3000 selected indices, six of the eight selection files score
0.000–0.028 as expected, but **both `standard` files score ~0.28–0.31**. The
files are verified index-aligned (identical timestamp/entity/relation/target at
every row for raw, gtkg, ragtkgc), so this is not an ordering bug — the on-disk
`standard` results appear not to be the ones used to build the index.

Also contradicts the paper's claim that the selected samples are ones "where no
version of any model was able to predict the correct output".

### F28 — the `standard` dataset uses a different prompt format · M · REGENERATE

Row 0 across result files:

```
gtkg / ragtkgc / raw : ('334.0', 'Malaysia', 'Express_intent_to_cooperate', 'Thailand')
standard             : ('8016',  'Malaysia', 'Express_intent_to_cooperat',  'Thailand')
```

Timestamps are raw hours rather than day floats, and the relation name loses its
final character because `run_hf.py:324` does `rel = rel.strip()[:-1]` to strip a
trailing comma that the `standard` query line does not have. Metrics are
unaffected (they use only `targets`/`predictions`), but the recorded fields are
corrupted and the prompt format is inconsistent. `raw` is unaffected. Concrete
reason to regenerate `standard`.

### F29 — GPT index written and read at different paths · L

Written to `{dataset}_gpt_index.json` in the working directory; read by
`run_hf.py:200-202` from `test_rag/{dataset}_gpt_index.txt`. Manual move needed.

### F30 — argparse `type=bool` makes flags impossible to disable · L

`bool("False")` is `True`, so a `type=bool` flag cannot be set to False from the
CLI. Should be `action="store_true"` / `store_false`.

- `retrieve.py` — **fixed** (F30): `--rule_length_all` is now `--length_1_only`,
  a presence flag with `dest="rule_length_all"`, `action="store_false"`.
- `rag_with_gpt_4_1.py` — `--use_llm_similarity` and `--no_similarity` still have
  the defect. Not in the Phase 1–5 plan; that script is not part of the re-run.

### F31 — hardcoded selection file list · M · BLOCKS RAG REBUILD

`get_wrong_predictions_count` hardcodes 8 result filenames in the old naming
scheme; they will not resolve against the new run's files.

### F32 — instruction text only on RAG samples · M · PAPER CLAIM

RAG'd contexts gain "The above set of quadruples are factually correct. …"
plus "Input quadruple: ", while §4.1 states no system prompt is used so that
PLMs and LLMs are compared fairly.

### F33 — the reported mining time excludes the new algorithm's main cost · H · TABLE 3

`learn.py:151` starts the timer **after** `Grapher`, `Temporal_Walk` and
`get_unique_quads_per_rels` have already run. Measured on ICEWS14,
`ragtkgc_no_mining`:

| Phase | Time | In the reported figure? |
|---|---|---|
| imports | 1.08 s | no |
| `Grapher` | 0.31 s | no |
| `Temporal_Walk.__init__` | 1.93 s | no |
| `get_unique_quads_per_rels(infer_from_type=True)` | **31.43 s** | **no** |
| walk / rule construction + confidence | printed figure | yes |

So 33.7 s of ICEWS14 work is invisible, and for `ragtkgc_no_mining` that
excluded block **is** the candidate-generation algorithm. `gtkg` skips
`get_unique_quads_per_rels` entirely, so its published 16.6 s and the new
algorithm's figure are not measuring comparable spans. Table 3 must either
report end-to-end wall clock for every miner or break the phases out.

### F34 — `ragtkgc_no_mining` is exhaustive over candidates, not over rules · H · PAPER CLAIM

Measured on ICEWS14:

| Quantity | Value |
|---|---|
| relations incl. inverse | 460 |
| all conceivable (head, body) pairs | 211,600 |
| type-compatible candidates enumerated | **62,156** |
| rules stored | **41,068** (66.1% of candidates) |
| candidates dropped | 21,088 |

`rule_learning.py:287` (`if rule["conf"]:`) silently discards every rule whose
confidence is 0 — a body that occurs but is never followed by the head. Verified:
zero rules with `conf == 0` in any bank on disk.

The correct claim is therefore **"all length-1 rules that are type-compatible and
have positive confidence"**, which is 19.4% of the full pair space — not "all
length-1 rules". The dropped 21,088 are genuinely useless for retrieval, so the
filter is right; only the wording needs fixing.

### F35 — confidence is not comparable across mining algorithms · H · AFFECTS top_k / threshold

`create_rule` passes `full_samples = custom_generated`, so:

| Algorithm | Bodies used for `conf` | `body_supp` min / median / max |
|---|---|---|
| gtkg | 500 random draws | 1 / 393 / **497** |
| ragtkgc | 500 random draws | 1 / 136 / **497** |
| ragtkgc_no_mining | **every edge** of the body relation | 1 / 141 / **12,980** |

The 497 ceiling is the `num_samples=500` cap showing through. Taking
`no_mining` as exact, the sampled estimator's error on shared rules is:

| Comparison | shared rules | exactly equal `conf` | median abs. error | max abs. error |
|---|---|---|---|---|
| gtkg vs no_mining | 8,027 | 19.6% | 0.0022 | 0.0569 |
| ragtkgc vs no_mining | 33,972 | 45.1% | 0.0003 | 0.0551 |
| gtkg vs ragtkgc | 7,577 | 19.9% | 0.0035 | 0.0813 |

Consequence: `--confidence_threshold 0.5` and `--top_k_rules 10` select on a
noisy quantity for `gtkg`/`ragtkgc` and an exact one for `no_mining`. Any
retrieval comparison that uses either flag is partly comparing estimator noise.
Safest for the paper: run the main comparison with **no** confidence filter, and
treat `--top_k_rules` as a separate ablation on `no_mining` alone.

### F36 — `ragtkgc_no_mining` is a strict superset of both other banks · GOOD NEWS · USE THIS

Measured by `rule_id` intersection over the bank files (ICEWS14):

| | rules | ⊆ no_mining? |
|---|---|---|
| gtkg | 8,027 | yes — all 8,027 present |
| ragtkgc | 33,972 | yes — all 33,972 present |
| ragtkgc_no_mining | 41,068 | — |

`gtkg ∪ ragtkgc` = 34,422, so `no_mining` contributes **6,646 rules neither
random-walk method found**. Also note `gtkg ⊄ ragtkgc`: they share only 7,577, so
**450 rules were found by uniform random walks but missed by `ragtkgc`'s
start-quad walks** — direct evidence that the pre-`no_mining` approach had a
coverage gap, which is exactly the claim the new algorithm is meant to support.

This is the cleanest positive result available and it costs nothing to report.

**The superset relation is provable, not merely measured.** Let a length-1 rule
`head(s,o) ← body(s,o)` be found by any walk-based miner on the same training
graph. The walk instantiated both edges on real entities, so `s`'s relation
signature contains `body` and `o`'s contains `body`'s inverse — the rule is
therefore type-compatible and gets enumerated. The walk is cyclic, so a head edge
exists at a later timestamp than the body edge, giving `rule_supp >= 1` and hence
`conf > 0`, so it survives the zero-confidence filter. Any length-1 rule
discoverable by sampling is necessarily in the exhaustive bank. This holds for
`gtkg`, `ragtkgc`, and any future sampling miner, and is a one-sentence argument
in the paper rather than a table.

### F41 — `standard` has a generator after all, but it is unreachable · M · UNBLOCKS BASELINE

`TLR.build_bs` (`TLR.py:181`) is documented as *"Pure entity-based retrieval (no
rules); used as a baseline"* — exactly what the `standard` dataset is. It is
selected by `retrieve_type == 'bs'` (`TLR.py:514`), and `retrieve.py` does expose
`--retrieve_type`.

But `retrieve.py:202-221` constructs `Retriever(...)` **without passing
`retrieve_type`**, so it keeps its default `'TLogic'`, and nothing ever calls
`build_bs`. `--retrieve_type` is read, logged at `:103`, and then discarded.

So rebuilding the `standard` baseline under the current regime is a small patch
(pass `retrieve_type` through, dispatch on it) rather than a reimplementation.
Supersedes the earlier conclusion that no code path existed.

### F42 — confidence does not affect the retrieved history unless a flag says so · CLARIFICATION

`TLR.py:412-414`:

```python
idx = sorted(fact_to_conf.keys(), reverse=True)   # fact index = recency
if num_facts is not None:
    idx = idx[:num_facts]
```

Facts are trimmed by **recency**, never by confidence. All fired rules contribute
their facts to one union first. So with `--confidence_threshold`, `--top_k_rules`
and `--early_stop_at_num_facts` all off, the history is a function of rule-bank
*membership* only, and the cross-algorithm confidence incomparability of F35 does
not touch the result.

`--early_stop_at_num_facts` is the exception: it breaks the rule loop
(`TLR.py:333-335`) in confidence-descending order, which does make the noisy
estimates matter. Keep it off for the headline comparison.

### F37 — `common_rule_pool.json` is not a reliable provenance record · M

Checked against the bank files:

| Algorithm | pool agrees with bank | disagrees |
|---|---|---|
| gtkg | 8,027 | 0 |
| ragtkgc | 27,905 | **6,067** |
| ragtkgc_no_mining | 41,068 | 0 |

The pool also claims `ragtkgc` found all 8,027 `gtkg` rules, while the files
share only 7,577. Cause: `algorithm_stats[alg]` is keyed by algorithm name only,
so re-running an algorithm **overwrites** the previous run's numbers, and
`update_common_rule_pool_from_rules_file` can import a different run's file under
the same label. Do not cite the pool for anything; derive provenance from the
bank files by `rule_id`.

### F38 — `-p` changes the rule bank, not just the runtime · M · REPRODUCIBILITY

`learn_rules` does `np.random.seed(seed)` inside **every** worker with the same
seed, then each worker handles a different slice of relations. So which random
draws a given relation receives depends on how relations were grouped, i.e. on
`--num_processes`. Two runs with identical `-d -l -n -s` but different `-p`
produce different banks. Deterministic and therefore unaffected for
`ragtkgc_no_mining`; affects `gtkg` and `ragtkgc`.

Also `if seed:` (`learn.py:60`) means **`-s 0` is silently ignored**.

### F39 — node "types" are full relation signatures · L · DESCRIBE ACCURATELY

`get_unique_quads` labels a node by the exact set of relations it participates
in, inverse relations included. ICEWS14 train:

| | |
|---|---|
| entities | 6,616 |
| distinct signatures ("types") | 3,514 |
| signatures held by exactly one node | 3,168 (90.2% of types) |
| entities that are their own type | 3,168 (47.9% of entities) |
| largest type | 502 entities |
| signature size (min / median / max) | 1 / 6 / 157 |

So the labelling is real but very fine-grained: about half the entities are in a
class of one. Do not describe it as semantic typing (country, politician, …); it
is a structural signature. The mechanism it provides is correct — a body relation
`st` is admitted for head `rel` only when the subject's signature contains `st`
and the object's contains `st`'s inverse, which is exactly the condition for a
length-1 cyclic rule to be instantiable.

### F40 — stats file pairs dictionaries by insertion order · L · LATENT

`basic.py:163` does `zip(quads.items(), quads_all.items())`. The two dicts happen
to be built in the same order today (the head relation is always its own
type-compatible body, so every relation in `quads_all` also lands in `quads`), so
`node_labelling_stats_{ds}.txt` is currently correct. If that ever stops holding,
every line after the first gap reports the wrong relation, silently.

`quads_all` also retains all 149,690 expanded quads purely to print a count.

### F43 — retrieval rescans all 90,730 facts once per rule per sample · H · THE BOTTLENECK

`TLR.py:316`, inside the per-rule loop:

```python
for k in idx_chain:
    chain_rule = self.chains[str(head_rel)][k]
    rel = chain_rule['body_rels'][-1] % self.num_relations
    idx_rel = np.where(self.col_rel == self.rel_keys[rel])[0]   # full scan, every rule
```

`self.col_rel` is a fixed 90,730-element array of **strings** and `self.rel_keys`
never changes, so `idx_rel` is a pure function of `rel` with only 230 possible
values — yet it is recomputed for every rule of every sample.

Measured on ICEWS14 by summing, over every query, the number of rules its head
relation has. `all_facts` is 90,730 rows, so each iteration is one full scan.

| Bank | rules/head (median / mean / max) | rule iterations, test | scans × 10⁹ | train estimate |
|---|---|---|---|---|
| gtkg | 17 / 19 / 65 | 276,588 | 25 | 2.8 M iterations |
| ragtkgc | 57 / 81 / 333 | 1,836,897 | 167 | 18.7 M iterations |
| ragtkgc_no_mining | 76 / 95 / 344 | **1,965,954** | **178** | **20.0 M iterations** |

So the exhaustive bank costs ~7× the gtkg bank in retrieval time purely through
this loop, and the train split is ~10× the test split again. Caching collapses it
to at most 460 scans total — a **4,274×** reduction in scan count for
`ragtkgc_no_mining`.

Two more scans are cacheable for the same reason:

| Line | Call | Distinct inputs |
|---|---|---|
| 216 | `np.where(self.col_time < test_time)` | 365 dates — and `col_time` is sorted, so `searchsorted` applies |
| 220 | `np.where(self.col_sub == test_sub)` | 6,616 subjects |
| 290 | `np.where(self.col_obj == test_sub)` | 6,616 subjects |

Caching all four turns tens of millions of full-array scans into a few thousand.
This is almost certainly why retrieval is measured in hours, and it is a
correctness-neutral change. **Highest-value fix available for the re-run budget.**

### F44 — `--early_stop_at_num_facts` alone raises TypeError · M · **FIXED 2026-09-03**

`TLR.py:334` is `if len(cumulative_facts) >= num_facts`, and `num_facts` defaults
to `None`. Passing `--early_stop_at_num_facts` without `--num_facts` crashed with
`'>=' not supported between instances of 'int' and 'NoneType'` partway through
the first sample — meaning the flag had never been exercised on its own.

Fixed with an explicit `ValueError` in `Retriever.__init__`, alongside the
existing `early_stop + top_k_rules` warning so the two validations sit together.
Raising rather than silently ignoring the flag: asking to stop at a cap without
giving a cap is a user error, and quietly doing nothing would be a second silent
failure. The check also covers programmatic construction, not just the CLI.

Verified with the real constructor:

| case | result |
|---|---|
| `early_stop=True`, `num_facts=None` | `ValueError` with an actionable message |
| `early_stop=True`, `num_facts=50` | constructs normally |
| `early_stop=False`, `num_facts=None` (headline config) | constructs normally, unaffected |

No guard added at the comparison site in `build_tl` — the constructor check makes
that state unreachable.

### F45 + F30 — multi-hop retrieval now rejected instead of silently wrong · **FIXED 2026-09-03**

`TLR.py:314`: `body_rel_last = chain_rule['body_rels'][-1]`. For a length-2 rule
`head(s,o) ← b1(s,x), b2(x,o)`, retrieval matches `b2` anchored at the **query
subject**, i.e. it behaves as if the rule were `head(s,o) ← b2(s,o)`. The
intermediate entity `x` is discarded, so the retrieved facts are not instances of
the rule at all.

Inert today, but `--rule_length_all` defaulted to `True` and could not be set to
`False` from the CLI (F30), so the moment a longer bank was used the mis-anchoring
would have been silent.

Fixed as a pair (option C — make the switch real *and* make the unimplemented case
announce itself):

**F30.** `--rule_length_all` (`type=bool`) replaced by `--length_1_only`
(`action="store_false"`, `dest="rule_length_all"`). argparse applies `type=` to
the raw string and `bool()` is truthiness, not parsing, so the old form was
unusable:

| invocation | old result |
|---|---|
| omitted | `True` |
| `--rule_length_all False` | **`True`** |
| `--rule_length_all 0` | **`True`** |

No existing invocation passed it (`commands.txt` never does), so the rename breaks
nothing. README updated.

**F45.** `Retriever.__init__` now counts rules with `len(body_rels) > 1` and
raises `NotImplementedError` when `rule_length_all` is set, naming the count and
the flag to pass. Checked once over the bank rather than per sample, so it costs
nothing in the hot loop, and it fires before any retrieval work.

| bank | flag | result |
|---|---|---|
| length-1 | default | OK — today's behaviour, unchanged |
| length-1 | `--length_1_only` | OK |
| mixed | default | `NotImplementedError`, counts the 2 offending rules |
| mixed | `--length_1_only` | OK, filters to length-1 |

Every bank on disk is length-1 (`{1: N}` for all five ICEWS14 and both ICEWS18
banks), so the default remains safe and no current command changes behaviour.

### F46 — a test query's history may contain earlier test facts · L · DOCUMENT

`s_t` is built from `all_facts` (train + valid + test, 90,730 rows) with the sole
constraint `col_time < test_time`. So a test query at time T can retrieve facts
from the test split itself, provided they are strictly earlier.

The splits are chronologically disjoint, which bounds the effect:

| Split | rows | time range | dates |
|---|---|---|---|
| train | 74,845 | 0 – 7,272 | 2014-01-01 – 2014-10-31 |
| valid | 8,514 | 7,296 – 7,992 | 2014-11-01 – 2014-11-30 |
| test | 7,371 | 8,016 – 8,736 | 2014-12-01 – 2014-12-31 |

Candidate pool per test query (facts strictly earlier than the query):

| | min | median | mean | max |
|---|---|---|---|---|
| total pool | 83,359 | 87,015 | 86,912 | 90,553 |
| of which test-split | 0 | 3,656 | 3,553 | 7,194 |

So test facts are **4.1% of the average candidate pool**; the first December
query sees none and the last sees 7,194. Because the splits are disjoint in time,
**train and valid queries can never see test facts** — the exposure is one-way.

This is the standard TKG extrapolation setting (you know the past at prediction
time) and GenTKG and ISI both do the same, so it is not leakage. But the paper
should state that history is drawn from the full corpus up to the query
timestamp, rather than let a reviewer assume train-only history.

### F47 — prompt timestamps are floats · L

`TLR.py:127`, `:492`, `:504` render the day index as `int(time_in_id) / period`,
producing `334.0`. GenTKG emits `int(time/period)` → `292`. Worth aligning while
the answer format is being changed anyway.

### F48 — the three uses of "24" are unrelated · CORRECTION

An earlier note in this file described `period` as "the same constant used in
opposite directions". That framing was wrong; there are three independent
conversions that happen to share a number.

| Site | Value | Operation | Verdict |
|---|---|---|---|
| `retrieve.py:111` → `convert_dataset` → `id_words` | `period = 1` | multiplies before an `ts2id` lookup | **required** — `test.txt` col 3 is already the hour id (`8016`), and `flip(ts2id)` is keyed on it (`8016 → 2014-12-01`). Any other value breaks the lookup. |
| `TLR._time_period()` (`:114`), used at `:127`, `:492`, `:504` | `24` | **divides** the hour id for display | **correct** — `8016 / 24 = 334.0`, the day index shown in the prompt |
| `basic.py:63` default, used at `:67` | `24` | **multiplies** `train.txt` col 3 | **wrong** — that column is already hour-scaled (step 24, max 7272), so this inflates it 24× |

`learn.py:39,41` never pass `period`, so the wrong default always applies during
mining. Fix: pass `period=1` (or delete the multiplication). Affects `ragtkgc` and
`ragtkgc_no_walks` only.

### F49 — no script materialises the `standard` dataset · M · CONFIRMED BY ELIMINATION

Checked all three candidates:

| Script | What it actually does | Can it emit `standard`? |
|---|---|---|
| `compute_metrics_from_results.py` | reads `results/{ds}/*.jsonl`, prints H@1/H@3 | no — produces no dataset |
| `apply_history_filters.py` | filters `sample["facts"]` / `sample["fired_rules"]` from retrieval metadata | no — every fact it sees was retrieved *by a rule* |
| `naive_history_metadata.py` | builds naive subject history (`--include_indirect` adds object-side facts, exactly the `standard` definition) then writes **statistics only** | no — emits `*_summary.json` and a per-quad metadata JSONL, never `history_facts/` or `test_answers/` |

So the *logic* exists in `naive_history_metadata.build_subject_index` +
`compute_history`, and the *plumbing* exists in `TLR.build_bs` (F41), but nothing
connects either to the dataset writers. Cheapest route: wire `retrieve_type`
through in `retrieve.py` and extend `prepare_bs` (`TLR.py:174`) from
`col_sub == sub` to `(col_sub == sub) | (col_obj == sub)`, matching the
`--include_indirect` semantics.

---

## 4. Divergences from the GenTKG baseline (`TLR_gentkg.py`)

| Aspect | GenTKG original | This repo |
|---|---|---|
| Inverse body anchoring | subject only (F2) | object anchor when `--inverse_body_object_match` |
| Rule lengths used | **length 1 only** (`if body_rel_len == 1`) | all lengths admitted by default, but only the last body relation is consulted. Since F45 a bank containing longer rules is rejected with `NotImplementedError` unless `--length_1_only` is passed (the flag sets the internal `rule_length_all` to False) |
| Fact cap | `num_facts = 50` hardcoded | `--num_facts`, default `None` |
| Early stop | always (`break` at ≥ 50 facts) | opt-in via `--early_stop_at_num_facts` |
| Object in prompt | `18.Thailand` (id + `.` + name) | ~~`Thailand`~~ → matched under `--index_target` (F53) |
| Timestamp in prompt | `int(time/period)` → `292` | ~~`int(time)/period` → `292.0`~~ → matched (F47) |
| Confidence | used only via bank ordering | explicit `conf`, `--top_k_rules`, `--confidence_threshold` |
| Rule provenance | none | `rule_id`, `found_by`, `common_rule_pool.json` |
| Metadata / re-filtering | none | `--save_metadata` + `apply_history_filters.py` |

Two of these matter for the paper's claims:

- Staying at length 1 **matches** GenTKG rather than diverging from it — useful
  if a reviewer questions why `ragtkgc_no_mining` is length-1 only.
- The prompt format differs (object encoding, timestamp formatting), so this
  repo's "gtkg" numbers are not directly comparable to GenTKG's published ones.
  Do not quote their numbers alongside these as if the prompts matched.

---

## 4b. Why GenTKG reports higher LLaMA numbers (F12b — RESOLVED)

Source: `github.com/mayhugotong/GenTKG` (shallow clone) and arXiv:2310.07793.

GenTKG's reported ICEWS14 result for `Llama2-7B + GenTKG` is
**H@1 36.85 ± 0.75 / H@3 47.95 ± 0.75 / H@10 53.5 ± 0.8**; ICEWS18
**24.25 / 37.25 / 42.1**. This repo's `llama-2-7B-icews14-gtkg` scores
H@1 .317 / H@3 .383. Six concrete differences account for the gap, in
descending order of confidence.

### 1. They report temporal-aware FILTERED Hits@k; this repo reports RAW

Paper, Table 1 caption: "Temporal link prediction results on **temporal-aware
filtered** Hits@1/3/10(%)", citing Gastinger et al. 2023.

Implementation — `evaler.py:1617`:

```python
if (answerlow != gtlow and answerlow in dict_qu_ans_lower) and filter_yes:
    print("Got another answer: " + answer + ", ignored.")
    k_inloop += 1
    filter_m_count -= 1          # candidate removed from the ranking
```

`dict_qu_ans` (`gen_set_ans`) maps each query `"{time}: [{sub}, {rel},"` to the
**set of all true objects** for it, built from the full test set
(`--fulltest`, `--time2id`). A candidate that is a different-but-also-correct
object does not push the ground truth down the ranking. `--FILTER` defaults to 1.

This repo's `utils.update_metric` does
`[x for x in predictions[:index] if x not in targets]`, but `targets` always has
exactly one element (`run_hf.py:326`: `[x['target']]`), so **nothing is ever
filtered**. Raw vs filtered is not a small correction on ICEWS, where many
(s, r, t) queries have several valid objects.

**This is the single largest and most certain factor. Any comparison of these
numbers against GenTKG's is invalid until both use the same protocol.**

### 2. Their answer format is the entity INDEX, not the name

`data_utils/create_json_train.py`:

```python
"target": str(entities[name_obj]) + '.' + name_obj      # e.g. "18.Thailand"
```

and history facts are rendered the same way (`TLR.py:174-177`:
`str(id_obj)+'.'+obj_in_word`). Their own ablation (paper Table 2) reports
gpt-3.5-turbo on ICEWS14 with TLR: **lexical 0.21 vs index 0.26 H@1** — index
wins, and they argue this relieves the data-leakage concern. This repo uses bare
lexical names.

### 3. Decoding is constrained to digit tokens, which is what makes H@10 cheap

`evaler.py:104-133` (`restrict_list_hard`):

```python
top_10_indices = torch.topk(logits_last, k=logits.shape[-1], dim=-1).indices
values_to_extract = [29900, 29896, 29906, 29941, 29946,
                     29945, 29953, 29955, 29947, 29929]   # LLaMA tokens '0'-'9'
mask = np.isin(top_10_indices_np, values_to_extract)
extracted_elements = top_10_indices_np[mask][:10]
next_token = top_10_indices[m]                             # m-th most likely digit
```

Because the target starts with the entity index, the first generated token must
be a digit. Masking the vocabulary to those 10 tokens turns the first decision
from ~32000-way into 10-way, and taking the m-th ranked digit for m = 0..9 then
decoding greedily yields 10 distinct candidates at roughly the cost of 10 greedy
decodes — far cheaper than 10-way beam search. `first_checking` keeps the
candidates distinct via `self.constraints`.

This directly answers the "beam search is too slow" constraint in this repo:
GenTKG did not use beam search either. The index answer format is what makes the
cheap alternative possible.

### 4. 50 epochs, not 1

`config.py`: `EPOCHS=50`, `BATCH_SIZE=128`, `MICRO_BATCH_SIZE=2`
(→ grad-accum 64), `LEARNING_RATE=3e-4`, `WARMUP_STEPS=100`.
`train_llama2.sh` passes only `--OUTPUT_DIR` and `--DATA_PATH`, so those
defaults are what ran.

This repo: `num_train_epochs=1`, batch 1 × grad-accum 8. Same 1024 samples,
~50× the training steps.

### 5. fp16 LoRA, not 4-bit QLoRA

`config.py` has `BIT_8` and `BIT_4` both defaulting to False, so training runs
unquantized. LoRA is `r=8, alpha=16, dropout=0.05,
target_modules=["q_proj","v_proj"]`, plain `bias`, no rslora. This repo uses
4-bit nf4 QLoRA with `bias='lora_only'` and `use_rslora=True`.

Label masking is equivalent in intent but much simpler on their side
(`utils.llama2_tokenizer`): `labels = [-100]*CONTEXT_LEN + target_ids + [eos] +
[-100]*pad`, no token-id patching (contrast F16).

### 6. An instruction IS part of their training prompt

`data_utils/basic.get_ins()` prepends `<s>[INST] <<SYS>>` plus an explicit task
description ("You must be able to correctly predict the next {object_label} …
You must generate {object_label}.{object}") and `create_json_train` appends
`[/INST]`. This repo deliberately omits any system prompt (paper §4.1) for fair
PLM/LLM comparison — a defensible choice, but it removes something GenTKG
trained with.

Note their inference default is `--instruct_yes 0`, i.e. **no** instruction at
test time (`evaler.py:1550`), which is a train/test mismatch on their side.

### Unverified discrepancy worth flagging

`config.py` sets `CONTEXT_LEN=256` and `llama2_tokenizer` calls
`tokenizer(context, max_length=256, truncation=True)`. HF truncation defaults to
`truncation_side="right"`, which would cut the tail of the prompt — including the
query line — while a ~50-fact history is far longer than 256 tokens. Meanwhile
`eval_utils.parse_args` defaults `--CONTEXT_LEN 4096` at inference. Either the
paper runs used a larger value than the committed default, or the tokenizer's
truncation side differs in their environment. **Do not assume the committed
defaults reproduce the published numbers.**

### Consequences for this paper

- Reporting raw Hits@k alongside GenTKG's filtered Hits@k is not a valid
  comparison. Either implement temporal-aware filtering (the data needed is
  already available: full test set + `ts2id`) and report filtered, or state
  explicitly that these are raw numbers and do not table them against GenTKG's.
  Implementing filtering is the better option and needs no GPU re-run — it is a
  metric recomputation over existing `results/*.jsonl`.
- The index answer format is cheap to test (a retrieval re-run plus fine-tuning)
  and is supported by their own ablation. It also unlocks digit-constrained
  decoding, which would give real H@3/H@10 without beam search (F17).
- The 1-epoch vs 50-epoch difference is the largest training-side gap and is
  cheap to close at 1024 samples.

## 4c. The numeric object label — root cause of the metric gap

Source: `github.com/usc-isi-i2/isi-tkg-icl` (shallow clone) and arXiv:2305.10613,
the second repo this work builds on (`permute`, `parse_results`, `HitsMetric`,
`run_hf.py` structure all come from here).

### Their design

ISI §3.2: *"obtaining scores for entities based on these probabilities is
challenging as they may be composed of **several tokens**. To address this
challenge, we utilize a mapped numerical label as an indirect logit to estimate
their probabilities."*

Their prompt template (paper Table 2) carries the label in **both** modes:

```
Lexical:  2000: [Superbowl, Champion, 0. St Louis]
Index:    2000: [0, 0, 0. 0]
```

GenTKG uses the same idea (`18.Thailand`). This repo dropped the prefix and
emits bare lexical names.

### Why that made `permute` unusable

`isi/utils.py` defaults: `--max_length 1`, `--dec_cand 5`, `--top_k 100`.
`max_length=1` means `max_new_tokens=1` — a single decoded token, so
`permute` at `max_step=1` explores `dec_cand¹ = 5` leaves. Linear cost.

In label mode they bypass `permute` altogether (`isi/model_utils.py:70-80`):

```python
if args.label and "llama" not in args.model:
    probs = outputs.scores[0]                       # ONE forward pass
    probs_indices = torch.argsort(probs, dim=-1, descending=True)
    for tok in probs_indices[0][: args.top_k]:      # top 100 candidates
```

A full ranking over the vocabulary from one forward pass → H@1/3/10 for free.

This repo kept `permute` but replaced the single-token label with a multi-token
lexical name, so `max_step` became `maxl` (42 for Flan-T5/ICEWS14). At that
depth `dec_cand=5` is 5^42, which forces `dec_cand=1` — hence exactly one
candidate (F17). **`permute` was never designed for multi-token targets.**

### Both baselines report time-aware filtered

ISI §4.2: *"In this paper, we present performance with the time-aware filter."*
GenTKG Table 1 caption: *"temporal-aware filtered Hits@1/3/10(%)"*. Both cite
Gastinger et al. Measured on this repo's `icews14` test split: 577 queries have
more than one gold object, covering 1292 rows = **17.5%** of the test set, so the
raw/filtered distinction is material.

### One cause, three symptoms

| symptom | cause |
|---|---|
| H@1 ≡ H@3 ≡ H@10 for Flan-T5 (F17) | lexical multi-token target → `permute` unusable → `dec_cand=1` |
| raw metrics not comparable to either baseline (F12b) | a single candidate cannot be filtered — there is no ranking |
| `cs` introduced as a graded metric (F19) | plausibly a substitute for the ranking signal that was lost |

### ISI's labels are NOT global entity ids — two important caveats

`isi/utils.py:344-360`:

```python
for x in quadruples[-history_len:]:
    candidates_stats[x[2]] += 1                       # objects in the retrieved history
candidates_stats_sorted = sorted(..., reverse=True)   # by FREQUENCY, descending
for i, (entity, _) in enumerate(candidates_stats_sorted):
    candidates_mapping[entity] = i                    # labels 0, 1, 2, ...
```

1. **Closed candidate set.** Labels cover only entities appearing in that
   prompt's retrieved history — not the full entity vocabulary. So their Hits@k
   is a ranking over a shortlist of tens of candidates, and the ceiling is
   retrieval recall. Not the same quantity as Hits@k over all entities.
2. **Frequency-ordered labels leak the prior.** Label `0` is the most frequent
   object in the history — which is itself a strong TKG forecasting baseline
   (their own `--model frequency`). A model that only learns "emit 0" inherits
   it for free.

Also, ISI's formulation makes string-level errors impossible: the model emits a
digit, so `Japan` vs `Japan_(Government)` cannot occur. GenTKG still requires the
full name to be generated and compared, so it remains accountable for the string.

### Strictness hierarchy

| | candidate space | must produce | comparison | metrics |
|---|---|---|---|---|
| ISI | closed shortlist from history, frequency-ordered | one digit | label → entity lookup | filtered |
| GenTKG | all entities (global `entity2id` id) | `id.Name` in full | name, case-insensitive | filtered |
| this repo | all entities | bare name | exact match | raw, 1 candidate |

This repo performed the strictest variant and reported it against numbers
produced by the two laxer ones. Group C closed that gap against GenTKG: the
candidate space is still all entities, but the produced form is `id.Name` under
`--index_target`, the comparison is case-insensitive on the name, and F67 made
the metrics a real ranking. ISI's shortlist candidate space remains a
methodological difference to disclose, not a defect to fix.

### Recommendation

Adopt **GenTKG's** format — `id.Name` with **global** entity ids, generate the
full string, compare the name. Not ISI's per-prompt frequency-ordered labels.
This gives:

- cheap multi-candidate decoding (digit-constrained first token) → real H@3/H@10
- a ranking that time-aware filtering can operate on
- no frequency-prior leak, since the global id is arbitrary
- the model still accountable for producing the correct entity string

Supported by GenTKG's own ablation (their Table 2: lexical 0.21 vs index 0.26
H@1). Highest-priority change for the re-run.

When tabling against ISI-derived numbers, note explicitly that their candidate
space is a history-derived shortlist. This is a legitimate methodological point
for the paper, not merely a fix.

## 5. Variant glossary

Folder suffixes are auto-generated by `retrieve.py:224-230` **only**;
`apply_history_filters.py --output_dir` and `create_json_train.py --dataset` are
free text, so any other name component is a hand-typed label.

| Token | Meaning |
|---|---|
| `_inverse_included` | `--inverse_body_object_match` |
| `_num_facts` | **old** code: history capped at 50, all rules consulted (i.e. `--early_stop_at_num_facts` absent) |
| `_n50` | **new** code: `--num_facts 50` |
| `_early_stop` | new code: `--early_stop_at_num_facts` |
| `_top10rules` | `--top_k_rules 10` |
| `_thresh0.5_t5` | `--confidence_threshold 0.5 --model_type t5` |
| `_idn` (folder) | `--index_target` — objects rendered `id.Name`; subjects, relations and the answers file stay names |
| `test_ids` (inside a JSON filename) | nothing; naming convention |
| `full` (model name) | trained on the full train split, not the 16-sample subset |
| `tail_truncate` / `no_tail_truncate` (results name) | `--tail_truncate_long_inputs` at test time; suffix only exists after commit `19ac0ad` |
| `fancy` / `fancy_half` (model name) | hand label for the newer training pipeline; `half` = 6 → 3 epochs |

`num_train_epochs=3` is now hardcoded (`training_T5.py:129`), so every future
run is a "half" run and the label no longer discriminates. Drop
`fancy`/`fancy_half` from model names for the paper and report epochs in the
setup section instead.

---

## 6. Decisions

Taken:

- Length-1 rules only, for now.
- Re-run everything (both datasets) under one configuration, then pick the single
  best variant per model.
- `raw` and `standard` baselines must be rebuilt under the same training regime
  (they are affected by F3).
- GPT-4.1 RAG applied only to the overall best model, not three variants per model.
- Inverse anchoring: the new behaviour is the intended one.

Open:

- F1 `period`: fix to 1 and re-mine `ragtkgc`, or keep and document? Expect
  fewer rules and possibly faster runs once walks fail honestly.
- LLaMA2-7B stays in the paper (or a comparable LLM substitutes for it). The
  1024 cap stays too (F12). Open: whether `training_LLaMA.py` gains
  `--tail_truncate_long_inputs` / `--eval_file_path` (F15).
- **Object label format (§4c, HIGHEST PRIORITY):** restore the `id.Name` prefix
  used by *both* baselines. Single decision point per answer → full entity
  ranking from one forward pass → real H@1/3/10 and a rankable list, at lower
  compute cost than now. Requires a retrieval re-run + retraining.
- **Metric protocol (§4b/§4c):** implement time-aware filtered Hits@k and report
  filtered, or state plainly that these are raw and stop tabling them against
  GenTKG/ISI. Only meaningful once there is a ranking to filter (see above).
- **Epochs for LLaMA (F12b):** 1 → closer to GenTKG's 50, at 1024 samples.
- Epoch count for Flan-T5: keep 3 (controller inert, F13) or raise so the
  controller can act?
- Decoding (F17/F18): settled — top-1 for Flan-T5 (beam search too slow), so
  report H@1 only for it. Open: how to state H@3 comparability against GenTKG.
- Metrics (F19/F20): drop CS, or replace with an embedding cosine? Recompute
  BERTScore with `rescale_with_baseline=True`?
- RAG (F24/F25): if the RAG experiment is repeated for the single best model,
  the history source must match that model's own history modeling rather than
  being hardcoded to ragtkgc, and the instruction text must either be applied to
  all samples or to none. Otherwise the delta is not attributable to RAG.
- RAG framing: report the selected subset as an explicitly hard subset with its
  own before/after numbers, rather than as a full-test-set delta.
- Re-time all miners on this machine, or keep the published gtkg/ragtkgc timings
  and footnote the hardware difference?
- Rule-count convention for Table 3.
- Whether to keep the old LLaMA / ICEWS18 rows as published while adding
  `ragtkgc_no_mining` only where measured.
- `rag_with_gpt_4_1.py` defaults to the ICEWS18 `ragtkgc` bank; needs updating if
  banks are re-mined.

---

## 7. Recommended target configuration

Mining: `-l 1 -n 200 -s 1 -m <alg>`, `period` resolved per F1.

Retrieval: `--inverse_body_object_match --num_facts 50 --save_metadata`, no
`--top_k_rules`, no `--use_ids`. `--save_metadata` on the base run allows any
later filter combination without re-retrieving.

Training: `--eval_file_path` set (activates the adaptive controller and
best-checkpoint selection) and `--tail_truncate_long_inputs` **on**, so all
74845 train samples are used (F3).

Testing: `--tail_truncate_long_inputs` matched to training.

Back up `data/processed_new/{ds}/output/{ds}/common_rule_pool.json` before any
re-mining — `save_rules` mutates it in place and it is untracked.

### F50 — nothing validates context/answer alignment · H · LATENT

`create_json_train.convert_txt_to_json` drives its loop from the answers file and
indexes into the contexts:

```python
for i in range(len(test_ans)):
    data = {"context": inputs[i], "target": test_ans[i][2]}
```

`inputs` comes from `content.split('\n\n')`. Nothing checks the two lengths.
Measured across six variants on disk:

| Variant | chunks | answers | rule_ids | json records |
|---|---|---|---|---|
| gtkg/test | 7,372 | 7,371 | 7,371 | 7,371 |
| gtkg/test_inverse_included_num_facts | 7,372 | 7,371 | 7,371 | 7,371 |
| gtkg/train_inverse_included_num_facts | 74,846 | 74,845 | 74,845 | 74,845 |
| ragtkgc/test_inverse_included_num_facts | 7,372 | 7,371 | 7,371 | 7,371 |
| no_mining/test_..._50_top10rules | 7,372 | 7,371 | 7,371 | 7,371 |
| no_mining/train_..._50_top10rules | 74,846 | 74,845 | 74,845 | 74,845 |

Always exactly **one extra chunk**, always the last, always empty (the file ends
`\n\n`), and there are no empty chunks anywhere else. Alignment is therefore
**correct today** — but only because the loop is driven by the shorter list.

The failure modes are silent or late:

- `inputs` shorter than `test_ans` → `IndexError` at some arbitrary row.
- a lost chunk boundary anywhere → every subsequent `(context, target)` pair is
  shifted by one, with no error and plausible-looking training data.
- `rule_ids` shorter → `rule_ids[i] if i < len(rule_ids) else []` pads with empty
  lists instead of failing.

One assertion (`len(inputs) - 1 == len(test_ans) == len(rule_ids)`) removes the
whole class. Worth adding because a silent one-row shift would be invisible in
metrics and would look like a modelling result.

### F51 — `random.sample` is unseeded and can raise · M

`create_json_train.py:49`. The `_16.json` subset differs on every invocation, and
`num > len(data_list)` raises `ValueError` rather than clamping.

### F52 — `get_ins()` is called and thrown away · M

`convert_txt_to_json` line 8: `ins = get_ins()`, never referenced again. In GenTKG
this returns the instruction block that gets prepended to every training example.
Here the omission is deliberate (paper §4.1, no system prompt, for fair PLM/LLM
comparison) but the dead call reads as though instructions are included. Relevant
to §4b item 6: GenTKG *trains* with an instruction.

### F53 — two arguments of `create_json_train.py` do nothing · L

**RESOLVED in Group C.** `use_ids` was accepted and never read; `entities` was
loaded from `--dir_of_entities2id` and never used. `entities` turned out to be
the stump left when the `id.Name` rendering was removed, not a vestigial
parameter: it is now wired up for `--index_target`. `use_ids` is deleted.

### F54 — F11 refined: only the `_ids` variants have blank answer lines · L

Measured: the non-`_ids` answers files have 7,371 raw lines and 7,371 non-blank
lines — no padding at all. The 50%-blank behaviour is specific to the `_ids`
retrieval path, and `convert_txt_to_json` filters blanks anyway (`if x.strip()`).

### F55 — the disk cost of "retrieve once, filter many" · M · PRACTICAL

Measured for `ragtkgc_no_mining` on ICEWS14 (one mining algorithm, one dataset):

| Artifact | test | train | valid | total |
|---|---|---|---|---|
| `metadata/` | 1.1 G | 4.6 G | 1.2 G | **6.9 G** |
| uncapped `history_facts/` | 765 M | 3.3 G | 871 M | **4.9 G** |
| filtered `history_facts/` (n50, top10) | 39 M | 314 M | 46 M | **399 M** |

`data/processed_new/icews14/` totals **20 G**. So ~11.8 G of scratch produces
399 M of actually-used dataset.

**`_idx_fine_tune_all.txt` is a byte-identical duplicate with no readers.**
`retrieve.py:284-287` writes the same content twice — `write_txt(path_idx,
test_text)` joins each one-element row (so the element itself) and appends `\n`,
while `path_txt` writes `test_text[i][0] + '\n'`. Verified on
`gtkg/test_inverse_included_num_facts`: both files are 20,469,610 bytes and `cmp`
reports them identical. A repo-wide grep for `idx_fine_tune` finds only the four
lines that *write* it — nothing reads it, in this repo or downstream.

| File in `history_facts/` | Read by | Needed |
|---|---|---|
| `history_facts_{ds}.txt` | `create_json_train.py --dir_of_trainset` | yes |
| `history_facts_{ds}_rule_ids.txt` | `create_json_train.py`, auto-inferred by `.replace('.txt', '_rule_ids.txt')` | yes |
| `history_facts_{ds}_idx_fine_tune_all.txt` | **nothing** | **no — delete** |

Dropping it halves every `history_facts/` directory. Note the related trap: the
auto-inference at `create_json_train.py:31` only finds the rule-ids file when
`--dir_of_trainset` is the plain `.txt`; passing the `_idx_fine_tune_all` variant
yields a non-existent path and provenance is silently dropped.

Three further reductions, all safe:

1. **The uncapped `history_facts/` (4.9 G) is redundant when `--save_metadata` is
   on.** Nothing trains on it, and `apply_history_filters.py` with no filters
   regenerates it exactly. Add a flag to skip writing it.
2. **The metadata itself is inflated by two redundant fields.** Each fact entry
   stores `"text"` (reconstructible from `all_facts[idx]` via `_fact_to_line`) and
   a list of full 40-character SHA-1 `rule_id`s (a fact matched by five rules
   carries ~210 characters of hex). Dropping `text` and interning rule ids into a
   per-file table would cut this several-fold.

### F56 — the filter logic exists twice · M

`apply_history_filters._apply_filters` re-implements `TLR.build_tl`'s filtering
against the metadata instead of the graph. Verified equivalent on every point
that affects output: both treat metadata facts as newest-first, both apply
`top_k_rules` before `num_facts`, both reverse to oldest-first when rendering
(`TLR._indices_to_history_lines` vs `to_lines`), both handle the empty case the
same way, and both resolve `model_type` to `t5` by default. Sanity check on
sizes: 271,967 lines / 7,371 samples = 36.9 lines per sample under the n50 cap
(≈34.9 facts + query + separator), against 833.5 uncapped — consistent with the
measured fact counts.

Correct today, but it is two hand-synchronised implementations of one rule. Any
future change to trimming or rendering must be made in both.

### F57 — `training_LLaMA.py` never saves the trained adapter · H · **FIXED 2026-09-03**

The script ends at `trainer.train()` (line 197). There is no `save_model`, no
`save_pretrained`, and `trained_model_name` is assigned at line 120 and **never
referenced again** — verified by grep, the only three hits are the argparse
declaration, that assignment, and an unrelated comment.

Nor does the Trainer save one implicitly. `TrainingArguments` here sets no
`save_strategy`, so it defaults to `"steps"` with `save_steps=500`. The run is
1024 samples at batch 1 × grad-accum 8 = **128 optimiser steps**, so step 500 is
never reached and no checkpoint directory is written either.

Whatever produced the existing `models/llama-2-7B-icews14-*` adapters, it was not
this script as committed.

Fixed by adding `import os` and a save block after `trainer.train()`, mirroring
`training_T5.py:209-213`:

```python
save_path = os.path.join(args.output_dir, trained_model_name.replace("'", "").replace('"', ''))
trainer.save_model(save_path)          # PEFT-aware: writes adapter_config.json + adapter_model.safetensors
tokenizer.save_pretrained(save_path)
trainer.state.save_to_json(os.path.join(save_path, "trainer_state.json"))
```

No `tie_weights()` call — that is specific to T5's shared encoder/decoder
embeddings. Path convention verified against the loader: the save target is
`./models/{trained_model_name}` and `run_hf.py:150-152` reads
`./models/{finetuned_model}`, so passing the same string to both works.

Two related changes applied at the same time:

- `save_strategy='no'` set explicitly. This preserves the existing behaviour (the
  implicit `"steps"`/500 default never fired at ~128 steps) and documents it.
  **Carries a forward note in the code: it must become `"epoch"` and match
  `eval_strategy` if F15 adds an eval set with `load_best_model_at_end`,
  otherwise there are no checkpoints for the best model to be restored from.**
- A fail-fast guard on `--trained_model_name` in **both** training scripts,
  placed immediately after `args = parser()` and well before any model is
  loaded. Previously omitting it raised `AttributeError` on `None.replace(...)`
  *after* training had finished.

**Which model is saved:** the last one. LLaMA has no eval set (`eval_dataset` is
commented out at `:193`), no `load_best_model_at_end`, and
`num_train_epochs = 1`, so there is exactly one end state — "last" and "best"
coincide. This becomes a genuine choice only when F15 lands, at which point
`load_best_model_at_end`, `metric_for_best_model`, `eval_strategy` and
`save_strategy` all have to be set together. Contrast `training_T5.py:137`, where
`load_best_model_at_end=_has_eval` already restores the best epoch (F14).

### F58 — CORRECTED: the forced EOS label is right at batch size 1 · L · LATENT ONLY

`training_LLaMA.py:111`, inside the per-example loop:

```python
labels[i, -1] = 2
```

`DataCollatorForLanguageModeling` pads to the batch maximum, and because
`tokenizer.pad_token = tokenizer.eos_token` (line 161) every EOS — including the
intended final one — is masked to `-100` by the base collator. Line 111 exists to
put it back. But index `-1` is the last position **of the padded row**, not the
end of that example's real sequence. For any example shorter than the batch
maximum (with default right padding, most of them) this supervises a padding
position as EOS.

**But `per_device_train_batch_size=1` (line 135), so there is never any padding.**
`tokenizer.pad` on a single example pads to that example's own length, and
`pad_to_multiple_of` is not set. Index `-1` is therefore the true final token, and
line 111 does exactly what it intends. `auto_find_batch_size=True` only *halves*
the batch on OOM and cannot go below 1, so the batch size never rises above 1
either.

Conclusion: **correct as configured.** The earlier severity-H reading in this file
was wrong — it assumed multi-example batches.

The residual risk is only that the correctness is coupled to a batch size of 1 in
a non-obvious way. In practice the batch size is pinned there by the Colab VRAM
budget reported in the paper, so this is unlikely ever to fire. Action: a comment
at line 111 recording the dependency. If the batch size is ever raised, switch to
`batch["attention_mask"][i].sum() - 1` at the same time.

### F59 — the 1024 subsample degenerates to "first 1024" on smaller files · M

`training_LLaMA.py:186-188`:

```python
step = round(length_ti / 1024)
tokenized_input = tokenized_input.select(range(0, length_ti, step)[:1024])
```

For `length_ti` in **[1024, 1536)** the ratio rounds to 1, so the selection is
`range(0, n, 1)[:1024]` — the first 1024 rows contiguously. Since the train split
is chronologically ordered, that is the earliest month or so, not a spread of the
year. At ICEWS14's 74,845 the step is 73 and the spread is correct; this is latent
rather than active, but it fires silently on any smaller training file.

### F60 — the supervised span differs between datasets · H

`training_LLaMA.py:105-108`:

```python
if args.dataset == 'icews14':
    labels[i, len(response_token_ids)+response_token_ids_start_idx+1:] = -100
else:
    labels[i, len(response_token_ids)+response_token_ids_start_idx:] = -100
```

An unexplained `+1` for ICEWS14 means one extra token is supervised on that
dataset and not on others. Confirms F16 item 2. Consequence for the paper: the
ICEWS14 and ICEWS18 LLaMA models are trained against subtly different objectives,
so cross-dataset comparison of those two rows is not clean. Either justify the
`+1` or remove it and retrain both.

Also confirms F16 item 3: `np.where(...)[0][-1]` at line 74 raises `IndexError`
on an empty match **before** the `if response_token_ids_start_idx is None` check
below it, so the `RuntimeError` at line 79 is unreachable.

**Needs 1-to-1 verification before acting.** The trace above is derived from the
code, not measured — the LLaMA tokenizer is not cached on this machine, so the
exact token boundaries are unconfirmed. Verify by loading
`AutoTokenizer.from_pretrained('TheBloke/Llama-2-7B-fp16')`, running one real
training row through `process_function` and `DataCollatorForCompletionLM`, and
printing the `(token, label)` pairs for both the `icews14` and the else branch.
The `+1` may be compensating for the merged-token cases patched at lines 96-103
(`)]`, `))]`, `.]` folding the bracket into the target's last token), in which
case removing it would break those rows. Do not change the slice until the
printout shows what is actually supervised.

### F61 — Flan-T5 tokenises the whole dataset twice, with the slow tokenizer · M

`training_T5.py:165` tokenises every context to compute the length-stats printout,
then `:172` tokenises everything again inside `dataset.map`. And `:153` uses
`T5Tokenizer`, not `T5TokenizerFast` — measured earlier in this investigation to
be roughly an order of magnitude slower. Two easy wins: switch to the fast
tokenizer, and derive the stats from `input_lengths` computed inside `map` (or on
a sample) instead of a separate full pass.

### F62 — `process_function` reads module-level globals · M

`TOKEN_LIMIT` and `TAIL_TRUNCATE_LONG_INPUTS` are module globals mutated at
`:154-155` and read inside `process_function`. `datasets.map` caches results
keyed by a fingerprint of the function; whether a changed global invalidates that
fingerprint depends on how `dill` captures referenced globals in the installed
version. Passing them explicitly via `fn_kwargs=` makes the dependency visible
and the cache key correct by construction. Worth doing before any run that
toggles `--tail_truncate_long_inputs`, so a stale cache cannot silently supply the
other variant's tokenisation.

### F63 — every T5 run already uses the same seed, so the spread is not seed noise · M · IMPORTANT

`Seq2SeqTrainingArguments` is constructed without `seed`, so HuggingFace's default
`seed=42` applies to **every** run. The observed spread between a `.369` run and
a `~.365` rerun therefore cannot be seed variance — the seed was identical. The
remaining sources are:

- bf16 / cuDNN non-deterministic reductions on GPU,
- `load_best_model_at_end` selecting a different epoch when two epochs' eval
  losses are close,
- `device_map='auto'` placement.

Two consequences. First, that ±0.004 band is irreducible run-to-run noise at
fixed seed, and it is the same size as the gtkg-vs-exhaustive gap — so a single
number cannot support a claim either way. Second, **reporting mean ± std over
seeds requires adding a `--seed` argument first**; re-running today just re-rolls
hardware noise at seed 42.

### F64 — the learning-rate comment describes the wrong training regime · L

`training_T5.py:127`: *"higher than full fine-tuning; fine for adapter-scale
updates"*. This path is **full** fine-tuning — `T5ForConditionalGeneration` with
no PEFT wrapper. The rate may still be fine for a 77 M model, but the stated
justification is for a regime this script does not use.

### F65 — `auto_find_batch_size` cannot do anything at batch size 1 · L

`training_LLaMA.py:134-135` sets `auto_find_batch_size=True` alongside
`per_device_train_batch_size=1`. The mechanism halves the batch size on OOM and
cannot go below 1, so it is inert.

### F66 — `weight_decay` is silently disabled for LLaMA only · L

Commented out at `training_LLaMA.py:142` while `training_T5.py:128` uses 0.1. An
undocumented divergence between the two training regimes; add it to the
regime-comparison table in the setup section (F12).

### F13 — controller inertness re-verified

`_marginal_improvement_negligible` returns `False` whenever
`len(self._eval_history) < 4` (`training_controller.py:307`), and `STOP`
additionally requires `_lr_reduction_count >= max_lr_reductions` (2). At three
epochs there are three evaluation points, so neither condition can be reached and
`STOP` is unreachable. `REDUCE_LR` can fire at most once, on the final
evaluation, with zero training steps remaining. The callback therefore never
alters training; the real effect of `--eval_file_path` is `load_best_model_at_end`.

### F67 — `permute` is unsound for multi-token targets, not merely expensive · H · AFFECTS THE INDEX PLAN

Two separate points, both from `model_utils.py:9-29` and `:71-95`.

**At `dec_cand=1` it is dead weight.** `model.generate` is called with no
`num_beams` and no sampling, i.e. greedy. `outputs.scores` is therefore the
per-step log-probability tensor *along the greedy path*. `permute` with
`dec_cand=1` follows `argsort(...)[0][:1]` at every step — the greedy choice —
so it reconstructs exactly the sequence `generate` already returned, then reports
its mean log-prob. `parse_results` softmaxes a single value to 1.0. The whole
recursion could be replaced by reading `outputs.sequences`.

**At `dec_cand>1` it is wrong.** All the logits come from one greedy forward
pass. If `permute` selects a non-greedy token at step *t*, the logits it then
uses at step *t+1* are still those conditioned on the **greedy** token at step
*t* — the model was never run on the alternative prefix. So every non-greedy
branch is stitched together from a path the model did not take, and its score is
not the probability of that sequence.

This is why ISI ran `--max_length 1`: with a single decoded token there is no
prefix to condition on, so the top-k of that one distribution are all valid. The
technique is sound for one token and unsound for many. Raising `dec_cand` here
would not buy real H@3 — it would fabricate candidates.

Consequence for the planned `id.Name` change: it is sound to take top-k over the
**first** token only (the digit), which is exactly GenTKG's
`restrict_list_hard`, and then decode each branch greedily with a real forward
pass per branch. It is not sound to keep `permute` and raise `dec_cand`.

### F68 — `get_filename` discards the variant, so results can silently overwrite · H

`utils.py:84-92`:

```python
dataset_path.split('/')[-1].split('.')[0]
```

Only the JSON basename survives. For
`--dataset_path exhaustive/test_inverse_included_n50/history_modeling_test/icews14_test.json`
the output is `{model}_icews14_test_{tail_truncate}.jsonl` — the variant folder is
gone. Two different retrieval variants evaluated with the same model write to the
**same** file, the second overwriting the first with no warning.

The existing results survive this only because the variant was hand-typed into
the JSON filename via `create_json_train.py --dataset` (e.g.
`icews14_gtkg_test_ids_inverse_included_num_facts.json`). So that naming
convention is load-bearing after all — the `test_ids` token within it is
meaningless (F54), but the rest of the string is the only record of which
retrieval variant produced the file. Fix: include the variant directory in the
filename, or record it inside the JSONL.

### F69 — the pre-scan is a redundant full pass · M

`run_hf.py:247-251` iterates the whole test set and tokenises every context to
log length statistics. But the prediction loop already accumulates
`overall_len_stats` at `:286-287`, **before** any truncation is applied
(`:294-302`), so the runtime `overall` block reports exactly the same numbers.
The pre-scan is fully redundant and costs one extra tokenisation of every sample
before the first prediction is made.

Counting the pass inside `predict` (`model_utils.py:60`), each sample is
tokenised three times per run.

### F70 — tail truncation round-trips through text · M

`run_hf.py:296-302` tokenises, keeps the last `token_limit` ids, **decodes back
to a string**, and hands the string to `predict`, which tokenises it again.
SentencePiece decode→encode is not guaranteed to be length-preserving, so the
final input may not be exactly at the limit. Passing `input_ids` straight to
`predict` would be exact, at the cost of changing its signature. Worth measuring
the post-truncation length distribution before assuming the cap holds.

### F71 — the `rank = 5` sentinel will silently break H@10 · M · **FIXED 2026-09-03**

`compute_metrics_from_results.py:44` initialises `rank = 5` for the
target-not-found case, then only tests `rank == 1` and `rank <= 3`, so a miss
scores nothing today. Adding the obvious `if rank <= 10: h10 += 1` would count
**every miss** as a hit@10.

Changed to `float('inf')`, which fails every `rank <= k` test by construction, so
no future cutoff can be added unsafely. `float('inf')` was chosen over `None`
because it keeps all the existing comparisons working rather than raising.

**Behaviour-neutral, verified.** Re-ran `--file_name all` over all 52 result
files before and after: output byte-identical.

**The trap, demonstrated** on
`flan-t5-small-icews14-ragtkgc_no_mining-..._50_top10rules_..._tail_truncate`
(7,371 rows, 1 candidate each) by adding `if rank <= 10: h10 += 1` under each
sentinel:

| Sentinel | H@1 | H@3 | H@10 |
|---|---|---|---|
| old, `rank = 5` | 0.363 | 0.363 | **1.000** |
| new, `rank = inf` | 0.363 | 0.363 | 0.363 |

Every miss passed `5 <= 10`, so H@10 came out at a perfect 1.000. The same would
have happened for LLaMA's three-candidate files.

### F72 — token-limit inference can refuse to start · L · LATENT

`get_token_limit` (`run_hf.py:27-33`, and the twin in `training_T5.py:46-52`)
rejects any value above 1,000,000 and falls back to `model_max_length`. Some
LLaMA tokenizer configs ship `model_max_length` as the sentinel
`1000000000000000019884624838656`, in which case both candidates are rejected and
the function raises `ValueError` before the run begins. It evidently resolves to
a real value in the environment used so far; worth an explicit per-model default
rather than relying on that.

### F73 — partial result files are scored without complaint · L

`run_hf.py` opens the output with `"w"` and appends per sample, with no
completion marker. `compute_metrics_from_results.py` divides by the number of
lines it actually read, so a crashed or interrupted run yields a plausible-looking
metric over a subset — exactly the o4-mini case (F22, 1,291 of 7,371 rows scored
as `.259`). Fix: refuse to score, or loudly flag, any file whose line count is
below the split size.

### F74 — T5's pad token is overwritten with EOS · L

`run_hf.py:130` sets `tokenizer.pad_token = tokenizer.eos_token` for Flan-T5,
which already has a real `<pad>` (id 0). Harmless while generation is
single-prompt and nothing pads, but it would corrupt any batched implementation —
worth removing at the same time as batching.

### F75 — `logging.basicConfig` silently keeps writing to the first run's log · H · **FIXED 2026-09-03**

`_setup_logging` (`run_hf.py:92-101`, and the twins in `retrieve.py` and
`apply_history_filters.py`) calls `logging.basicConfig(...)`. That function is a
**no-op when the root logger already has handlers** and `force=True` is not
passed. In a fresh process it works; in a notebook session that evaluates more
than one configuration, every run after the first keeps logging to the *first*
run's file.

Measured evidence — exactly one log on disk contains two runs:

```
logs/gpt-4o-mini-2024-07-18_..._tail_truncate.log
  2026-05-20 15:43:50  Final metrics: {... hit1: 0.2091 ...}
  2026-05-20 16:15:27  OpenAI model: o4-mini-2025-04-16      <- different model, same log
  2026-05-20 16:15:27  Token limit : 128000
```

So a log filename does not reliably identify the run inside it. This is why the
test-time flags for the `.369` and `.363` runs could not be recovered from
artifacts, and it compounds F68 (the results filename drops the variant) and the
fact that `write_results` never stores the prompt.

Fixed by adding `force=True` to all three `_setup_logging` helpers (`run_hf.py`,
`data_utils/retrieve.py`, `data_utils/apply_history_filters.py`), each with a
comment explaining why it is load-bearing rather than cosmetic.

Verified by simulating two runs in one interpreter session:

| | `runA.log` | `runB.log` |
|---|---|---|
| without `force` | run A's line **and run B's line** | **empty** |
| with `force=True` | run A's line | run B's line |

That is exactly the on-disk symptom: the gpt-4o-mini log holds two runs while its
companion results file is absent, and the o4-mini results file exists with its
metrics recorded in the other log.

Note that `FileHandler` still defaults to append mode, so re-running an identical
configuration adds to the same file rather than replacing it. That is desirable —
timestamps separate the runs — but it means a log may legitimately contain more
than one run of the *same* config.

**Still open, and the remaining half of the provenance problem:** `run_hf.py`
logs the base model, token limit and output path, but never the parsed arguments.
`retrieve.py:105-109` and `apply_history_filters.py:284-287` both dump their
parameters; `run_hf.py` does not. One line — `logger.info("Args: %s",
json.dumps(vars(args), indent=2))` — would make every future evaluation
self-describing, which is precisely what was missing for the `.369` run.

Related: that log's companion results file
(`results/icews14/gpt-4o-mini-2024-07-18_....jsonl`) does not exist on disk, so a
log survives with no results and a results file (the o4-mini one, F22) survives
with its metrics logged elsewhere.

### F76 — an OpenAI results file can be named after a model that did not run · M

`run_hf.py:209`: `model_name_for_file = args.finetuned_model or openai_model_name`.
The model actually called comes from `os.environ.get('OPENAI_MODEL')` (`:165`).
If `--finetuned_model` is passed on the OpenAI path, the output file and log are
named after it while a different model produced the predictions.

### F77 — the offline scorer agrees with the live metric · VERIFIED OK

Cross-checked on
`flan-t5-small-icews14-ragtkgc_no_mining-full-..._50_top10rules_..._tail_truncate`:

| Source | H@1 |
|---|---|
| logged by `run_hf.py` at run time | 0.3626373626 |
| `compute_metrics_from_results.py` | 0.363 |

Same file, same value. So recomputing metrics from `results/*.jsonl` is
trustworthy, which is what makes the filtered-Hits@k plan viable without any GPU
re-run.

### F78 — `eval()` instead of `json.loads()` · M

`compute_metrics_from_results.py:39` parses each line with `eval`. The files are
self-produced so this is not a live security problem, but it executes arbitrary
code, is markedly slower than `json.loads`, and turns a malformed line into a
`SyntaxError` rather than a clear JSON error.

### F79 — the scorer hardcodes split sizes and silently accepts short files · M

`compute_metrics_from_results.py:30`: `limit = 10000 if dataset == 'icews18' else 7371`.
Two problems. The split size is hardcoded rather than read from the data, so it
must be edited for any new dataset. And because the loop takes `lines[:limit]` and
divides by the number of rows actually seen, a file **shorter** than the split is
scored over whatever it contains, with no warning — the F22 o4-mini case
(1,291 rows scored as `.259`). See also F73.

### F80 — `bertscore.py` details · L

- `load("bertscore")` is called twice (lines 19 and 24); the first result is
  discarded.
- No `rescale_with_baseline=True`, so values sit on the raw similarity scale
  (F20).
- `predictions[0]` / `targets[0]` only, so for LLaMA's three candidates just the
  first is scored — the metric is top-1 regardless of model.
- No first-N cap, unlike the Hits@k scorer. Harmless today because result files
  are exactly 7,371 / 10,000 rows, but the two scripts would disagree on a
  partial file.
- `--results_file` must already include the dataset subdirectory; there is no
  `--dataset` argument.

**F20 remains unverified.** `roberta-large` (BERTScore's default for `lang="en"`)
is not in the local HuggingFace cache, so confirming whether the published
0.905–0.923 range sits near a floor needs a ~1.4 GB download. No GPU required —
a handful of string pairs is enough once the model is present.

---

## 9. Fix list — Stages 1 and 2

> **Status lives in §0, not here.** This list was written during the audit and
> the "Why it matters" column is in the present tense of that moment. Every
> Phase 0 and Phase 1 item is now **done**, and F62 / F68 / F69 / F72 were
> withdrawn or reframed — §0 says what was actually applied. Line references
> in this section predate the Phase 1 edits and have shifted.

"Re-run?" says whether applying the fix invalidates artifacts already on disk:
**no** = outputs unchanged, safe to apply any time; **mine** = rule banks must be
rebuilt; **retrieve** = retrieval must be re-run; **train** = models must be
retrained. Effort S = under an hour, M = a few hours, L = a day or more.

### Stage 1 — mining

| ID | Fix | Effort | Re-run? | Why it matters |
|---|---|---|---|---|
| F5 | Copy `all_facts.txt` into `data/original/icews18/` (verify row count and sort order first) | S | no | **Blocker** — ICEWS18 retrieval fails at validation without it |
| F48 | Pass `period=1` in `learn.py`'s two `get_unique_quads_per_rels` calls | S | mine | Removes the 24× timestamp inflation. Affects `ragtkgc` / `ragtkgc_no_walks` only |
| F33a | Cache `unique_quads` to disk, keyed by dataset + `infer_from_type` | S | no | 31.4 s saved per ICEWS14 run; more on ICEWS18. Enables honest amortised timing |
| F33b | Move `start = time.time()` above `Grapher`; print a phase breakdown | S | no | Table 3 currently omits the new algorithm's main cost |
| F37a | Rename output files to carry the algorithm, and only parameters that apply | S | mine | `no_mining` banks currently record `n200_exp_s1`, none of which affect them |
| F37b | Rebuild all banks and `common_rule_pool.json` from scratch | M | mine | Pool disagrees with bank files on 6,067 `ragtkgc` rules and over-claims provenance |
| F37c | Key `algorithm_stats` by run id, not algorithm name | S | mine | Re-running an algorithm silently overwrites its own previous stats |
| F38a | `if seed:` → `if seed is not None:` | S | mine | `-s 0` is currently ignored |
| F38b | Derive each worker's seed from `seed + worker_index` | S | mine | Today `-p` changes the resulting bank, not just the runtime |
| — | Rename the algorithm `ragtkgc_no_mining` → `exhaustive` | S | mine | It mines exhaustively; the current name says the opposite |
| F40 | Iterate `quads_all` and look `quads` up by key | S | no | Currently correct by luck; noted in code |

Paper-only, no code: **F34** (say "all type-compatible length-1 rules with
positive confidence"; 62,156 candidates → 41,068 kept), **F35** (run the headline
comparison with no confidence filter), **F36** (state the superset argument),
**F39** (call it relation-signature labelling, not entity types).

### Stage 2 — retrieval

| ID | Fix | Effort | Re-run? | Why it matters |
|---|---|---|---|---|
| F43 | Cache the four `np.where` lookups (`col_rel` by relation, `col_sub` / `col_obj` by entity, `col_time` via `searchsorted`) | M | no | **Highest value in the audit.** ~2.0 M full scans per test split become ≤460. Verify by byte-diffing output against a current run |
| F6 | Derive `num_relations` from `relation2id.json` | S | no | Hardcoded per dataset; load-bearing for both the modulo and the direction test |
| F7 | Delete the dead `idx_test` guard (correct only for `test`; provably removes nothing, since the strict `<` filter already excludes the target) | S | no | Repairing it needs a split offset threaded through; deleting is the smaller correct change. Keep it only if the time filter ever becomes `<=` |
| F44 | Guard `early_stop_at_num_facts` against `num_facts is None` | S | no | Currently raises `TypeError` when used alone |
| F30 | `--rule_length_all` from `type=bool` to `store_true` / `store_false` | S | no | Cannot be disabled from the CLI today |
| F45 | Restrict retrieval to length-1 rules in code, or implement multi-hop matching | M | retrieve | Only `body_rels[-1]` is consulted, anchored at the query subject — wrong for longer rules. Noted in code |
| F41 + F49 | Pass `retrieve_type` into `Retriever`; extend `prepare_bs` from `col_sub == sub` to `(col_sub == sub) \| (col_obj == sub)` | S | retrieve | Makes the `standard` baseline reproducible; no script currently emits it |
| ~~F47~~ | **Done** — `int(time_in_id) // period` instead of `/` | S | retrieve + train | Format parity with the baseline. The token saving turned out to be ~1%, not 5–10%: T5's vocabulary holds `4.0` as one piece, so `334.0` and `334` are both two tokens |
| F2 | Add a note at the anchoring branch | S | no | Approved as a note only; not yet written |

Paper-only, no code: **F42** (confidence does not select facts — a clearance, not
a defect), **F46** (state that history comes from the full corpus up to the query
timestamp).

### Stage 3 — training files and the re-filter path

| ID | Fix | Effort | Re-run? | Why it matters |
|---|---|---|---|---|
| F55a | **DONE.** Stopped writing `_idx_fine_tune_all.txt` (byte-identical duplicate, zero readers) in both `retrieve.py` and `apply_history_filters.py`, and removed it from the README's output list. The 30 existing files (4.61 GB) have since been **deleted from disk** | S | no | Halves every future `history_facts/` directory |
| F55b | Drop `facts[].rule_ids` from the metadata — derivable by inverting `fired_rules[].fact_indices`, and `_apply_filters` already contains that inversion | S | no | 24.1% of metadata. No change to rule identity anywhere |
| F55c | Skip writing the uncapped `history_facts/` when `--save_metadata` is on | S | no | 4.9 G per run; regenerable via `apply_history_filters` with no filters |
| F55d | Drop `facts[].text` from the metadata — `text = _fact_to_line(all_facts[idx])` | M | no | 37.2% of metadata, but the 2b path gains a dependency on `all_facts.txt` + `ts2id.json` and must share the formatter |
| F50 | Assert `len(inputs) - 1 == len(test_ans) == len(rule_ids)` in `convert_txt_to_json` | S | no | A lost blank line shifts every later (context, target) pair silently |
| F51 | Seed the subset sampler (`random.Random(seed)`), skip rather than raise when `num > len(data_list)` | S | no | `_16.json` currently differs every run; oversized `--nums_sample` crashes |
| F52 | Delete the discarded `get_ins()` call and comment why; optionally add `--instruction` to enable it | S | no | Dead call implies instructions are used when they are not (§4b item 6) |
| F53 | Keep `--dir_of_entities2id` and wire `entities` up for `id.Name` targets; drop the no-op `use_ids`; add `--index_target` | S | train | The dead parameter is exactly the hook the answer-format change needs |
| F54 | ~~Strip the blank padding in the `--use_ids` answers writer~~ — **dropped, `--use_ids` is not in the plan.** Only the naming hygiene survives: stop putting `test_ids` in hand-typed dataset names | S | no | Two unrelated things share the `_ids` suffix, which caused real confusion |
| F56 | Cross-reference comment in `TLR.build_tl` and `apply_history_filters._apply_filters` | S | no | Two hand-synchronised implementations of one filter rule |

### Stage 4 — fine-tuning

| ID | Fix | Effort | Re-run? | Why it matters |
|---|---|---|---|---|
| F57 | Add `trainer.save_model(os.path.join(output_dir, trained_model_name))` + `tokenizer.save_pretrained` to `training_LLaMA.py`, mirroring `training_T5.py:209-213` | S | no | **Blocker.** LLaMA training currently produces no artifact at all |
| F15 | Add `--tail_truncate_long_inputs` and `--eval_file_path` to `training_LLaMA.py` | M | train | Otherwise LLaMA runs on the old regime while Flan-T5 runs on the new one |
| F58 | **No fix needed** — correct at batch size 1 (no padding ever occurs). Add a comment at line 111 recording that it depends on that, and switch to `attention_mask[i].sum() - 1` only if the batch size is ever raised | S | no | Correctness is coupled to `per_device_train_batch_size=1` non-obviously |
| F60 | Remove or justify the `icews14` `+1` in the label-mask slice; fix the unreachable guard at line 74-80 | S | train | ICEWS14 and ICEWS18 LLaMA models are trained on different objectives |
| F63 | Add `--seed`, pass it into `TrainingArguments` | S | no | Required before mean ± std over seeds is possible; every run is currently seed 42 |
| F61 | Use `T5TokenizerFast`; drop the duplicate full tokenisation at `training_T5.py:165` | S | no | Two full passes with the slow tokenizer on every run |
| F62 | **WITHDRAWN — refuted by measurement.** The globals are already part of the map fingerprint and the cache invalidates correctly on `datasets` 3.2.0 and 3.5.0; the loader also re-reads a rewritten JSON. See §0. `download_mode='force_redownload'` added as version-insurance only, plus a comment on the global dependency | S | no | — |
| F59 | Guard the 1024 subsample so `step >= 1` still spreads (use `max(1, n // 1024)` with an index list) | S | train | Degenerates to "first 1024 contiguous" for `1024 <= n < 1536` |
| F65 | Drop `auto_find_batch_size` or raise the base batch size | S | no | Inert at batch size 1 |
| F66 | Set `weight_decay` explicitly for LLaMA, or document why it differs | S | train | Undocumented divergence from the T5 regime |
| F64 | Correct the learning-rate comment — this path is full fine-tuning, not adapter | S | no | Comment describes a regime the script does not use |
| F13 | Either raise the epoch count so the controller can act, or describe `--eval_file_path` honestly as best-checkpoint selection | S | no | Do not call it adaptive LR scheduling in the paper at 3 epochs |
| F12 | Add the regime table (Flan-T5 74,845 / 3 epochs vs LLaMA 1,024 / 1 epoch) to the setup section | S | no | Paper currently compares across regimes without saying so |

### Stage 5 — testing and decoding

| ID | Fix | Effort | Re-run? | Why it matters |
|---|---|---|---|---|
| F71 | Change the `rank = 5` sentinel to `float('inf')` | S | no | **Do this before adding H@10 or filtered metrics**, or every miss scores as a hit@10 |
| F68 | **REFRAMED — premise was wrong.** The variant is already in the JSON basename and the model name, twice. The real hazard is path length: 240 of the 260 characters Windows allows. Fixed by auto-numbering (`_2`), a run manifest at `results/{ds}/_runs.jsonl`, a warning above 240 chars, and `makedirs` for the results folder. See §0 | S | no | Silent overwrite on a name collision; provenance unrecoverable from the filename |
| ~~F67~~ | **Done** — beam search at k=10, `length_penalty` 0.6, replacing `permute` in both branches | M | no | `permute` is unsound above `dec_cand=1`; raising it fabricates candidates rather than finding them |
| F43-adjacent | Batch the prediction loop (`run_hf.py:278-312`); `padding_side='left'` for LLaMA; drop F74 at the same time | M | no | The only large inference speedup available; makes the LLaMA re-run affordable |
| F69 | **REFRAMED — the redundant half was the other one.** The pre-scan is cheap and reports lengths before GPU time is spent; `overall_len_stats` inside the loop was the duplicate (proven identical across all five branch combinations, prompts byte-identical). Removed that, and added `effective_after_truncation` — the length actually fed to the model, which neither statistic reported. See §0 | S | no | One redundant tokenisation pass over the whole set per run |
| F73 | Refuse to score, or loudly flag, result files shorter than the split | S | no | A crashed run currently yields a plausible metric over a subset (F22) |
| F70 | Pass `input_ids` to `predict` instead of decoding back to text | M | no | Truncation is currently approximate; verify the cap actually holds |
| F28 | Fix `rel = rel.strip()[:-1]` to strip a trailing comma only if present | S | no | Corrupts the recorded relation for `standard` (metrics unaffected, fields wrong) |
| F72 | **REJECTED as written.** Hardcoding duplicates a fact the tokenizer already owns, breaks silently on any model change, and the `ValueError` is correct behaviour — it refuses to guess rather than truncating to a wrong length. The `> 1_000_000` guard is HF's own idiom for its `VERY_LARGE_INTEGER = 1e30` sentinel. Applied instead: `get_token_limit` returns which attribute the value came from, and that is logged. See §0 | S | no | The value governs how much history survives truncation, and its source was invisible |
| F19 | Drop `cs`, or replace with an embedding cosine | S | no | Character-frequency cosine; an anagram scores 1.0 |
| F20 | Recompute BERTScore with `rescale_with_baseline=True` | S | no | Un-rescaled values may sit near a floor |
| F17/F18 | Report H@1 only for Flan-T5; H@1/H@3 for LLaMA. Drop the duplicated columns | S | no | H@10 is not computable from 1 or 3 candidates |
| — | Implement time-aware filtered Hits@k over existing `results/*.jsonl` | M | no | Required for any comparison against GenTKG / ISI / RECIPE-TKG |

### Stage 6 — metrics

| ID | Fix | Effort | Re-run? | Why it matters |
|---|---|---|---|---|
| F75 | `logging.basicConfig(..., force=True)` in all three `_setup_logging` helpers, or use a named logger with a fresh handler | S | no | Runs after the first in a notebook session log to the wrong file; this is why provenance was unrecoverable |
| F79 | Read the split size from the data; **error** rather than score when a results file is shorter | S | no | A crashed run currently produces a plausible metric over a subset |
| F78 | `json.loads` instead of `eval` | S | no | Faster, safer, clearer failures |
| F76 | On the OpenAI path, name the output after the model that actually ran | S | no | A file can be named after a model that never ran |
| F80 | Remove the duplicate `load("bertscore")`; add `rescale_with_baseline=True`; state that BERTScore is top-1 | S | no | Un-rescaled values may sit near a floor (F20) |
| F20 | Recompute BERTScore rescaled and compare against the published range | S | no | Needs a ~1.4 GB model download; still unverified |
| — | Add H@10 **only after** F71 (the `rank = 5` sentinel) is fixed | S | no | Otherwise every miss counts as a hit@10 |

---

## 10. Consolidated plan

Five phases. Phases are ordered by dependency, not by importance: each one is
safe to start only once the previous is done. Within a phase, order does not
matter.

### If only five things get done

1. **F57** — `training_LLaMA.py` saves nothing. Any LLaMA work is void until fixed.
2. **F71** — the `rank = 5` sentinel, before any H@10 or filtered metric exists.
3. **F43** — retrieval caching. Decides whether the full experiment grid is affordable.
4. **Time-aware filtered Hits@k** — the only way the comparison table is defensible.
5. **F36** — state the superset argument. Free, and it is the paper's strongest claim.

### Phase 0 — blockers (nothing else is safe first) — **DONE**

| ID | Action |
|---|---|
| F5 | Copy `all_facts.txt` into `data/original/icews18/`; verify row count = train+valid+test and chronological sort |
| F57 | Add `trainer.save_model` + `tokenizer.save_pretrained` to `training_LLaMA.py`, mirroring `training_T5.py:209-213` |
| F71 | `rank = 5` → `float('inf')` in `compute_metrics_from_results.py` |
| F75 | `logging.basicConfig(..., force=True)` in `run_hf.py`, `retrieve.py`, `apply_history_filters.py` |

F75 belongs here rather than in hygiene: until it is fixed, no run that follows
is reliably traceable, and that is what made the `.369` provenance unrecoverable.

### Phase 1 — output-neutral hygiene, one pass — **DONE (28/28)**

Nothing here changed any output on complete artifacts; `.369` on the reference
file still reproduces. Two items do change *behaviour* on defective inputs, by
design: F79 now skips incomplete results files rather than scoring a subset, and
F80 reports rescaled BERTScore rather than raw.

| Area | IDs |
|---|---|
| retrieval | F6, F7, F44, F30, F45 (hard-filter to length 1), F28 |
| mining | F38a, F40 (comment only, by decision) |
| training files | F50 (+ the same guard in `convert_txt_to_json`), F51, F52, F55a |
| training | F61, F63, F64, F65 · **F62 withdrawn** |
| testing | F68 (reframed), F69 (reframed), F72 (rejected), F74 |
| metrics | F78, F79, F76, F80 |
| comments only | F58 (batch-size-1 dependency), F60 (slice left untouched, marked unverified), F56 (cross-reference) |

Applied beyond the original list: `T5TokenizerFast` in `run_hf.py` as well as
`training_T5.py`; `download_mode='force_redownload'` on all six `load_dataset`
calls; `--seed` on `create_json_train.py`; `os.makedirs` for the results folder.

### Phase 2 — make it fast, before pricing any experiment

| ID | Action | Verification |
|---|---|---|
Three independent groups. Suggested order **A → C → B**: F43 carries the most
value and needs a clean before/after; group C is format-coupled and must land in
one shot; group B is the most invasive to `run_hf.py` and benefits from the rest
being settled.

**Group A — the retrieval bottleneck (alone, first)**

| ID | File | Action | Verification |
|---|---|---|---|
| F43 | `data_utils/TLR.py` | **DONE.** Memoised `col_rel` / `col_sub` / `col_obj` lookups in `_column_indices`; replaced the time scan and its Python set with one `searchsorted` cut point; `assume_unique=True` on `intersect1d` | Byte-identical on all three output files for both algorithms; anchors and cut proven equal to the old path on 10,371 real samples across both datasets; six filter configurations deterministic |
| F33a | `data_utils/rules_learning/basic.py`, `learn.py` | **DONE.** `get_unique_quads_per_rels_cached` pickles the result to `data/processed_new/<dataset>/cache/`, named by dataset, `period` and `infer_from_type`. `learn.py` calls it instead of the raw function | Cached result compared element-wise against a fresh computation for both modes; a truncated cache recomputes rather than crashing |

**F33a measured result** (`gtkgt` env):

| mode | used by | ICEWS14 first run | cached | speedup | cache file |
|---|---|---|---|---|---|
| `infer_from_type=False` | `ragtkgc`, `ragtkgc_no_walks` | 54.7 s | 0.211 s | 258× | 3.4 MB |
| `infer_from_type=True` | `ragtkgc_no_mining` | 53.6 s | 0.031 s | 1,736× | 0.1 MB |

ICEWS18 is far worse and is what makes this item matter for Phase 3:

| mode | ICEWS14 | ICEWS18 | ratio |
|---|---|---|---|
| `infer_from_type=False` | 54.7 s | **1,601.8 s** (26.7 min) | 29× |
| `infer_from_type=True` | 53.6 s | **1,137.4 s** (19.0 min) | 21× |

ICEWS18's `train.txt` has 373,018 lines — **5.0× the data for 29× the time**, so
the inner loop is superlinear: `add_quad` rescans the accumulated per-relation
list for every quad. The cache sidesteps it rather than fixing it, which is the
right trade here (the function is called once per dataset, not per sample).

The plan estimated 31.4 s; it is **44–59 s per ICEWS14 run**, paid on every invocation.
The type mode returns 62,156 elements, matching the candidate count in F34 —
this function is what produces the exhaustive algorithm's candidate set.

`period` is in the cache name deliberately: **F48 changes it from 24 to 1**, and
a cache keyed only by dataset and `infer_from_type` would hand a Phase 3 re-mine
the pre-F48 timestamps — reproducing the exact bug F48 exists to fix.

`pickle`, not JSON: the two modes return different value types (list of arrays,
set of ints) under integer keys, none of which JSON represents. Content hashing
of the inputs was considered and dropped — the splits are fixed for these
experiments, so editing one means deleting the cache file by hand.

**F43 measured result** (300 ICEWS14 test samples, `gtkgt` env):

| algorithm | before | after | speedup | full test split | full train split |
|---|---|---|---|---|---|
| `gtkg` | 23.93 s | 5.12 s | **4.67×** | 9.8 min → 2.1 min | — |
| `exhaustive` | 75.98 s | 6.24 s | **12.18×** | 31.1 min → 2.6 min | ~5.3 h → ~26 min |

`build_tl`'s own time fell from 59.9 s to 1.56 s. One line — a full comparison
against the 27 MB fixed-width `col_rel` array, executed once per rule per
sample — was 61% of total runtime. The cache that removes it holds one index
per fact: 0.73 MB for ICEWS14, 3.75 MB for ICEWS18.

The cost gap between the algorithms narrowed from 3.2× to 1.4× per sample,
because the fix scales with rule-bank size and `exhaustive` has 5× more rules
than `gtkg`. Relevant to Table 3.

F43c (a `searchsorted` intersection to remove `intersect1d`'s internal sort)
was **deliberately dropped**: the bottleneck is gone, retrieval no longer
dominates anything, and it would be a third logic change for a small absolute
gain. `prepare_bs` was left untouched — it has the same pattern but is
unreachable until F41/F49, so a change there could not be byte-verified.

F43 is the measured bottleneck: `np.where(col_rel == ...)` recomputed per rule
per sample — 1,965,954 rule iterations for the exhaustive test split, ~4,274×
reduction available. It decides how long Phase 5 takes, which is why the phase
exists. Verify on a bounded slice (~300 samples per algorithm) rather than full
splits, keeping the pre-change output to diff against.

F33a is the "get_unique should happen once, not per run" item raised directly:
~55 s per ICEWS14 run and **26.7 min per ICEWS18 run**, and it makes amortised
timing honest for Table 3.

**Group B — batch the evaluation loop (independent)**

| ID | File | Action | Verification |
|---|---|---|---|
| — | `run_hf.py` | Batch the prediction loop; `padding_side='left'` for LLaMA; length-bucket the batches | Metrics identical on the test split |

Now genuinely safe because F74 restored T5's real `<pad>` — before that, batched
encoder inputs would have been padded with `</s>`.

**Group C — metadata slimming — DONE 2026-09-08**

Applied as one change, because F55b and F55d together mean the whole `facts`
array goes: all four of its fields derive from `fired_rules`.

| ID | File | Action |
|---|---|---|
| F55b | `data_utils/TLR.py`, `data_utils/apply_history_filters.py` | `facts[].rule_ids` dropped — derived by inverting `fired_rules[].fact_indices` |
| F55c | `data_utils/retrieve.py` | `--save_metadata` now refuses any filter flag, and writes no `history_facts/` at all (the directory is not even created) |
| F55d | `data_utils/TLR.py`, `data_utils/apply_history_filters.py` | `facts[].text` dropped — rendered from `all_facts.txt` by the extracted `fact_to_line` |
| — | both | `facts[].conf` and `facts[].idx` dropped as a consequence; nothing outside `_apply_filters` read them |

Per-sample metadata is now `{sample_idx, query_line, fired_rules}`.

`fact_to_line` and `time_period` were lifted to module level in `TLR.py`;
`Retriever._fact_to_line` delegates to them and `apply_history_filters` imports
them, so the rendering half of F56's duplication is gone. `_apply_filters` takes
a `render` callable rather than reading stored text.

Measured on ICEWS14 `test` / `gtkg` / `--inverse_body_object_match`:

| | |
|---|---|
| metadata, test split | **600.1 MB → 39.3 MB** (93.4% smaller, 15.3×) |
| re-filtered output == direct retrieval, 5 filter configs | 10/10 files byte-identical |
| retrieval output unchanged by the edits, 5 configs | 10/10 byte-identical |
| rule-id content actually compared | 347,128 ids (base), 118,266 (n50), 57,359 (top10+n50) |

93.4% rather than the estimated 61.3%: removing the array also removes `idx`,
`conf` and the JSON punctuation. The first verification round used bank
`060426144613`, in which **0 of 11,707 rules carry a `rule_id`**, so its rule-id
comparisons were vacuous; the numbers above come from re-running with
`170426145844` (41,068 rules, all with ids). The comparison script now aborts
when a bank carries no rule ids.

Two consequences to keep in mind:

- **The metadata is no longer self-contained.** It stores fact indices, so
  `apply_history_filters` needs `data/original/<dataset>/` present
  (`all_facts.txt`, `ts2id.json`, plus `entity2id.json` / `relation2id.json`
  under `use_ids`). The header carries `original_data_dir`; a missing directory
  raises `SystemExit` rather than a `KeyError` mid-render.
- **`fired_rules` is now stored in build order**, not confidence-sorted, and the
  re-filter path stable-sorts only when `top_k_rules` is set. This keeps the
  derived rule-id order equal to `build_tl`'s in both cases. Measured: the fix is
  **latent, not load-bearing** — all 434 relations in the bank are already
  confidence-descending on disk (`Rule_Learner.sort_rules_dict` sorts before
  saving), so an unconditional sort matches on 7,371/7,371 samples too. It is
  insurance against an unsorted bank, nothing more.

`retrieve.py --save_metadata` combined with `--num_facts`, `--top_k_rules`,
`--confidence_threshold` or `--early_stop_at_num_facts` now exits before any I/O.
`early_stop_at_num_facts` is included because it truncates retrieval itself, so
its metadata is partial even with no post-filter. Verified end to end on the
`valid` split: the output directory contains only `metadata/` and
`test_answers/`.

### Phase 3 — re-mine, as one batch

Every item invalidates the rule banks, so applying them separately means
re-mining several times.

| ID | Action |
|---|---|
| F48 | **DONE 2026-09-08.** `period=1` in `learn.py`'s two `get_unique_quads_per_rels_cached` calls |
| F86 | **DONE 2026-09-09**, as an **opt-in** third value of `--transition_distr`. `exp` keeps the published behaviour and stays the default; `exp_scaled` normalises the time gap by the smallest timestamp spacing in the data. Opt-in rather than unconditional so the `gtkg` baseline remains the published algorithm — the paper cites GenTKG's numbers, and silently substituting a corrected transition would leave no faithful bank to sanity-check the pipeline against. `transition_distr` is already part of the bank filename, so a bank records which transition produced it. Both variants are mined, `exp_scaled` as an ablation |
| F37a | **DONE 2026-09-09.** `Rule_Learner._rules_filename` puts the algorithm first and records only the parameters that apply; both save methods share it |
| F37c | **DONE 2026-09-09**, reframed. Literal re-keying of `algorithm_stats` would break `conf_stats.load_rule_conf_map` and its two consumers, so `algorithm_stats[alg]` still holds the latest run and an append-only `runs` list carries the history |
| F38b | **DONE 2026-09-09.** Worker seed = `seed + i`. joblib gives each worker its own process and numpy's global RNG is per-process, so upstream's shared seed handed every worker the identical stream: two relations with equal candidate counts drew the same index in lockstep, leaving one independent stream rather than one per worker. Verified directly — three workers, shared seed, identical first six draws. **Deliberately deviates from upstream** (TLogic and GenTKG both use `np.random.seed(seed)`), which is acceptable only because bit-exact reproduction of the published bank is not attempted: mining at `-l 1` rather than `-l 1 2 3` already changes the RNG stream, since walks at lengths 2 and 3 advance it between relations. It does **not** make the bank independent of `--num_processes`: measured, same seed, `-p 1` → 268 rules, `-p 2` → 272. Reproduction needs seed **and** process count, as F63 records |
| F33b | **DONE 2026-09-09.** Phase origins above `Grapher`; a four-phase breakdown printed after the mining total **and persisted** to `logs/learn_stats_{dataset}_{mining}_{dt}.json` with the parameters, the rule counts and the bank filename it describes |
| — | **DONE 2026-09-09.** `ragtkgc_no_mining` → `exhaustive` throughout |
| — | **DONE 2026-09-09.** Naming-convention sync, below |

**Naming-convention sync (DONE 2026-09-09).** Artifact names now carry the
experiment's identity end to end, so a results file traces back to a rule bank
without external notes. The variant token and the per-stage names are documented
in the README's *Naming conventions* section, which is the reference; recorded
here is only what changed and why.

| Problem | Fix |
|---|---|
| `--mining` named the output folder only, while the bank came from `glob('*rules.json')[0]` — so `-m gtkg` could retrieve with the `exhaustive` bank, silently | `retrieve.py:resolve_rules_file` selects by `_{mining}_` and refuses on 0 or >1 matches; explicit `-r` always wins, which the `raw`/`standard` baselines need |
| bank filenames recorded walk parameters for algorithms that never walk | `_rules_filename` omits them; `exhaustive` banks are `{dt}_exhaustive_rules.json` |
| `--mining` accepted free text, so a typo mined nothing and still saved a bank | `choices=` on both `learn.py` and `retrieve.py` |
| `create_json_train --dataset` was the output filename, so every variant of a dataset wrote the same `icews14.json` | renamed to `--name_train`, required |
| suffix tokens were verbose and propagated into three name layers | `_inverse_included`→`_inv`, `_early_stop`→`_es`, `_top{k}rules`→`_k{k}`, `_thresh{x}_{mt}`→`_ct{x}-{mt}` |
| every training run wrote checkpoints to `./models/checkpoint-{step}`, so seed repeats of one configuration overwrote each other and `load_best_model_at_end` could restore another run's checkpoint | `TrainingArguments(output_dir=./models/{trained_model_name})` in both trainers; the final model saves to the same directory |
| `--dataset_path` defaulted to a path from a layout no longer produced, and its basename becomes part of the results filename | required, no default |

Verified: `gtkg` and `exhaustive` both mine and write the new names;
`exhaustive` reproduced **41,068 rules**, matching the pre-existing bank, so the
rename changed no behaviour. Bank resolution returns the right bank for `gtkg`
and `exhaustive`, refuses for `ragtkgc`/`standard` with the candidates listed,
and honours an explicit `-r`. `load_rule_conf_map` still reads both algorithms
from the rebuilt pool (268 / 41,068 rules), confirming F37c's non-breaking form.
Worst-case full path under the new scheme is **145 characters** against the
240-character budget. Zero occurrences of the old tokens remain outside this
file.

Incidental observation from the rebuilt pool, relevant to **F36**: of 41,068
rules, 40,800 were found by `exhaustive` only and 268 by both, with **none found
by `gtkg` alone** — direct evidence for the superset argument, though at
`--num_walks 1` it is a weak sample. Worth re-checking after the real re-mine.

**The published mining settings**, recovered from TLogic's `mycode/run.txt` and
GenTKG's README, are `-l 1 2 3 -n 200 -p 15 -s 12` with `--transition_distr exp`
(TLogic used `-p 16` for ICEWS14, `-p 15` for ICEWS18). GenTKG notes "only
length=1 is used", i.e. the bank holds all three lengths and retrieval consults
only length-1.

The rebuild uses **`-s 12` and `-p 15`, matching the paper**, but mines `-l 1`
only: lengths 2 and 3 cost far more and are never consulted. That alone
forecloses bit-exact reproduction, because walks at lengths 2 and 3 advance the
RNG between relations and so change the length-1 rule set. Bit-exactness was
therefore not a goal, which is what makes F38b's deliberate deviation
acceptable. The in-house `gtkg` row is a sanity check within a few points, not a
reproduction; the published numbers are cited directly, and the §4b/§4c
comparability disclaimer still applies because retrieval and prompt format also
differ (F47, F53).

**Consequence of mining the ablation:** two banks per dataset now match
`_gtkg_`, so `resolve_rules_file` refuses to guess and every `gtkg` retrieval
must pass `-r` explicitly. That is the intended behaviour for an ablation — the
choice of transition becomes a stated parameter of the run rather than a
filesystem accident.

Then: back up `common_rule_pool.json`, **delete it**, and rebuild banks for
ICEWS14 and ICEWS18 for `gtkg` and `exhaustive` (F37b). Record end-to-end wall
clock plus the phase split for Table 3.

`ragtkgc` is dropped from the paper, so it needs no re-mine — F48 is applied for
correctness, not for a reported number.

**F48 measured result.** `data/original/<ds>/train.txt` and `ts2id.json` already
express time in the same unit (day × 24), so the default `period=24` multiplied
it a second time: stored timestamps ran `0 … 174,528` against a graph maximum of
`8,736`. `sample_start_edge2` hands that inflated value straight to
`transition_step`, which compares it against `Grapher.train_idx` timestamps, so
the "strictly earlier" constraint was tested against the wrong scale.

| | before | after |
|---|---|---|
| `unique_quads` timestamp range (ICEWS14) | 0 … 174,528 | 0 … 7,272 |
| `Grapher.train_idx` range | 0 … 7,272 | 0 … 7,272 |
| start edges above the graph maximum, i.e. filter admits everything | 78,814 / 83,866 (**94.0%**) | **0** |
| first day at which the filter goes inert (train covers 0–303) | 16 | never |
| length-1 walk success, first 60 relations, seed 42 | 39,261 / 39,365 (99.74%) | 15,300 / 39,365 (38.87%) |

The fix removes **61.0%** of previously successful walks — those succeeded only
because the constraint was inert, meaning they were built on facts at or after
the start edge. Verified that `unique_quads` timestamps are now a strict subset
of the graph's, with identical min, max and distinct count.

Affects `ragtkgc` and `ragtkgc_no_walks` only: `gtkg` calls `sample_walk` without
`unique_quads`, and `exhaustive` never calls it. So no reported number changes.

Also measured: **the type mode is invariant to `period`** — `add_quad`'s
`infer_from_type` branch never reads the timestamp column, confirmed identical
for `period=1` and `period=24` (452 relations, 62,156 elements both ways). The
F33a cache name therefore carries `period` for the node mode only
(`unique_quads_<ds>_p1_node.pkl` vs `unique_quads_<ds>_type.pkl`), which avoids a
19-minute ICEWS18 recomputation for a byte-identical result.

### Phase 3 — the rebuild, as mined 2026-09-09

Six banks, `-s 12 -p 15` matching the published settings, `-l 1` only.

| dataset | algorithm | rules | wall clock |
|---|---|---|---|
| icews18 | `exhaustive` | 87,860 | 35m15s (`unique_quads` 17.9 min, mining 16.7 min) |
| icews18 | `gtkg` `exp` | 12,339 | 52 s |
| icews18 | `gtkg` `exp_scaled` | 13,047 | 52 s |
| icews14 | `exhaustive` | 41,068 | 1m59s |
| icews14 | `gtkg` `exp` | 7,965 | 21 s |
| icews14 | `gtkg` `exp_scaled` | 8,472 | 22 s |

`exhaustive` on ICEWS14 reproduced **41,068 rules exactly**, matching the bank
that existed before any of today's changes — it is deterministic and F48, F86 and
F37a leave it untouched.

**F86 ablation result:** `exp_scaled` finds **~6% more rules** than `exp` on both
datasets (8,472 vs 7,965; 13,047 vs 12,339). A graded transition explores more
diverse walks, so it discovers more distinct rules. This is the number to report
alongside the regime split (0% graded under `exp`, 65.3% under `exp_scaled`).

**F36 superset claim — now measured, not argued:**

| | ICEWS14 | ICEWS18 |
|---|---|---|
| `gtkg` rules (union of both transitions) | 10,642 | 17,372 |
| `exhaustive` rules | 41,068 | 87,860 |
| `exhaustive`-only | 30,426 | 70,488 |
| **`gtkg`-only** | **0** | **0** |

Every rule `gtkg` finds is among `exhaustive`'s, which finds 3.9× more on
ICEWS14 and 5.1× more on ICEWS18. Since the union is a subset, each transition
variant is individually a subset too.

### Phase 4 — the protocol change

**Group E dropped 2026-09-09.** The `standard` baseline will not be rebuilt;
ISI's published numbers are cited instead, with the differences stated. That
removes F41, F49 and the F43 optimisation of `prepare_bs` from the plan.

Reasons the existing `standard` data could not have been used regardless:
it cannot be regenerated (`retrieve_type` is never passed to `Retriever`, so
`build_bs` is unreachable); its history lines carry **raw** timestamps
(`6264:`) where every other variant carries days (`197.0:`); it averages ~1,471
tokens against ~456 for the `n50` variants, so most of it is truncated away; and
`prepare_bs` matches `col_sub == sub` only, excluding the object-role facts that
the main method uses via `--inverse_body_object_match` — an asymmetry that
flatters the method rather than testing it.

For the record, ISI's history construction (`usc-isi-i2/isi-tkg-icl`):
`history_type` `entity` (all relations) or `pair` (query relation only);
`history_direction` `uni` or `bi`, where `bi` is `head_search_space.update(
tail_search_space)` — a shallow top-level merge whose union semantics could not
be verified from the source; strictly-earlier time filter; `quadruples[
-history_len:]`; and per-fact string `"{time}:[{entity},{relation},{label}.{
target}]\n"`, which is the same shape our repo uses because GenTKG inherited it
from ISI.

**Three `X.Name` schemes, which F53 must not conflate:** ISI's number is a
**per-prompt local label** from `candidates_mapping`; GenTKG's is the **global
entity id** (`id_obj = self.entities[obj_in_word]`); ours carries **no index at
all**. F53 follows GenTKG, which is what `--dir_of_entities2id` was always the
hook for. `--use_ids` is a fourth thing — the id *instead of* the name — that
neither upstream does.

**Group A — DONE 2026-09-09.** Scoring only; no run required.

| ID | File | Change |
|---|---|---|
| filtered H@k | `utils.py`, `run_hf.py`, `compute_metrics_from_results.py` | time-aware filtered ranking beside the raw ranking, from one shared implementation |
| F19 | `utils.py` | `cs` removed |

`load_true_objects` indexes `all_facts.txt` — verified to be exactly
train ∪ valid ∪ test (90,730 = 74,845 + 8,514 + 7,371 for ICEWS14) — as
`(subject, relation, day) -> frozenset(objects)`. The key uses the **rendered
day**, computed as `int(ts2id[date]) // period`, so a results row needs no
timestamp conversion beyond `int(float(...))` and the code is correct both
before and after F47. `all_facts.txt` stores **date strings**, not integers, so
`ts2id.json` is required for the mapping.

`ranks()` returns raw and filtered together and is used by both the live metric
and the offline scorer, so the two cannot drift; `metric.resolved` records how
many queries were found in the index, because an unresolved query silently
returns its raw rank.

**Measured on the 50 existing ICEWS14 results files:**

- **Filtering was inert on every Flan-T5 file** — `permute` yields one candidate,
  so the rank is 1 or infinity and there is nothing above the target to filter.
  This was F84 from a second direction, and F67 resolved it: on the same model
  and split, beam search lifts fH@1 to 0.353 against a raw 0.343.
- On LLaMA, which already uses 3 beams, filtering lifts H@1 by **+0.010**
  (0.338 → 0.348 on `ragtkgc_test`) and **+0.006** (0.317 → 0.323 on
  `gtkg_test`). H@3 is unchanged, as it must be when only 3 candidates exist.
- The `resolved` guard caught two classes of silent failure it was written for:
  every `--use_ids` variant resolves **0.0%** (its query lines carry ids, the
  index is keyed by names) and every `standard` file resolves **0.0%** (raw
  timestamps against day keys). Without the guard both would have reported raw
  numbers under a "filtered" heading.

**Decoding cost measured** (RTX 4050, 40 real prompts at the representative ~456
tokens, one sample at a time):

| decoding | ms/sample | 7,371-row split |
|---|---|---|
| greedy + `permute` (current, 1 candidate) | 178.6 | 21.9 min |
| beam k=3 | 296.3 | 36.4 min |
| beam k=10 | 316.6 | 38.9 min |

**k=10 costs only 1.07× k=3**, because the beams batch on the GPU and are
trivial next to the 456-token prefill. So k=10 for H@1/3/10 is the right choice,
not k=3. Dropping `cs` saves **76.6 ms across the whole split** — it was removed
for being a character-bag cosine, not for speed. BERTScore is a separate script
and never ran inside the test loop.

**F67 decided: beam search**, replacing `permute` in both branches.
`permute` is verbatim from ISI's `model_utils.py`, where `dec_cand` is
configurable; this repo hardcodes 1, collapsing it to greedy. Raising
`dec_cand` would not fix it: `permute` recurses over the scores of the *greedy*
generation, so a branch at step *t* reads a distribution the model computed for
a different prefix. No forward pass ever runs for the alternatives. GenTKG's
LLaMA path is already beam search, so beam search also makes the two models
symmetric.

#### F67 done — beam search in, `permute` out

`model_utils.py` rewritten: `permute`, `deduplicate` and `parse_results` are
gone, replaced by `_clean`, `beam_candidates`, `predict` and
`max_target_tokens`. The two model families now differ in exactly one integer —
`prompt_len`, 0 for T5 and `len(input_ids[0])` for LLaMA, whose output repeats
the prompt. New flags: `--num_beams` (10), `--length_penalty` (0.6),
`--max_new_tokens` (derived), `--limit` (smoke tests).

Two side fixes fell out:

- The `'gtkg' in args.finetuned_model` sniff is gone. Keeping only the first
  line is now unconditional: it is what a history-completion prompt needs, and
  it is inert on a target that has no newline, which is all of them.
- `max_new_tokens` was a hardcoded table. Under `permute` it was a recursion
  depth cap; under beam search it is a hard truncation, and truncation can cut
  a long name down onto a shorter *true* one and score a hit the model never
  proposed. It is now derived from the longest target in the split, which is
  also the only way to get it right for LLaMA's tokenizer on Colab. The old
  constants were adequate for correctness — the gold targets fit — so no
  existing result was corrupted by them.
- **F88 closed, not by retraining.** `run_hf.py` falls back to the base
  tokenizer when the checkpoint's cannot be loaded, catching `TypeError` (which
  is what stale protobuf raises, not `ImportError`). Safe because fine-tuning
  does not change a T5 tokenizer: `vocab_size` is the base 32128 and the 100
  "added" tokens are T5's standard `<extra_id_*>` sentinels. The substitution
  is logged, deferred until the logger exists so it lands in the run record.

**Verified on 300 ICEWS14 test samples.** `--num_beams 1` reproduces
H@1 = 0.38333, byte-identical to `permute` — so the rewrite is equivalent to
the code it replaced and there is no detokenization regression, despite
`permute` joining per-token `decode()` calls where the new path decodes whole
sequences. At k=10: H@1 0.343 < H@3 0.430 < H@10 0.523, **closing F84**;
fH@1 0.353 > H@1 0.343, so filtered ranking is live on T5 for the first time.
Distinct candidates per query average 9.97 (min 9), so duplicate collapse is a
non-issue and over-generating beams to backfill is unnecessary.

**`length_penalty` = 0.6, taken as a standard rather than tuned.** Beam search
ranks by total log-probability divided by `length ** length_penalty`, so it
decides rank 1; at the HF default of 1.0, H@1 fell 0.040 below greedy. A
validation sweep (300 samples, never test) gave:

| config | H@1 | H@3 | H@10 | fH@1 |
|---|---|---|---|---|
| greedy | 0.353 | 0.353 | 0.353 | 0.353 |
| k=10, lp=0.0 | 0.343 | 0.490 | 0.603 | 0.363 |
| k=10, lp=0.5 | 0.353 | 0.480 | 0.600 | 0.370 |
| **k=10, lp=0.6** | **0.347** | **0.477** | **0.600** | **0.363** |
| k=10, lp=1.0 | 0.340 | 0.447 | 0.590 | 0.353 |
| k=10, lp=2.0 | 0.313 | 0.403 | 0.587 | 0.320 |

At n=300 the standard error on H@1 is ±0.028, so 0.0/0.5/0.6 are
indistinguishable and picking between them from this data would be fitting
noise; lp ≥ 1.0 is outside it on H@3. 0.6 is the conventional seq2seq
length-normalisation value (Wu et al., 2016), so it is citable rather than
tuned — worth more in the paper than a 3-sample win. HF's parameterisation is
not GNMT's formula, so this is a convention transferred by analogy, not an
inherited constant. T5's own published value is 2.0, but only for
summarization, which rewards long outputs; it was the worst value measured.
Net cost of beam search at rank 1: **0.006, two samples of 300**, against
H@3 +0.124 and H@10 +0.247.

Still untested: **LLaMA at k=10**. The code path changed only in that it now
deduplicates and takes `num_beams` from a flag; VRAM is the risk, not logic,
since beams replicate the KV cache. First Colab run should try 10 and fall back
to `--num_beams 3` (reporting H@1/H@3 only) if it OOMs.

#### Group C done — GenTKG's answer format

Verified against their source, re-cloned rather than recalled. Their
`data_utils/TLR.py:174-178` expression transcribed verbatim and run against
`fact_to_line` over the whole fact universe: **0 mismatches in 90,730 history
lines and all 365 query timestamps**. That also settled a detail taken on trust
— their `int(time_in_id / period)` and this repo's `// period` agree on every
timestamp, so F47 is exact rather than merely equivalent-looking.

The format, from their code:

| element | form |
|---|---|
| history line | `334: [Malaysia, Engage_in_diplomatic_cooperation, 18.Vietnam] ` |
| query line | `334: [Iran, Engage_in_diplomatic_cooperation,` |
| target | `18.Thailand` |
| comparison | strip the `id.` prefix, lowercase, exact match on the name |

**The prefix is object-only.** Subjects and relations stay bare names even
though a subject in one line may be the object of another: only the position
the model has to generate is indexed. Intuition would have got this wrong.

**`--use_ids` is deleted, not renamed.** It rendered subject, relation *and*
object as bare integers, which is not a format being reported. Gone from
`TLR.py`, `retrieve.py`, `create_json_train.py` and `apply_history_filters.py`,
along with the `_ids` token and the dead `test_ans_ids` alias. The answers file
is now always names; `create_json_train.py` adds the prefix when asked, so
there is one canonical answer form on disk.

`create_json_train.py`'s `entities` parameter was not vestigial — it was the
stump left when the prefix rendering was removed. Deleting it, which was the
first instinct, would have destroyed the hook the format needs.

**Scoring.** One `normalize_entity` in `utils.py` strips a digits-only prefix
and casefolds, used by `ranks`, the fact index and beam dedup. Measured: 43
ICEWS14 and 194 ICEWS18 entity names contain a dot (middle initials such as
`Vincent_C._Siew`), and **none** begin with digits-then-dot, so the digit guard
is unambiguous and one function serves both formats. Deduplicating on the
normalized form matters: without it `18.Thailand` and `18.thailand` both
survive and one consumes a rank above the target.

Regression check: every existing bare-name results file scores identically to
before the change, so the two formats are comparable on identical code.

**Prompt prefix.** `--prompt_prefix`, off by default, LLaMA only. Applied at
run time in `run_hf.py` and `training_LLaMA.py` rather than baked into the
dataset as GenTKG does — the block sits at the front of the prompt and
`--tail_truncate_long_inputs` keeps the end, so a stored prefix would be the
first thing truncation discarded. Its token cost is reserved out of the
truncation budget (`history_limit = token_limit - prefix_tokens`), or the
wrapped prompt would exceed the model's limit. It must be set identically for
training and evaluation; both scripts report it.

**`data_utils/diff_variants.py`** asserts that an indexed and a bare history
file differ *only* by the object's id prefix, and that the answers files are
identical. Tested against five injected faults — missing prefix, altered
object, altered subject, altered day, and a float timestamp — all caught.

### F60 — the supervised span, investigated 2026-09-11 · PLANNED, NOT APPLIED

Investigated in full; no change made to `training_LLaMA.py`'s masking. Everything
below is measured, and the scripts that produced it are reproducible.

**What the `+1` does.** Nothing is truncated: both branches supervise the whole
target, 300/300 on ICEWS14. The `+1` adds exactly one token, the bare `']'`, and
only when the tokenizer did not glue it to the target's last character. The rule
is exact — 125/125 of targets ending in `)` see both branches agree, 175/175 of
all others differ, with no exceptions.

**Why that is worse than a per-dataset divergence.** The three label fixups
(`4638 ')]' -> ')'`, `28166 '))]' -> '))'`, `5586 '.]' -> '.'`) exist to *strip*
the bracket from the merged tokens. The `+1` then *re-adds* it for the unmerged
ones. So within ICEWS14 the model is taught two different stopping conventions,
chosen by whether the entity name happens to end in a parenthesis. ICEWS18 takes
the `else` branch and never supervises `']'`, including for its 8,125
bare-bracket entities.

**The fixup list is complete, for both datasets.** Over all 7,128 ICEWS14 and
23,033 ICEWS18 entities, only four tokens ever land at that position — the three
handled ids plus the bare `']'`. Zero unhandled in either. The `'.]'` case is 1
entity in ICEWS14 (`6630.Paquito_Ochoa,_Jr.`) and 20 in ICEWS18.

**The root cause is one line.** Appending `']'` to the training text is the sole
source of glued tokens: 4,221 of 7,128 ICEWS14 entities glue with it, **0**
without. The fixups, the `+1`, and the inconsistency are all downstream of it.

**Two defects found alongside.** `if response_token_ids_start_idx is None` is
unreachable — `np.where` returns an empty array, not `None`, so `[0][-1]` raises
`IndexError` first (0 occurrences in 3,000 samples, so latent). And
`labels[i,-1] = 2` assumes nothing is padded; it exists because `pad_token` is
`eos_token`, so the base collator masks the real final EOS along with padding.
Enabling evaluation at HF's default `per_device_eval_batch_size=8` would have
written EOS over a padding position in every row — `per_device_eval_batch_size=1`
is now set explicitly, and the `attention_mask`-based form is the real fix.

**The planned replacement.** Take the span from the tokenizer's character offsets
at encode time, where both halves of the text are known, instead of recovering it
by decoding and re-encoding:

```python
text = f"{context} {target}"                     # no trailing bracket
enc = tok(text, add_special_tokens=True, return_offsets_mapping=True)
start = next(j for j, (a, b) in enumerate(enc["offset_mapping"])
             if b > a and b > len(context))
input_ids = enc["input_ids"] + [tok.eos_token_id]
```

The collator then masks `[:target_start]` and restores EOS at
`int(attention_mask[i].sum()) - 1`. That deletes the decode, the
`split('\n')[-1].split(' ')[-1]` extraction, the four `replace` calls, the
re-encode, the last-occurrence search, all three fixups, the unreachable check,
and the dataset branch. It also makes the LLaMA target identical to the Flan-T5
one, and is correct at any batch size.

**EOS must be appended by id, not written into the text.** A literal `"</s>"`
parses as the special token only when a space or bracket precedes it; glued to
the answer it becomes the three tokens `'</'`, `'s'`, `'>'`. The current code
works *because* the bracket is there, so removing the bracket naively replaces
EOS with junk and the model never learns to stop — silently, with no error.

**Verified equivalent** by running both collators over the same samples and
comparing the text each leaves in the loss, bracket discounted since scoring
strips it either way:

| check | result |
|---|---|
| new span == target exactly | 3000/3000 |
| new == old `icews14` branch | 3000/3000 |
| new == old `else` branch | 3000/3000 |
| EOS retained | 3000/3000 |
| contains `']'` | 0/3000 |
| with `--prompt_prefix` (337 chars of offset shift) | 1000/1000 |
| ICEWS18, bare-name targets, old format | 3000/3000 |

So the span logic holds across both entity vocabularies, both target formats
(`id.Name` and bare), and with or without the instruction prefix.

**Do it before the first Colab LLaMA run, not after.** It changes the training
text, so it invalidates any adapter trained under the old shape. No LLaMA model
has been trained under the new pipeline yet, so the cost today is zero.

Still open: ICEWS18 has no new-format retrieval yet, so its verification used the
old-format file. The span does not depend on the history format, but the check
should be repeated once ICEWS18 is rebuilt.

### LLaMA hyperparameter search — plan, 2026-09-14

Compute-bound: every run costs Colab units, so the plan is built to spend as
few as possible before the shape of a good configuration is known. Epochs are
held at 1 throughout and raised only at the end, since they scale run cost
directly while the other parameters do not.

**What was wrong with the adapter.** `target_modules` was never set, so PEFT
applied its Llama default of `['q_proj', 'v_proj']` — the original LoRA paper's
choice, two of the seven linear layers in a decoder block, omitting `o_proj` and
the entire MLP. QLoRA's finding is that adapting every linear layer matters more
than rank for approaching full fine-tuning. At r=8 that is ~4.2M trainable
parameters against ~20M. `bias='lora_only'` was inert: Llama-2 builds its
projections without bias terms, so there were none to train. And `use_rslora`
divides by sqrt(r) rather than r, which at r=8/alpha=16 raised the update scale
from 2.0 to 5.66 on top of the learning rate — and is absent from the pinned
`peft==0.7.1`, where passing it raises `TypeError`, so the script could not have
run at all from a clean install of the requirements.

**Two decisions carry most of the saving.**

1. *Select on `eval_loss`, never on test Hits@k.* Test evaluation is beam-10
   generation over 7,371 samples with a 7B model, far more expensive than the
   training run it would be judging. It runs once, at the end.
2. *Screen at 256 training samples*, confirm the winner at 1024. Quarter cost
   per run, same number of points on the curve when `--eval_steps` is lowered
   to match.

**A third, easy to miss:** the validation split is 8,514 samples against 1,024
training samples, and it is evaluated once per `eval_steps`. Uncapped, at batch
1, validation costs roughly twenty times the training it exists to monitor.
`--max_eval_samples` defaults to 512 for this reason.

Calling one full 1024-sample run one unit:

| Stage | Runs | Cost | Decides |
|---|---|---|---|
| 0 · reference | 1 full | 1.00 | that the pipeline runs remotely at all, and the per-run wall time everything else is budgeted against |
| 1 · learning rate | 3 screening | 0.75 | `lr` in {1e-4, 3e-4, 1e-3} — largest effect, and it interacts with every other parameter, so it is fixed first |
| 2 · rank | 2 screening | 0.50 | `r` in {16, 32} against r=8, at the chosen lr |
| 3 · dropout | 0–2 screening | 0–0.50 | run **only if** the eval curve turns upward; a curve still falling at the end is a signal for more data, not more regularisation |
| 4 · confirm | 1 full | 1.00 | the best configuration at 1024, and whether the screening order survived |
| 5 · epochs | — | — | last, once the shape is settled |

Roughly 3.25–3.75 units for seven or eight experiments.

`--lora_alpha` defaults to `2 * lora_r`, holding the update scale at 2.0 so that
changing rank changes capacity alone. Changing both at once would confound them.

**Risks, stated in advance.** `eval_loss` is a proxy for Hits@k, not the same
quantity: before trusting the ladder, run test Hits@k on the best two
configurations from stage 4 and confirm the ordering agrees. If it does not, the
proxy is invalid and the plan needs rebuilding rather than extending. Screening
at 256 samples may also rank configurations differently from 1024 — optimal
learning rate in particular tends to move with dataset size — which is what
stage 4 exists to catch; if the winner fails to reproduce, screen at 512 instead.

The controller helps here: `patience=2` stops plateaued runs early, so a bad
configuration costs less than a good one.

### Phase 4 — remaining groups

This is the expensive phase and the one that makes the comparison defensible.
Everything here forces the same re-run, so it must go together.

| ID | Action |
|---|---|
| ~~F53~~ | **Done** — `--index_target` renders objects and targets as `id.Name` |
| ~~F47~~ | **Done** — integer day, and one renderer for all three call sites |
| ~~F67~~ | **Done** — beam search, k=10, `length_penalty` 0.6 |
| F41 + F49 | Pass `retrieve_type` into `Retriever`; extend `prepare_bs` to `(col_sub == sub) \| (col_obj == sub)` → the `standard` baseline. Still open; `--retrieve_type bs` now **fails loudly** instead of silently running `build_tl`. Note `build_bs` has no `save_metadata` branch, so it needs one before it can feed the derive-from-base workflow |
| ~~F15~~ | **Done** — `--eval_file_path`, controller, best-checkpoint selection. Tail truncation deliberately **not** added: measured at `num_facts=50` with the instruction prefix, the longest prompt is 3,136 LLaMA tokens against a 4,095 cut, and 0 of 74,845 training samples are over it. The filter now reports its drop count so the assumption is checked at every run, not assumed |
| ~~F59~~ | **Done** — the stride is gone; the subset comes from `create_json_train.py`'s seeded `rng.sample` via `<name>_1024.json`. The old `round(n / 1024)` collapsed to "first 1024 in file order" for `1024 < n < 1536`, and could silently return fewer than 1024 (at n=2,000 it returns 1,000) |
| F66 | Set `weight_decay` explicitly for LLaMA |
| F60 | Verified on ICEWS14 and ICEWS18; replacement designed and proven equivalent. See the section above — **apply before the first Colab LLaMA run** |
| ~~—~~ | **Done** — time-aware filtered Hits@k is live in `run_hf.py` and in the offline `compute_metrics_from_results.py`, with the resolved-share guard so filtered figures are never reported when the fact index missed the queries |
| ~~F19~~ | **Done** — `cs` is gone from `utils.py` and from every scoring path. The character-frequency cosine survives only in `rag_with_gpt_4_1.py`, where it snaps a generated string to a known entity rather than being reported as a metric; crude for that job but not a published claim |
| F20 | Verify BERTScore with `rescale_with_baseline=True` before reporting it |

### Stage separation — retrieval vs. filtering · **FIXED 2026-09-11**

`--index_target` was accepted by `retrieve.py --save_metadata`, recorded in the
metadata header, and then read back as authoritative by
`apply_history_filters.py:283`. Three consequences, all now closed:

1. **A base directory was named `_idn` while holding no rendered history.** In
   `--save_metadata` mode retrieval writes no `history_facts/` at all, so the
   token described a rendering that had not happened. The six ICEWS14 bases have
   been renamed `*_inv_idn` → `*_inv`.
2. **The format could not be chosen when deriving.** The header was the only
   source, so one base could serve only one format — despite the metadata
   storing *fact indices*, which carry no format at all. The bare-name control
   run appeared to need a second full retrieval (~22 min); it needs none.
3. **`--retrieve_type` was silently inert.** F41: the flag is read and logged but
   never passed to `Retriever`, so `build_tl` always runs. Any value other than
   `'bs'` selected it anyway, so a typo was undetectable. The default string
   `TLogic-3` was itself never parsed.

**Changes.** `retrieve.py` adds `--index_target` to the `--save_metadata`
rejection list, constrains `--retrieve_type` to `choices=["TLogic", "bs"]`, and
fails with an explanation if `bs` is requested while F41 is open. `TLR.py` drops
`index_target` and `model_type` from the metadata header, leaving retrieval-time
state only. `apply_history_filters.py` gains `--index_target` (default off),
defaults `model_type` in argparse instead of the header, records `index_target`
in `filter_stats.json`, and resolves `test_answers` from the metadata's own
directory rather than the header's machine-specific absolute path — which the
rename would otherwise have broken. `diff_variants.py` now recognises query
lines, which have no object and so must be byte-identical across variants.

**Verified.** Re-deriving `gtkg/test_inv_n50_idn` from the renamed base is
**byte-identical** to the pilot's existing file, for both the history and the
rule ids. The bare derivation differs in exactly the intended way: 288,656
history lines all correctly prefixed, 7,371 query lines identical, answers files
identical. All three new guards reject their bad invocations.

**Standing policy.** One base per `(dataset, rule bank)`:
`retrieve.py -d <ds> -m <bank> --length_1_only --inverse_body_object_match
--save_metadata`. `num_facts`, `top_k_rules`, `confidence_threshold` and
`index_target` are derived from it. `inverse_body_object_match`,
`early_stop_at_num_facts`, the bank and `retrieve_type` cannot be — each changes
which facts are retrieved. `rule_length_all` is not an axis: `TLR.py:121-134`
rejects any bank with multi-hop rules because `build_tl` anchors the last body
relation at the query subject instead of the intermediate entity, so the flag
can only be set on a bank where it does nothing. Always pass `--length_1_only`.

Note: `filter_stats.json` in the pre-existing derived directories records a
`metadata` path ending `_inv_idn`, which no longer exists. The stale
`index_target` and `model_type` keys also remain in the six base header lines
and are now ignored rather than read.

### Phase 5 — pilot run (ICEWS14, `gtkg`, `id.Name`, seed 1)

One narrow slice taken end to end before committing to the full grid.

**Stage timings, measured.** Retrieval `gtkg` **21.7 min** (train 17.5, test 1.9,
valid 2.2); retrieval `exhaustive` **24.5 min** — barely slower despite 3.5x the
rules, because cost is dominated by fact scanning, not bank size, so ICEWS14
retrieval is ~22-25 min per miner regardless of bank. Filtering all three
splits: **6 s**. Training 3 epochs at batch 8: **1.89 h**. Evaluation at k=10:
**32 min**.

**Results.** raw H@1 .3397 / H@3 .4489 / H@10 .5276; filtered .3506 / .4522 /
.5276. Queries resolved against the fact index **7371/7371 = 100%**, which is
what proves `normalize_entity` strips the id prefix so filtered ranking still
works under `--index_target`. Distinct candidates per query 9.7 (min 3).
`max_new_tokens` derived as 44, against 32 for bare names — the id prefix
lengthens targets and the derivation adapted on its own.

For scale: GenTKG report filtered 36.85 / 47.95 / 53.5 for **LLaMA-2-7B**. This
is flan-t5-small at 77M parameters, within ~1.8 points of H@1 on a now-
comparable protocol. One seed against their 3-seed mean, and a different bank.

#### Training batch size: 8, and `group_by_length` does nothing here

`per_device_train_batch_size` was 2, commented "fixed — hardware limit on this
laptop". It is not: flan-t5-small is 77M parameters. Measured on the 6 GiB card
with worst-case batches (every sample at the 511-token limit), since the
collator pads to the longest member and memory follows that, not the mean:

| batch | peak allocated | samples/s |
|---|---|---|
| 2 | 1.62 GiB | 11.7 |
| 4 | 2.12 GiB | 23.1 |
| **8** | **3.25 GiB** | **33.9** |
| 16 | 5.57 GiB | 32.9 |
| 32 | 10.14 GiB | 6.0 |
| 64 | out of memory | |

Three things worth keeping:

- **Batch 32 did not fail, it paged to host RAM.** 10.14 GiB "allocated" on a
  6 GiB card is only possible because the Windows driver spills silently, which
  is why the step time collapsed by 22x. On this platform "it ran" is not
  evidence that a batch size fits.
- **`max_memory_allocated` understates device usage by ~1.6 GiB** (CUDA context
  plus allocator reserve). The real run sits at 4.9 of 6.1 GiB at batch 8, so
  batch 16's 5.57 GiB allocated would have meant ~7.2 GiB and a genuine OOM in
  training. Batch 8 is the ceiling, not merely the throughput optimum.
- **Padding is inert above batch 1**, verified rather than assumed:
  `DataCollatorForSeq2Seq` pads inputs with `<pad>` (id 0, not EOS), zeroes the
  attention mask there, pads labels with -100, and builds `decoder_input_ids`
  itself. The same sample's loss alone versus inside a padded batch differs by
  9.5e-07.

`group_by_length=True` was expected to matter more than the batch size, since
over half of every shuffled batch is padding. It delivered nothing: the real run
sustained 33.1 samples/s against 35.2 measured for shuffled batches and 55.1 for
globally length-sorted ones. HF groups only within megabatches of
`batch_size * 50`, which is far weaker than sorting, and the 55.1 figure came
from too few batches to be reliable. Left enabled — harmless, and ICEWS18's
length distribution differs — but not counted in any estimate. The honest
speedup is batch 2 to 8 alone: 3 epochs on ICEWS14 fell from ~5.3 h to
**1.89 h measured**, not the 1.13 h projected from the length-sorted number.

#### Two bugs the pilot caught, both of which would have hit every grid run

**The trained model was never saved.** `print(controller.summary())` sat between
`trainer.train()` and `trainer.save_model()`. The summary contained a `→`, the
Windows console encodes stdout as cp1252, the print raised, and 1.89 h of
training reached no artifact — after a log that looked like a complete success.
The unencodable character was the trigger; the ordering was the defect. Model,
tokenizer and trainer state now persist first and the summary prints after.
Nothing cosmetic belongs between the end of training and the model reaching
disk. `training_LLaMA.py` was never exposed: it saves immediately after
`trainer.train()`. The run itself was recovered from the best checkpoint, which
was also the last, so nothing was retrained.

**`create_json_train.py` rejected every split.** `content.split('

')` on a
history file that ends with the sample separator leaves a trailing empty chunk,
so the strict alignment check read one more context than answers. The assertion
this file recommended was `len(inputs) - 1 == len(test_ans)`; it had been
implemented without the `- 1`. Fixed by dropping the trailing chunk, which is
provably safe: there is exactly one empty chunk per file, always last, and a
sample with no retrieved facts is a *single-line* chunk holding just its query
line — 5,689 / 211 / 272 of them, matching the empty-`fired_rules` counts in the
metadata exactly. Upstream never noticed because GenTKG's loop iterates the
answers and indexes into the contexts, so it never visits the extra chunk.

#### Why the raw H@1 is .340 where an older run reached .369

The old ICEWS14 `gtkg` Flan-T5 runs span **.328-.369**; .369 is the best of
about twenty, not a typical figure. Five things differ between it and the pilot,
and only some are separable.

| candidate | verdict |
|---|---|
| training-data regime (F3, the 30% short-history subset) | **ruled out** — the `.369` model trained on the full split with tail truncation. The 15-point collapse to .209 under eval-time truncation belongs to the **non-`fancy`** model, which is the one F3 describes |
| decoding, greedy to beam k=10 at lp 0.6 | **measured: .3472 greedy vs .3397 beam, so 0.0075.** The cost buys H@3 and H@10, which a single candidate cannot express at all |
| eval-time truncation | leading remaining suspect. The `.369` file predates the `_tail_truncate` suffix, so its test-time flag is unrecoverable. Tail truncation keeps ~14 of 50 facts on 83% of test prompts |
| rule bank (F48-fixed mining, seed 12) | not separable without retrieving from the old bank |
| answer format, bare vs `id.Name` | needs the bare-name control run |

Note the asymmetry when interpreting a no-truncation result: it feeds T5 inputs
up to ~2,268 tokens when it was fine-tuned on 511, and per F82 everything past
128 tokens of distance shares one relative-position bucket, so the model has no
positional resolution over most of that history. It is also ~8x slower to
decode. If the untruncated configuration scores higher, the honest reading is
that more evidence helps even without positional resolution over it, and the
paper should report the consistent train/test configuration with the other as
an ablation.

#### `exhaustive` buys coverage, not volume

At `--num_facts 50` the cap binds, so a 3.9x larger bank cannot add history to
samples that already reach 50 facts. Measured against `gtkg` on ICEWS14:
history lines rise only **1.04-1.07x**, while samples with **no** history fall
from 7.6% to 6.5% (train), 2.9% to 2.2% (test), 3.2% to 2.5% (valid). The
larger bank rescues the queries no rule fired for; it does not enrich the rest.

### Phase 5 — the experiment grid

Per dataset (ICEWS14, ICEWS18) and per miner (`gtkg`, `exhaustive`):

1. `retrieve.py --inverse_body_object_match --index_target --save_metadata` — uncapped, once.
2. `apply_history_filters.py --num_facts 50` — the training variant.
3. `create_json_train.py` for train / valid / test.
4. Train Flan-T5 (**3 seeds**, per F63) and LLaMA2-7B.
5. Test with `--tail_truncate_long_inputs` matched to training, `--num_beams 10`.
6. Score raw **and** filtered — both come from the same run.

Plus the `raw` baseline per dataset, once each (`standard` dropped — ISI's published numbers are cited instead). No
`--confidence_threshold`, no `--top_k_rules`, no `--early_stop_at_num_facts` in
the headline runs (F35, F42). `--top_k_rules` becomes a separate ablation on
`exhaustive` only.

### Paper-only — no code, no runs

| ID | What to write |
|---|---|
| F36 | The superset argument, stated as a proof rather than a table |
| F34 | "all type-compatible length-1 rules with positive confidence" — 62,156 candidates → 41,068 kept |
| F39 | Relation-signature labelling, a data-driven proxy for entity type — not entity typing |
| F35 | Confidence is exact for `exhaustive`, sampled for `gtkg`; headline runs use no confidence filter |
| F12 | The regime table: Flan-T5 74,845 / 3 epochs vs LLaMA 1,024 / 1 epoch |
| F13 | Describe `--eval_file_path` as best-checkpoint selection, not adaptive LR scheduling |
| F17/F18 | ~~H@1 only for Flan-T5~~ — superseded by F67. Both models now rank 10 candidates, so H@1/H@3/H@10 are reportable for both, raw and time-aware filtered |
| F46 | History is drawn from the full corpus up to the query timestamp |
| F8 | Same-timestamp facts are excluded by design |
| F23 | Scoring is exact string match |
| F63 | Report mean ± std over seeds; the single-run spread is ±0.004. Rule banks are reproducible given seed **and** process count — worker seeding is shared, so `-p` changes the bank (F38b fixes this in Phase 3) |
| **F82** | T5 has no absolute position embeddings and `relative_attention_max_distance = 128`, while effective inputs average 456 tokens. Beyond 128 tokens of distance all relative positions collapse into one bucket, so the model has no positional resolution over most of a history and perceives recency only through the literal timestamp strings. Use this to justify the `num_facts` cap as a relevance measure, not merely a length one |
| **F84** | Closed by F67. Worth a methods sentence: the earlier decoder returned one candidate, so H@3 and H@10 were undefined rather than equal to H@1 |
| **F20** | Report BERTScore **rescaled** (0.4116 on the reference file), not raw (0.9007). Raw sits just above a ~0.83 random-pairing floor. Also note that BERTScore was calibrated on sentences, not 1–3 token entity names, so it is a weak metric here regardless |
| §4b/§4c | Comparability disclaimer for GenTKG / ISI / RECIPE-TKG: filtered vs raw, `id.Name` vs lexical, and ISI's candidate set is a history-derived shortlist |
| F24-F27, F31, F32 | If the RAG experiment is repeated: report the selected subset as an explicitly hard subset, match the history source to the model, and apply the instruction to all samples or none |

### Deliberately not doing

| ID | Why |
|---|---|
| F1 | `ragtkgc` is dropped from the paper; F48 fixes the cause anyway |
| F2 | Behaviour is already correct with the flag; note only |
| F11, F54 | `--use_ids` is not in the plan |
| F58 | Correct at batch size 1; comment only |
| F10 | Superseded by F53 — the parameter gets wired up, not removed |
| F22 | The partial o4-mini file is not reported; F79 now skips it automatically |
| F77 | Verified correct — the offline scorer matches the live metric |
| F62 | Refuted by measurement; see §0 |
| F72 | Rejected as written; the tokenizer stays authoritative. See §0 |
| F81 | Legacy-data artifact only — every set is being regenerated. See §0 |

---

## 8. Pipeline reference

1. **Rules learning** — `data_utils/rules_learning/learn.py`.
   In: `processed_new/{ds}/` (via `Grapher`) + `original/{ds}/` (via
   `get_unique_quads_per_rels`). Out:
   `processed_new/{ds}/output/{ds}/{ts}_r[..]_n..._rules.{json,txt}`,
   `common_rule_pool.json`, `node_labelling_stats_{ds}.txt`.
   Timestamped filenames do **not** record the algorithm — read `found_by`.
2. **History retrieval** — `data_utils/retrieve.py` → `TLR.Retriever`.
   In: rule bank + `original/{ds}/`. Out:
   `processed_new/{ds}/{mining}/{suffix}/history_facts/*`, `test_answers/*`,
   optional `metadata/*.jsonl`; stats to `logs/`.
   Processes train, test and valid in one invocation (hardcoded).
3. **Post-hoc filtering** — `data_utils/apply_history_filters.py`. Re-applies
   `num_facts` / `top_k_rules` / `confidence_threshold` to saved metadata in
   seconds. `--stats_only` simulates without writing the dataset.
4. **Training files** — `data_utils/create_json_train.py`. Splits the history
   file on `\n\n`, pairs chunk *i* with answer line *i*, emits
   `{"context", "target", "rule_ids"}`. Also always writes a `_16.json` subset
   (`--nums_sample` default 16, sampled without a seed).
   `--dataset` is only the output filename.
5. **Fine-tuning** — two separate, non-equivalent paths.
   - `training_T5.py`: full fine-tuning of `google/flan-t5-small`. 3 epochs,
     lr 3e-4 cosine with 6% warmup, weight decay 0.1, batch 2, bf16,
     `save_strategy="epoch"`, `save_total_limit=3`. `--eval_file_path` enables
     per-epoch eval, the (inert, F13) controller and `load_best_model_at_end`
     (F14). `--tail_truncate_long_inputs` decides clip-vs-delete (F3).
   - `training_LLaMA.py`: QLoRA on `TheBloke/Llama-2-7B-fp16` — 4-bit nf4,
     LoRA r=8 alpha=16 dropout=0.05 `bias='lora_only'` `use_rslora=True`.
     1 epoch, lr 3e-4, batch 1 × grad-accum 8, 20 warmup steps. Loss is masked
     to the target entity only via `DataCollatorForCompletionLM` (F16). Training
     set is capped at 1024 samples (F12). No eval, no truncation option (F15).
6. **Testing** — `run_hf.py`, args in `utils.py`, decoding in `model_utils.py`.
   Results to
   `results/{ds}/{model}_{dataset}_{tail_truncate|no_tail_truncate}.jsonl`,
   one JSON line per sample with `timestamp/entity/relation/targets/direction/
   predictions` (the prompt is **not** stored). Metrics `hit1/hit3/hit10/cs`
   logged at the end and to `logs/<same name>.log`.
   Decoding: Flan-T5 uses greedy generation plus `permute(dec_cand=1)` → **one**
   candidate (F17); LLaMA uses `num_beams=3, num_return_sequences=3` → **three**
   (F18). `max_new_tokens` is dataset/model specific (`model_utils.py:61-69`):
   Flan-T5 42/45, LLaMA 26/35 for ICEWS14/18. ICEWS18 test is truncated to the
   first 10000 samples.
7. **Metrics** — `compute_metrics_from_results.py` (H@1, H@3 only; reads the
   first 7371/10000 lines; uses `eval()` per line), `bertscore.py` separately
   (top-1, un-rescaled). `cs` is character-frequency cosine, not semantic (F19).
8. **GPT-4.1 RAG** — `rag_with_gpt_4_1.py`. Selects the samples most models got
   wrong (from a hardcoded 8-file list, F31), asks GPT-4.1 for a plausible
   history in one of three prompt styles (`gpt-given-rules`,
   `gpt-given-relations`, `gpt-rule-miner`), snaps unknown entities/relations to
   the nearest vocabulary item by character-frequency cosine (or MiniLM with
   `--use_llm_similarity`), and writes `test_rag/{ds}_{version}.json` plus an
   index of modified samples. `run_hf.py --dataset_rag_path` then substitutes
   only those indices at test time. History source is hardcoded to **ragtkgc**
   (F24). One API call per sample.
</content>
</invoke>
