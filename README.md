# RAGTKGC
This is the official implementation of the paper **RAGTKGC: Undertaking Temporal Knowledge Graph Completion with Retrieval Augmented Generation**

**IF YOU CANNOT CLONE THE REPO, PLEASE DOWNLOAD IT FROM HERE --> https://drive.google.com/file/d/1Re-H4k78tfqoC9D_hzHjmObIGL7oChMj/view?usp=drive_link**

**Important Note! The code is written to run locally. However, the fine tuning part is also provided as a Jupyter Notebook, so you can move it on Google Colab if you don't have enough resources. Any part that is moved on Colab requires you to change the paths by hand, as Google Colab starts looking for files from /Content/...**

The short guide contains mostly the commands to run -> <a href = "#SG"> Short Guide </a>
The extended guide has explanations for each part -> <a href = "#EG"> Extended Guide </a>

## Naming conventions

Every artifact carries the identity of the experiment that produced it, so a
results file can be traced back to a rule bank without consulting a notebook.
The identity is a **variant token** assembled from the retrieval options:

```
variant = [_inv][_n50][_k10][_ct0.5-t5][_es][_idn]
run     = {dataset}_{mining}{variant}_{split}
```

| Token | Set by | Stage | Means |
|---|---|---|---|
| `_inv` | `--inverse_body_object_match` | retrieve | inverse body relations matched on fact object == query subject |
| `_es` | `--early_stop_at_num_facts` | retrieve | stop consulting rules once `num_facts` facts exist |
| `_n50` | `--num_facts 50` | filter | at most 50 history facts per sample |
| `_k10` | `--top_k_rules 10` | filter | only the 10 highest-confidence fired rules |
| `_ct0.5-t5` | `--confidence_threshold 0.5 --model_type t5` | filter | confidence split, rendered for t5 |
| `_idn` | `--index_target` | filter | objects rendered `id.Name`, e.g. `18.Thailand` |

Absent options contribute nothing, so the unfiltered base is just `_inv`. The
tokens are deliberately short: they appear in the directory name, the training
file name and the results file name, and Windows caps a path at 260 characters.

**Stage** says which script owns the token: `retrieve` options change which
facts are retrieved and are fixed in the base, `filter` options are derived from
it by `apply_history_filters.py`. A base variant therefore never carries a
`filter` token — `retrieve.py --save_metadata` refuses those flags outright.

### Prompt and answer format

A history line and its query are rendered as:

```
334: [Malaysia, Engage_in_diplomatic_cooperation, 18.Vietnam] 
334: [Iran, Engage_in_diplomatic_cooperation,
```

The day is an integer. Under `--index_target` the **object** carries its entity
id as an `id.Name` prefix, and so does the training target; subjects,
relations and the answers file on disk stay bare names. The prefix is
object-only because that is the one position the model has to generate. This
matches the format the LLaMA baseline (GenTKG) publishes, so its reported
numbers are comparable; without the flag, objects are bare names.

Scoring is indifferent to the choice: predictions and targets are compared
after stripping a digits-only prefix and lowercasing, so both variants score on
the same code. `data_utils/diff_variants.py` checks that an indexed and a bare
history file differ only by that prefix.

How the token flows through the pipeline:

| Stage | Artifact |
|---|---|
| mine | `output/{ds}/{timestamp}_{mining}_r{lengths}_n{walks}_{distr}_s{seed}_rules.json` |
| | `output/{ds}/{timestamp}_exhaustive_rules.json` (no walk parameters apply) |
| retrieve | `{ds}/{mining}/{split}_inv/metadata/history_metadata_{ds}_{split}.jsonl` |
| filter | `{ds}/{mining}/{split}_inv_n50/history_facts/history_facts_{ds}.txt` |
| jsonify | `{ds}/{mining}/{split}_inv_n50/json/{ds}_{mining}_inv_n50_{split}.json` |
| train | `models/{base}_{ds}_{mining}_inv_n50_s{seed}/` |
| test | `results/{ds}/{model}_{ds}_{mining}_inv_n50_test_tail_truncate.jsonl` |

A worked example, ICEWS14 with the `gtkg` miner and a 50-fact cap:

```
090926143210_gtkg_r[1]_n200_exp_s1_rules.json
data/processed_new/icews14/gtkg/train_inv_n50/json/icews14_gtkg_inv_n50_train.json
models/flan-t5-small_icews14_gtkg_inv_n50_s1/
results/icews14/flan-t5-small_icews14_gtkg_inv_n50_s1_icews14_gtkg_inv_n50_test_tail_truncate.jsonl
```

## <p name = 'SG'>Short Guide</p>

### Create a Conda Env

We advise you to create a new environment for our code, but you can also run it on already existing ones or Colab, just make sure to install requirements and torch as provided below:
```
git clone https://github.com/AnonOctopus/RAGTKGC
cd RAGTKGC

conda create -n ragtkgc python=3.9.21
conda activate ragtkgc

pip install -r requirements.txt 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### History Modeling

Code based on https://github.com/mayhugotong/GenTKG. Many thanks for their great contribution!

#### Rules learning
It works with files from RAGTKGC/data/processed_new/ . 
You can produce other rule banks besides the provided ones by running e.g. for icews14:

```
cd data_utils/rules_learning
python learn.py -d icews14 -l 1 -n 200 -p 15 -s 12 -m gtkg
python learn.py -d icews14 -m exhaustive
```

You will get a rule bank named `{timestamp}_{algorithm}_r{lengths}_n{walks}_{distr}_s{seed}_rules.json` under RAGTKGC/data/processed_new/{dataset_name}/output/{dataset_name}, e.g. "090926143210_gtkg_r[1]_n200_exp_s1_rules.json". `exhaustive` neither walks nor samples, so its bank records no walk parameters: "090926143512_exhaustive_rules.json". The algorithm is part of the name because `retrieve.py` selects the bank by it.

#### History retrieving

Find the file name of rule bank json (in RAGTKGC/data/processed_new/{dataset_name}/output/{dataset_name}) and run from the folder RAGTKGC/data_utils:
```
python retrieve.py --dataset icews14 -m gtkg --length_1_only --inverse_body_object_match --save_metadata
```
The bank is chosen automatically by matching `_{mining}_` in its filename; the run is refused if no bank, or more than one bank, matches. Pass `-r` to override, which the `raw` and `standard` baselines need since they have no bank of their own:
```
python retrieve.py --dataset icews14 -m exhaustive --length_1_only --inverse_body_object_match --save_metadata
```

#### Training files
For training, you need to convert history_facts files into json file:
```
python ./data_utils/create_json_train.py --dir_of_trainset 'the_full_trainset_to_convert (see [A])' --dir_of_answers 'the_test_answers (see [B])' --dir_of_entities2id 'the_json_of_entities2id' --path_save 'recommend_the_same_split_folder_as_the_one_converted' --name_train {dataset}_{mining}{variant}_{split}
```
Position on the RAGTKGC root folder. An example for the icews14 gtkg train split would be like:
```
python data_utils/create_json_train.py \
  --dir_of_trainset data/processed_new/icews14/gtkg/train_inv_n50/history_facts/history_facts_icews14.txt \
  --dir_of_answers data/processed_new/icews14/gtkg/train_inv_n50/test_answers/test_answers_icews14.txt \
  --dir_of_entities2id data/original/icews14/entity2id.json \
  --path_save data/processed_new/icews14/gtkg/train_inv_n50/json \
  --name_train icews14_gtkg_inv_n50_train --nums_sample 1024 --seed 1
```

You may do it for other splits too (test and valid), especially "test" as it will be needed for testing. Just write their name instead of "train" in the provided example.

### Fine tuning models

If you want to change any hyperparameter, please update them manually in the desired training file.
**Each run writes to `./models/{--trained_model_name}/`, holding the final model and that run's `checkpoint-*` directories. No renaming is needed. Name the model `{base}_{dataset}_{mining}{variant}_s{seed}`, e.g. `flan-t5-small_icews14_gtkg_inv_n50_s1`; the seed belongs in the name because the reported numbers are a mean over seeds. Checkpoints are per-run because Trainer names them by optimiser step, so two runs over the same training-set size would otherwise overwrite each other.**

#### Flan-T5-Small

To fine tune a Flan-T5-Small model, run the following command:
```
python training_T5.py --dataset name_of_dataset --trained_model_name 'name_of_fine_tuned_model' --train_file_path "path_to_the_training_file"
```

An example for icews18 would be:
```
python training_T5.py --dataset icews14 --seed 1 \
  --trained_model_name 'flan-t5-small_icews14_gtkg_inv_n50_s1' \
  --train_file_path "gtkg/train_inv_n50/json/icews14_gtkg_inv_n50_train.json"
```
#### LLaMA2-7B

To fine tune (QLORA) a LLaMA2-7B model, run the following command:
```
python training_LLaMA.py --dataset name_of_dataset --trained_model_name 'name_of_fine_tuned_model' --train_file_path "path_to_the_training_file"
```

An example for icews18 would be:
```
python training_LLaMA.py --dataset icews14 --seed 1 \
  --trained_model_name 'llama-2-7B_icews14_gtkg_inv_n50_s1' \
  --train_file_path "gtkg/train_inv_n50/json/icews14_gtkg_inv_n50_train_1024.json"
```

### Testing models

Code based on https://github.com/usc-isi-i2/isi-tkg-icl. Many thanks for their great contribution!

To test any model, you need to run from RAGTKGC folder:
```
python run_hf.py --base_model "the base model" --finetuned_model "fine tuned version of base model" --dataset "dataset name" --dataset_path "path to test file"
```

An example for testing a LLaMA2-7B-ragtkgc LORA version on icews18 dataset:
```
python run_hf.py --base_model "TheBloke/Llama-2-7B-fp16" \
  --finetuned_model "llama-2-7B_icews14_gtkg_inv_n50_s1" --dataset "icews14" \
  --dataset_path "gtkg/test_inv_n50/json/icews14_gtkg_inv_n50_test.json" --tail_truncate_long_inputs
```

The results file is named `{finetuned_model}_{dataset_path basename}_{tail_truncate|no_tail_truncate}.jsonl`, so it records both the model and the evaluation set. Those are independent axes: evaluating one variant's model on another variant's data is a legitimate experiment, so both belong in the name.

An example with RAG:
```
python run_hf.py --base_model "TheBloke/Llama-2-7B-fp16" \
  --finetuned_model "llama-2-7B_icews14_gtkg_inv_n50_s1" --dataset "icews14" \
  --dataset_path "gtkg/test_inv_n50/json/icews14_gtkg_inv_n50_test.json" \
  --dataset_rag_path "test_rag/icews14_gpt_given_rules.json"
```

Candidates come from beam search: `--num_beams` (default 10) is both the beam width and how many candidates get ranked, so H@k needs `--num_beams >= k`. `--length_penalty` (default 0.6) decides how candidates of different token length compare, and therefore what rank 1 is. See the Extended Guide for why those defaults.

#### Calculating BERTScore

It can be run using the following command:
```
python bertscore.py --results_file "name_of_results_file"
```
An example can be:
```
python bertscore.py --results_file "icews18/llama-2-7B-icews18-ragtkgc_icews18_ragtkgc_test.jsonl"
```

#### Compute metrics from results files

You can compute the metrics for already saved predictions from the results folder, by running:
```
python compute_metrics_from_results.py --dataset "name_of_dataset" --file_name "name_of_the_results_file.jsonl"
```

An example for icews14 is:
```
python compute_metrics_from_results.py --dataset "icews14" --file_name "flan-t5-small-icews14-ragtkgc_icews14_gpt_rule_miner.jsonl"
```

### RAG with GPT 4.1

**You need to save you own OpenAI key in api_key.txt file!**
To obtain RAG-enhanced input prompts, you can use the following command:
```
python rag_with_gpt_4_1.py --dataset "dataset_name" --rule_file "name_of_the_rule_bank.txt" --rag_version "desired_rag_version"
```

An example for icews14:
```
python rag_with_gpt_4_1.py --dataset "icews14" --rule_file "080525131706_r[1]_n200_exp_s1_rules.txt" --rag_version "gpt-given-rules"
```

## <p name = 'EG'>Extended Guide</p>

### Create a Conda Env

We advise you to create a new environment for our code, but you can also run it on already existing ones or Colab, just make sure to install requirements and torch as provided below:
```
git clone https://github.com/AnonOctopus/RAGTKGC
cd RAGTKGC

conda create -n ragtkgc python=3.9.21
conda activate ragtkgc

pip install -r requirements.txt 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### History Modeling

Code based on https://github.com/mayhugotong/GenTKG. Many thanks for their great contribution!

In the paper, we test four ways of modeling the history:
1. raw, where no history is retrieved
2. standard, where we select quads that contain the same subject as the target one (see https://github.com/usc-isi-i2/isi-tkg-icl for more details)
3. gtkg, where history is retrieved using the standard method + temporal logical rules that were mined from the given TKG (train split) (https://github.com/mayhugotong/GenTKG)
4. ragtkgc, our approach which modifies gtkg by considering all possible paths (i.e. combination of edges) between the subject and object for any fixed relationship during rules mining.

For 1 and 2, the datasets are already provided and is no need to generate them again (they will always result in the same input prompts).
For 3 and 4, we also provide the datasets that we have used for experiments. Additionally, you can also generate your own versions by following the instructions from below.

#### Rules learning
It works with files from RAGTKGC/data/processed_new/ . 
You can produce other rule banks besides the provided ones by running e.g. for icews14:
```
cd data_utils/rules_learning
python learn.py -d icews14 -l 1 -n 200 -p 15 -s 12 -m gtkg
python learn.py -d icews14 -m exhaustive
```
Rules learning parameters:
- **-d** **--dataset**, dataset name.
- **-l** **--rule_lengths**, length of the mined rules, by default it is 1.
- **-n** **--num_walks**, number of times the mining algorithm starts extracting rules from a sampled quadruple.
- **--transition_distr**, how to transition from an edge to another; default: `exp`, other choices are `unif` and `exp_scaled`.
  - `exp` weights a candidate edge by `exp(ts - cur_ts)`, which is the published form. It assumes consecutive timestamps differ by about 1; in this repo's `ts2id` a day is 24, so a one-day gap costs `exp(-24)` and the distribution collapses onto the newest admissible timestamp, dropping to uniform whenever every candidate is more than ~32 days old.
  - `exp_scaled` divides the gap by the smallest timestamp spacing present in the data, restoring a graded decay. It is the same expression on day-indexed input, so it changes nothing there.
  - `-n`, `-s`, `-p` and `--transition_distr` describe walking only, so they do not apply to `exhaustive`, which enumerates every type-compatible length-1 body deterministically. They are omitted from its bank filename.
- **-p** **--num_processes**, number of parallel processes, for accelerating. 
- **-s** **--seed**, for reproduction purposes.
- **-m** **--mining**, the rule mining approach, default: ragtkgc; choices are `gtkg` (TLogic-style temporal random walks), `exhaustive` (every type-compatible length-1 rule), `ragtkgc` and `ragtkgc_no_walks` (our algorithm, the latter without walks -- one pass per unique quadruple). The name is recorded in the bank filename and in the rule pool's `found_by`.

You will get a rule bank named `{timestamp}_{algorithm}_r{lengths}_n{walks}_{distr}_s{seed}_rules.json` under RAGTKGC/data/processed_new/{dataset_name}/output/{dataset_name}, e.g. "090926143210_gtkg_r[1]_n200_exp_s1_rules.json". `exhaustive` neither walks nor samples, so its bank records no walk parameters: "090926143512_exhaustive_rules.json". The algorithm is part of the name because `retrieve.py` selects the bank by it.

#### History retrieving

Find the file name of rule bank json (in RAGTKGC/data/processed_new/{dataset_name}/output/{dataset_name}) and run from the folder RAGTKGC/data_utils:
```
python retrieve.py --dataset icews14 -m gtkg --length_1_only --inverse_body_object_match --save_metadata
```
The bank is chosen automatically by matching `_{mining}_` in its filename; the run is refused if no bank, or more than one bank, matches. Pass `-r` to override, which the `raw` and `standard` baselines need since they have no bank of their own:
```
python retrieve.py --dataset icews14 -m exhaustive --length_1_only --inverse_body_object_match --save_metadata
```

History retrieving parameters:
- **--d** **--dataset**, dataset name; default: icews14
- **-t** **--retrieve_type**, which retrieval algorithm to run; choices are `TLogic` (rule-guided, the default) and `bs` (entity-only baseline — every fact sharing the query subject, newest first). **`bs` is currently unreachable**: `retrieve.py` never passes `retrieve_type` to `Retriever`, so `build_tl` always runs. Passing it now fails with an explanatory message rather than silently running the rule-guided path. (The former default was the string `TLogic-3`, whose `-3` was never parsed by anything.)
- **-r** **--name_of_rules_file**, the name of the file where the rules are stored.
- **--length_1_only**, flag (takes no value) restricting retrieval to rules whose body has a single relation. Off by default, i.e. all rule lengths are admitted. Multi-hop retrieval is **not implemented** — only the last body relation is matched, and it is anchored at the query subject rather than at the intermediate entity — so a rule bank containing longer rules is rejected with `NotImplementedError` unless this flag is passed. All current banks are length-1 only, so the default is safe. (Replaces the former `-l`/`--rule_length_all`, which was declared `type=bool` and could not be disabled: argparse applies `type=` to the raw string, and `bool("False")` is `True`.)
- **-m** **--mining**, which algorithm's bank to retrieve with; choices are `gtkg`, `exhaustive`, `ragtkgc`, `ragtkgc_no_walks`, `raw`, `standard`. It names the output folder **and** selects the rule bank by filename, so a `gtkg` run can no longer silently retrieve with another algorithm's bank.
- **--index_target**, render every history object as `id.Name` (e.g. `18.Thailand`). Subjects, relations and the answers file stay bare names. Adds `_idn` to the variant token, and must be passed to `create_json_train.py` as well so the target matches the history. **Cannot be combined with `--save_metadata`** — see below.
- **--save_metadata**, record the complete pre-filter retrieval state so `apply_history_filters.py` can derive any filter combination later. It cannot be combined with `--num_facts`, `--top_k_rules`, `--confidence_threshold`, `--early_stop_at_num_facts` or `--index_target`, and in this mode no `history_facts/` is written.

  The first four are excluded because a filtered run's metadata is not the full state, so anything derived from it would be silently wrong (a second cap applied to an already-capped list, a top-k over rules already trimmed). `--index_target` is excluded for the opposite reason: the metadata stores *fact indices*, not rendered text, so the flag would change nothing in what is written — it would only name the output directory `_idn` for a directory holding no rendered history at all. Rendering is `apply_history_filters.py`'s decision.

##### Where each option belongs

The metadata stores fact indices, so one base retrieval can serve every downstream variant. What it cannot serve is an option that changes *which facts are retrieved in the first place*:

| Fixed at retrieval (baked into the fact indices) | Derivable later, in `apply_history_filters.py` |
| --- | --- |
| `--mining` (the rule bank) | `--num_facts` |
| `--inverse_body_object_match` (swaps the anchor a rule body matches against) | `--top_k_rules` |
| `--early_stop_at_num_facts` (truncates rule iteration, so fewer rules are recorded) | `--confidence_threshold` + `--model_type` |
| `--retrieve_type` (selects the algorithm) | `--index_target` |

So: **retrieve once per `(dataset, rule bank)`** with `--length_1_only --inverse_body_object_match --save_metadata`, then derive every variant from that base. A variant directory's name then always describes what is inside it, because retrieval is no longer able to accept a rendering flag.

Output is saved under `RAGTKGC/data/processed_new/{dataset}/{mining}/{split}{variant}/`, where the variant suffix records the options in use (see **Naming conventions** below):

- `history_facts/history_facts_{dataset}.txt` [A]
- `history_facts/history_facts_{dataset}_rule_ids.txt` — one JSON list per sample, the ids of the rules that contributed that sample's facts
- `test_answers/test_answers_{dataset}.txt` [B] — written for every split, despite the name
- `metadata/history_metadata_{dataset}_{split}.jsonl` — only with `--save_metadata`

With `--save_metadata` only the last two are written; regenerate [A] and its rule ids by running `apply_history_filters.py` with no filters. Older runs also produced `history_facts_{dataset}_idx_fine_tune_all.txt`, a byte-identical copy of [A] that nothing read; it is no longer written.

#### Deriving variants

Every variant comes from a base with `apply_history_filters.py`, which writes the same [A] and [B] files a direct retrieval would. Run it from `data_utils/`:

```
python apply_history_filters.py \
    --metadata ../data/processed_new/icews14/gtkg/test_inv/metadata/history_metadata_icews14_test.jsonl \
    --output_dir ../data/processed_new/icews14/gtkg/test_inv_n50_k10_idn/ \
    --num_facts 50 --top_k_rules 10 --index_target
```

- **-meta** **--metadata**, the base's `history_metadata_{dataset}_{split}.jsonl`. Required.
- **-o** **--output_dir**, where to write the variant. Required. Name it with the same variant tokens the filters imply.
- **-n** **--num_facts**, keep at most N facts per sample, newest first.
- **-k** **--top_k_rules**, keep only the K highest-confidence fired rules, then union their facts, then apply `--num_facts`.
- **-ct** **--confidence_threshold**, split facts into high- and low-confidence groups at this rule confidence.
- **-mt** **--model_type**, how to format that split; `t5` (default) concatenates the groups with the high-confidence ones nearest the query, `llm` labels them with text headings.
- **-idn** **--index_target**, render objects as `id.Name`. Off by default. Must match the flag given to `create_json_train.py`.
- **--stats_only**, simulate the build: write `filter_stats.json` and nothing else. Runs in seconds, so use it to price a filter combination — post-filter history size and confidence distribution — before spending GPU time on it.

Each variant directory also gets a `filter_stats.json` recording the filters, `index_target`, the sample count, and the post-filter history-size and confidence distributions, so it documents its own contents rather than depending on the base it came from.

The source facts under `data/original/{dataset}/` must be present: the metadata holds indices into `all_facts.txt`, not rendered text.

To confirm a bare and an indexed variant differ in nothing but the prefix, `diff_variants.py` checks them line by line:

```
python diff_variants.py --indexed <indexed history_facts.txt> --bare <bare history_facts.txt> \
    --answers_indexed <indexed answers.txt> --answers_bare <bare answers.txt>
```

#### Training files
For training, you need to convert history_facts files into json file:
```
python ./data_utils/create_json_train.py --dir_of_trainset 'the_full_trainset_to_convert (see [A])' --dir_of_answers 'the_test_answers (see [B])' --dir_of_entities2id 'the_json_of_entities2id' --path_save 'recommend_the_same_split_folder_as_the_one_converted' --name_train {dataset}_{mining}{variant}_{split}
```
Position on the RAGTKGC root folder. An example for the icews14 gtkg train split would be like:
```
python data_utils/create_json_train.py \
  --dir_of_trainset data/processed_new/icews14/gtkg/train_inv_n50/history_facts/history_facts_icews14.txt \
  --dir_of_answers data/processed_new/icews14/gtkg/train_inv_n50/test_answers/test_answers_icews14.txt \
  --dir_of_entities2id data/original/icews14/entity2id.json \
  --path_save data/processed_new/icews14/gtkg/train_inv_n50/json \
  --name_train icews14_gtkg_inv_n50_train --nums_sample 1024 --seed 1
```

You may do it for other splits too (test and valid), especially "test" as it will be needed for testing. Just write their name instead of "train" in the provided example.

Create JSON train parameters:
- **--dir_of_trainset**, the path to the training set directory.
- **--dir_of_answers**, the path to the answers directory.
- **--dir_of_entities2id**, the path to the entities2id file.
- **--path_save**, where to save the results.
- **--nums_sample**, how many samples to convert for training; default: 16 (and the whole set). For example, you can write '16,32,128' if you want files with those amounts of trainins samples, besides the whole set.
- **--name_train**, required; the stem of the written json files. Follow `{dataset}_{mining}{variant}_{split}` so the training file, the model and the results file all carry the variant. It was `--dataset`, which made every variant of a dataset write the same `icews14.json`.
- **--index_target**, write the target as `id.Name`. Must match the `--index_target` used during retrieval, or the target format and the history format disagree.
- **--seed**, seed for the training-subset sampler; omit for a non-reproducible draw.

### Fine tuning models

If you want to change any hyperparameter, please update them manually in the desired training file.
**Each run writes to `./models/{--trained_model_name}/`, holding the final model and that run's `checkpoint-*` directories. No renaming is needed. Name the model `{base}_{dataset}_{mining}{variant}_s{seed}`, e.g. `flan-t5-small_icews14_gtkg_inv_n50_s1`; the seed belongs in the name because the reported numbers are a mean over seeds. Checkpoints are per-run because Trainer names them by optimiser step, so two runs over the same training-set size would otherwise overwrite each other.**

#### Flan-T5-Small

To fine tune a Flan-T5-Small model, run the following command:
```
python training_T5.py --dataset name_of_dataset --trained_model_name 'name_of_fine_tuned_model' --train_file_path "path_to_the_training_file"
```

An example for icews18 would be:
```
python training_T5.py --dataset icews14 --seed 1 \
  --trained_model_name 'flan-t5-small_icews14_gtkg_inv_n50_s1' \
  --train_file_path "gtkg/train_inv_n50/json/icews14_gtkg_inv_n50_train.json"
```
Parameters:
- **--dataset**, the name of the dataset, can be icews14 or icews18.
- **--trained_model_name**, name of the fine tuned model, use the same one when renaming the checkpoint folder.
- **output_dir**, path to where to save the model; default: "./models".
- **train_file_path**, path to the training file; by default it starts searching from "./data/processed_new/{dataset}/", only provide the rest of the path as in the example above.

#### LLaMA2-7B

To fine tune (QLORA) a LLaMA2-7B model, run the following command:
```
python training_LLaMA.py --dataset name_of_dataset --trained_model_name 'name_of_fine_tuned_model' --train_file_path "path_to_the_training_file"
```

An example for icews18 would be:
```
python training_LLaMA.py --dataset icews14 --seed 1 \
  --trained_model_name 'llama-2-7B_icews14_gtkg_inv_n50_s1' \
  --train_file_path "gtkg/train_inv_n50/json/icews14_gtkg_inv_n50_train_1024.json"
```

Parameters:
- **--dataset**, the name of the dataset, can be icews14 or icews18.
- **--trained_model_name**, name of the fine tuned model, use the same one when renaming the checkpoint folder.
- **output_dir**, path to where to save the model; default: "./models".
- **train_file_path**, path to the training file; by default it starts searching from "./data/processed_new/{dataset}/", only provide the rest of the path as in the example above.

This part is also provided as a Jupyter Notebook. The reason is that the training part was moved to Google Colab due to intensive resources needed to fine tune LLMs. On Colab, it is easier to work with notebooks than command lines. You can also run the notebook locally. All instructions are provided in the notebook, which is easy-to-follow.
For LLaMA2-7B, we have used A100 GPU setting, while for Flan-T5-Small we opted for T4 GPU.


### Testing models

Code based on https://github.com/usc-isi-i2/isi-tkg-icl. Many thanks for their great contribution!

To test any model, you need to run from RAGTKGC folder:
```
python run_hf.py --base_model "the base model" --finetuned_model "fine tuned version of base model" --dataset "dataset name" --dataset_path "path to test file"
```

An example for testing a LLaMA2-7B-ragtkgc LORA version on icews18 dataset:
```
python run_hf.py --base_model "TheBloke/Llama-2-7B-fp16" \
  --finetuned_model "llama-2-7B_icews14_gtkg_inv_n50_s1" --dataset "icews14" \
  --dataset_path "gtkg/test_inv_n50/json/icews14_gtkg_inv_n50_test.json" --tail_truncate_long_inputs
```

The results file is named `{finetuned_model}_{dataset_path basename}_{tail_truncate|no_tail_truncate}.jsonl`, so it records both the model and the evaluation set. Those are independent axes: evaluating one variant's model on another variant's data is a legitimate experiment, so both belong in the name.

An example with RAG:
```
python run_hf.py --base_model "TheBloke/Llama-2-7B-fp16" \
  --finetuned_model "llama-2-7B_icews14_gtkg_inv_n50_s1" --dataset "icews14" \
  --dataset_path "gtkg/test_inv_n50/json/icews14_gtkg_inv_n50_test.json" \
  --dataset_rag_path "test_rag/icews14_gpt_given_rules.json"
```

Parameters:
- **--base_model**, the name of the base model, as it is on HuggingFace or folder name if saved locally; default: "google/flan-t5-small".
- **--finetuned_model**, name of the folder (or HuggingFace) which holds the fine tuned model version or peft adapter.
- **--dataset**, the name of the dataset; default: "icews18", other choice is "icews14".
- **--dataset_path**, the path to the test dataset; by default, it starts from "./data/processed_new/{args.dataset}/".
- **--dataset_rag_path**, the path to the rag test dataset; by default, it starts from "./data/processed_new/{args.dataset}/".
- **--num_beams**, beam width, which is also how many candidates get ranked; default: 10. Values below 10 make H@10 carry no information beyond H@3.
- **--length_penalty**, exponent the sequence log-probability is divided by the token count with; default: 0.6.
- **--max_new_tokens**, generation budget; derived from the longest target in the split when omitted.
- **--prompt_prefix**, wrap each prompt in the LLaMA baseline's instruction block. Off by default. It is LLaMA-2-chat markup and inert text for an encoder-decoder model, it assumes `--index_target`, and it **must match the flag used during training** or the model is evaluated on a prompt shape it never saw. Applied after truncation, with its token cost reserved from the budget, so the instruction survives `--tail_truncate_long_inputs` instead of being the first thing cut.
- **--limit**, evaluate only the first N samples. For smoke-testing a change to the decoder; the resulting file is short, so `compute_metrics_from_results.py` skips it.

##### Decoding

Candidates come from beam search, with `num_beams` sequences returned per query, cleaned to a bare entity name, deduplicated, and ranked best first. H@k therefore needs `num_beams >= k`.

Two settings decide what rank 1 is:

- **`length_penalty`** divides a candidate's total log-probability by `length ** length_penalty`, so it decides how entity names of different token length compare. The default 0.6 is the conventional seq2seq length-normalisation value (Wu et al., 2016), taken as a standard rather than tuned on this data. Note that T5's own published value, 2.0, is for summarization, which rewards long outputs; it is the wrong end of the range for a short entity name.
- **`num_beams`** at 1 is greedy decoding, which yields a single candidate and collapses H@1, H@3 and H@10 onto the same number.

Results are saved in "./results/{dataset_name}/", with the name of "{finetuned_model}_{test_|rag|_dataset}.jsonl"
When testing with a rag dataset, we automatically load the original test file also, and take an input sample either from the original set or rag set, depending if it was modified by RAG. We do that by keeping a file with all modified indexes (more details on the RAG with GPT 4.1 section).

#### Calculating BERTScore

BERTScore is calculated using a PLM, such as roberta-large. Thus, if calculated live (during testing) after each prediction as the other metrics, it would require a lot of extra time. We provide a way of determining it afterwards, based on the results file.

It can be run using the following command:
```
python bertscore.py --results_file "name_of_results_file"
```
An example can be:
```
python bertscore.py --results_file "icews18/llama-2-7B-icews18-ragtkgc_icews18_ragtkgc_test.jsonl"
```

Parameters:
- **--rf** **--results_file**, the path of the results file; by default, it looks in the results folder.


#### Compute metrics from results files

You can compute the metrics for already saved predictions from the results folder, by running:
```
python compute_metrics_from_results.py --dataset "name_of_dataset" --file_name "name_of_the_results_file.jsonl"
```

An example for icews14 is:
```
python compute_metrics_from_results.py --dataset "icews14" --file_name "flan-t5-small-icews14-ragtkgc_icews14_gpt_rule_miner.jsonl"
```

Parameters:
- **--dataset**, the name of the dataset; default: "icews14".
- **--file_name**, the name of the results file. It automatically searches for it in "./results/{dataset}". Default: "all", as test all files from the target dataset folder.

### RAG with GPT 4.1

In Appendix A, there are examples of prompts for retrieving extra information using any rag version.
In Appendix B, there is an example of a test sample enhanced with information from RAGTKGC with GPT 4.1.

**You need to save you own OpenAI key in api_key.txt file!**
To obtain RAG-enhanced input prompts, you can use the following command:
```
python rag_with_gpt_4_1.py --dataset "dataset_name" --rule_file "name_of_the_rule_bank.txt" --rag_version "desired_rag_version"
```

An example for icews14:
```
python rag_with_gpt_4_1.py --dataset "icews14" --rule_file "080525131706_r[1]_n200_exp_s1_rules.txt" --rag_version "gpt-given-rules"
```

You will be prompt to input how many samples you want to enhance with RAG. Additional information about how many samples were wrongly predicted by already existing fine tuned versions of models will also be shown to make an informed decision. This metric is based on files from the results folder. You will see a message like "There are 2324 samples with the target object wrongly predicted 8/8 times. There are 789 samples with the target object wrongly predicted 7/8 times." It means that out of 8 results files (8 different predictions done by different models for the same test sample), 2324 samples had the target object wrongly predicted 8 times (basically no model was able to predict the right target answer), and so on. When you will be prompt to input the number of samples to be extended with RAGTKGC with GPT 4.1, you will have to input a number lower than the maximum available samples. Also, if the input number exceeds the first group of samples (e.g. 3000 is more than 2324), we subsequently take samples from the next group. 
Parameters:

- **--dataset**, name of the dataset, default: "icews18".
- **--rule_file**, name of the rule bank specific for the chosen dataset; default: "050525174831_r[1]_n200_exp_s1_rules.txt" which is the rule bank obtained with ragtkgc on icews18. We encourage you to use rule banks obtained with our approach, as they have plenty more mined rules, but you can use any desired one.
- **--conf_treshold**, confidence treshold, the minimum value that the confidence of a rule must have in order to be selected as input for GPT 4.1; default: 0.5 (between 0 and 1).
- **--rag_version**, the version of the input prompt given to GPT 4.1; choices=['gpt-given-rules','gpt-given-relations','gpt-rule-miner'], default='gpt-given-rules'.
- **--use_llm_similarity**, if entities/relations names should be mapped to known ones using llm (SentenceTransformer('all-MiniLM-L6-v2')) similarity; default = False.
- **--no_similarity**, if entities/relations names should be kept as generated by GPT 4.1; default = False.

Files are saved at './data/processed_new/{args.dataset}/test_rag/{args.dataset}_{args.rag_version}'. You will also see there temporary files marked with the suffix 'temp', which dynamically stores each processed sample, in case any unexpected interruptions occur before being able to save the whole proccesed set of samples. If it happens, you will have to manually write in the rag_with_gpt_4_1.py file to ignore the first n processed samples (an easy "if i < n: continue" will do). Also, you will have to manually concatenate the resulted temporary files (make sure to put "[" at the beginning and "]" at the end of the new file).
You will also see a file such as "{dataset}_gpt_index.txt" which stores a list of indexes of those samples from the test set that were enhanced with RAG. Thus, when testing, we know to take its enhanced version if necessary.
