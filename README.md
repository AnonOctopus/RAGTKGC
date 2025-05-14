# RAGTKGC
This is the official implementation of the paper **RAGTKGC: Undertaking Temporal Knowledge Graph Completion with Retrieval Augmented Generation**

## History Modeling

Code based on https://github.com/mayhugotong/GenTKG. Many thanks for their great contribution!

In the paper, we test four ways of modeling the history:
1. raw, where no history is retrieved
2. standard, where we select quads that contain the same subject as the target one (see https://github.com/usc-isi-i2/isi-tkg-icl for more details)
3. gtkg, where history is retrieved using the standard method + temporal logical rules that were mined from the given TKG (train split) (https://github.com/mayhugotong/GenTKG)
4. ragtkgc, our approach which modifies gtkg by considering all possible paths (i.e. combination of edges) between the subject and object for any fixed relationship during rules mining.

For 1 and 2, the datasets are already provided and is no need to generate them again (they will always result in the same input prompts).
For 3 and 4, we also provide the datasets that we have used for experiments. Additionally, you can also generate your own versions by following the instructions from below.

### Rules learning
It works with files from RAGTKGC/data/processed_new/ . 
You can produce other rule banks besides the provided ones by running e.g. for icews14:
```
cd data_utils/rules_learning
python learn.py -d icews14 -l 1 2 3 -n 200 -p 15 -s 12 -m ragtkgc
```
Rules learning parameters:
- **-d** **--dataset**, dataset name.
- **-l** **--rule_lengths**, length of the mined rules, by default it is 1.
- **-n** **--num_walks**, number of times the mining algorithm starts extracting rules from a sampled quadruple.
- **--transition_distr**, how to transition from an edge to another; default: exp (exponential), other choice is unif (uniform).
- **-p** **--num_processes**, number of parallel processes, for accelerating. 
- **-s** **--seed**, for reproduction purposes.
- **-m** **--mining**, the rule mining approach, default: ragtkgc, other choices are gtkg and ragtkgc_no_walks (our algorithm but without walks, only start mining from each unique quadruple)

You will get a rule bank file similar to "060723022344_r[1,2,3]_n200_exp_s12_rules.json" under the RAGTKGC/data/processed_new/{dataset_name}/output/{dataset_name} folder.

### History retrieving

Find the file name of rule bank json (in RAGTKGC/data/processed_new/{dataset_name}/output/{dataset_name}) and run from the folder RAGTKGC/data_utils:
```
python retrieve.py --name_of_rules_file name_rules.json --dataset icews18
```
An example for icews18 would be like:
```
python retrieve.py --name_of_rules_file 050525174831_r[1]_n200_exp_s1_rules.json --dataset icews18
```

History retrieving parameters:
- **--d** **--dataset**, dataset name; default: icews14
- **-t** **--retrieve_type**, the underlying mining algorithm to use; default: TLogic (TLogic-3), other choice is bs (check the original paper for more details).
- **-r** **--name_of_rules_file**, the name of the file where the rules are stored.
- **--l** **--rule_length_all**, boolean value to set whether to use rule of lengths greater than 1 or not; default: True, set to False if you have rule banks with rule lengths greater than 1 and do not want to use them for facts retrieval.

Output is saved at:
- RAGTKGC/data/processed_new/{dataset_name}/{rule_learning_algorithm}/[train, valid, test]/history_facts/history_facts_{dataset}.txt [A]
- RAGTKGC/data/processed_new/{dataset_name}/{rule_learning_algorithm}/[train, valid, test]/history_facts/history_facts_{dataset}_idx_fine_tune_all.txt
- RAGTKGC/data/processed_new/{dataset_name}/{rule_learning_algorithm}/[train, valid, test]/test_answers/test_answers_{dataset}.txt [B]

By default, {rule_learning_algorithm} is set to "ragtkgc", and needs to be manually changed in the RAGTKGC/data_utils/retrieve.py file, line 27, to another rule_learning_algorithm name.

**For icews18, you have to manually create the all_facts.txt file, because it is too large to load it on the hub. Please go to  RAGTKGC/data/processed_new/icews18 and just create a new .txt by concatenating train + valid + test, in this order.**

### Training files
For training, you need to convert history_facts files into json file:
```
python ./data_utils/create_json_train.py --dir_of_trainset 'the_full_trainset_to_convert (see [A])' --dir_of_answers 'the_test_answers (see [B])' --dir_of_entities2id 'the_json_of_entities2id' --path_save 'recommend_the_same_split_folder_as_the_one_converted'
```
Position on the RAGTKGC root folder. An example for icews18 train split would be like:
```
python data_utils/create_json_train.py --dir_of_trainset data/processed_new/icews18/ragtkgc/train/history_facts/history_facts_icews18.txt --dir_of_answers  data/processed_new/icews18/ragtkgc/train/test_answers/test_answers_icews18.txt --dir_of_entities2id data/processed_new/icews18/entity2id.json --path_save data/processed_new/icews18/ragtkgc/train/history_modeling_train
```

You may do it for other splits too (test and valid), especially "test" as it will be needed for testing Just write their name instead of "train" in the provided example.

Create JSON train parameters:
- **--dir_of_trainset**, the path to the training set directory.
- **--dir_of_answers**, the path to the answers directory.
- **--dir_of_entities2id**, the path to the entities2id file.
- **--path_save**, where to save the results.
- **--nums_sample**, how many samples to convert for training; default: 16 (and the whole set). For example, you can write '16,32,128' if you want files with those amounts of trainins samples, besides the whole set.
- **--name_train**, the name of the output file; default: "icews18".

## Fine tuning models

The other parts of the framework run using command lines, however this part is provided as a Jupyter Notebook. The reason is that the training part was moved to Google Colab due to intensive resources needed to fine tune LLMs. ON Colab, it is easier to work with notebooks than command lines. You can also run the notebook locally. All instructions are provided in the notebook, which is easy-to-follow.
For LLaMA2-7B, we have used A100 GPU setting, while for Flan-T5-Small we opted for T4 GPU.
Model's checkpoints are automatically saved at "./models". However, you need to rename the final checkpoint as desired. We recommend naming it as "name_of_base_model_dataset_history_modeling_algorithm". An example would be "Llama-2-7B-icews18-ragtkgc". You can delete the other checkpoints if not needed.

## Testing models

Code based on https://github.com/usc-isi-i2/isi-tkg-icl. Many thanks for their great contribution!

To test any model, you need to run from RAGTKGC folder:
```
python run_hf.py --base_model "the base model" --finetuned_model "fine tuned version of base model" --dataset "dataset name" --dataset_path "path to test file"
```

An example for testing a LLaMA2-7B-ragtkgc LORA version on icews18 dataset:
```
python run_hf.py --base_model "TheBloke/Llama-2-7B-fp16" --finetuned_model "Llama-2-7B-icews18-ragtkgc" --dataset "icews18" --dataset_path "ragtkgc/test/10000/history_modeling_test/icews18_test.json"
```

Parameters:
- **--base_model**, the name of the base model, as it is on HuggingFace or folder name if saved locally; default: "google/flan-t5-small".
- **--finetuned_model**, name of the folder (or HuggingFace) which holds the fine tuned model version or peft adapter.
- **--dataset**, the name of the dataset; default: "icews18", other choice is "icews14".
- **--dataset_path**, the path to the test dataset; by default, it starts from "./data/processed_new/{args.dataset}/".
- **--dataset_rag_path**, the path to the rag test dataset; by default, it starts from "./data/processed_new/{args.dataset}/".

Results are saved in "./results/{dataset_name}/", with the name of "{finetuned_model}_{test_|rag|_dataset}.jsonl"
