# RAGTKGC
This is the official implementation of the paper Undertaking Temporal Knowledge Graph Completion with Language Models.

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

