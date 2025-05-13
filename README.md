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
- 
Output is saved at:
- RAGTKGC/data/processed_new/{dataset_name}/{rule_learning_algorithm}/[train, valid, test]/history_facts/history_facts_{dataset}.txt [A]
- RAGTKGC/data/processed_new/{dataset_name}/{rule_learning_algorithm}/[train, valid, test]/history_facts/history_facts_{dataset}_idx_fine_tune_all.txt
- RAGTKGC/data/processed_new/{dataset_name}/{rule_learning_algorithm}/[train, valid, test]/test_answers/test_answers_{dataset}.txt [B]

By default, {rule_learning_algorithm} is set to "ragtkgc", and needs to be manually changed in the RAGTKGC/data_utils/retrieve.py file, line 27, to another rule_learning_algorithm name.

For icews18, you have to manually create the all_facts.txt file, because it is too large to load it on the hub. Please go to  RAGTKGC/data/processed_new/icews18 and just create a new .txt by concatenating train + valid + test, in this order.
