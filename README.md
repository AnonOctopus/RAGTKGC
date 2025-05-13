# RAGTKGC
This is the official implementation of the paper Undertaking Temporal Knowledge Graph Completion with Language Models.

### Rules learning
It works with files from ../../data/processed_new/ . 
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
    
You will get a rule bank file similar to "060723022344_r[1,2,3]_n200_exp_s12_rules.json" under the ../../data/processed_new/output/ folder.
