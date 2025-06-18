# RAGTKGC
This is the official implementation of the paper **RAGTKGC: Undertaking Temporal Knowledge Graph Completion with Retrieval Augmented Generation**

**IF YOU CANNOT CLONE THE REPO, PLEASE DOWNLOAD IT FROM HERE --> https://drive.google.com/file/d/1Re-H4k78tfqoC9D_hzHjmObIGL7oChMj/view?usp=drive_link**

**Important Note! The code is written to run locally. However, the fine tuning part is also provided as a Jupyter Notebook, so you can move it on Google Colab if you don't have enough resources. Any part that is moved on Colab requires you to change the paths by hand, as Google Colab starts looking for files from /Content/...**

## Create a Conda Env

We advise you to create a new environment for our code, but you can also run it on already existing ones or Colab, just make sure to install requirements and torch as provided below:
```
git clone https://github.com/AnonOctopus/RAGTKGC
cd RAGTKGC

conda create -n ragtkgc python=3.9.21
conda activate ragtkgc

pip install -r requirements.txt 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

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
- **-m** **--mining**, the rule mining approach, default: ragtkgc, other choices are gtkg and ragtkgc_no_walks (our algorithm but without walks, only start mining from each unique quadruple).

You will get a rule bank file similar to "060723022344_r[1,2,3]_n200_exp_s12_rules.json" under the RAGTKGC/data/processed_new/{dataset_name}/output/{dataset_name} folder.

### History retrieving

Find the file name of rule bank json (in RAGTKGC/data/processed_new/{dataset_name}/output/{dataset_name}) and run from the folder RAGTKGC/data_utils:
```
python retrieve.py --name_of_rules_file name_rules.json --dataset icews18 -m ragtkgc
```
An example for icews18 would be like:
```
python retrieve.py --name_of_rules_file 050525174831_r[1]_n200_exp_s1_rules.json --dataset icews18 -m ragtkgc
```

History retrieving parameters:
- **--d** **--dataset**, dataset name; default: icews14
- **-t** **--retrieve_type**, the underlying mining algorithm to use; default: TLogic (TLogic-3), other choice is bs (check the original paper for more details).
- **-r** **--name_of_rules_file**, the name of the file where the rules are stored.
- **--l** **--rule_length_all**, boolean value to set whether to use rule of lengths greater than 1 or not; default: True, set to False if you have rule banks with rule lengths greater than 1 and do not want to use them for facts retrieval.
- **-m** **--mining**, the rule mining approach, default: ragtkgc, other choices are gtkg and ragtkgc_no_walks (our algorithm but without walks, only start mining from each unique quadruple). Here, it is used ONLY for saving it in the correct folder.

Output is saved at:
- RAGTKGC/data/processed_new/{dataset_name}/{rule_learning_algorithm}/[train, valid, test]/history_facts/history_facts_{dataset}.txt [A]
- RAGTKGC/data/processed_new/{dataset_name}/{rule_learning_algorithm}/[train, valid, test]/history_facts/history_facts_{dataset}_idx_fine_tune_all.txt
- RAGTKGC/data/processed_new/{dataset_name}/{rule_learning_algorithm}/[train, valid, test]/test_answers/test_answers_{dataset}.txt [B]

### Training files
For training, you need to convert history_facts files into json file:
```
python ./data_utils/create_json_train.py --dir_of_trainset 'the_full_trainset_to_convert (see [A])' --dir_of_answers 'the_test_answers (see [B])' --dir_of_entities2id 'the_json_of_entities2id' --path_save 'recommend_the_same_split_folder_as_the_one_converted' --dataset name_of_dataset
```
Position on the RAGTKGC root folder. An example for icews18 train split would be like:
```
python data_utils/create_json_train.py --dir_of_trainset data/processed_new/icews18/ragtkgc/train/full//history_facts/history_facts_icews18.txt --dir_of_answers  data/processed_new/icews18/ragtkgc/train/full/test_answers/test_answers_icews18.txt --dir_of_entities2id data/processed_new/icews18/entity2id.json --path_save data/processed_new/icews18/ragtkgc/train/full/history_modeling_train --dataset icews18
```

You may do it for other splits too (test and valid), especially "test" as it will be needed for testing. Just write their name instead of "train" in the provided example.

Create JSON train parameters:
- **--dir_of_trainset**, the path to the training set directory.
- **--dir_of_answers**, the path to the answers directory.
- **--dir_of_entities2id**, the path to the entities2id file.
- **--path_save**, where to save the results.
- **--nums_sample**, how many samples to convert for training; default: 16 (and the whole set). For example, you can write '16,32,128' if you want files with those amounts of trainins samples, besides the whole set.
- **--dataset**, the name of the output file is going to be the dataset name.

## Fine tuning models

If you want to change any hyperparameter, please update them manually in the desired training file.
**Model's checkpoints are automatically saved at "./models". However, you need to rename the final checkpoint as desired. We recommend naming it as "name_of_base_model_dataset_history_modeling_algorithm". An example would be "llama-2-7B-icews18-ragtkgc". You can delete the other checkpoints if not needed.**

### Flan-T5-Small

To fine tune a Flan-T5-Small model, run the following command:
```
python training_T5.py --dataset name_of_dataset --trained_model_name 'name_of_fine_tuned_model' --train_file_path "path_to_the_training_file"
```

An example for icews18 would be:
```
python training_T5.py --dataset icews18 --trained_model_name 'flan-t5-small-icews18-ragtkgc' --train_file_path "ragtkgc/train/full/history_modeling_train/icews18.json"
```
Parameters:
- **--dataset**, the name of the dataset, can be icews14 or icews18.
- **--trained_model_name**, name of the fine tuned model, use the same one when renaming the checkpoint folder.
- **output_dir**, path to where to save the model; default: "./models".
- **train_file_path**, path to the training file; by default it starts searching from "./data/processed_new/{dataset}/", only provide the rest of the path as in the example above.

### LLaMA2-7B

To fine tune (QLORA) a LLaMA2-7B model, run the following command:
```
python training_LLaMA.py --dataset name_of_dataset --trained_model_name 'name_of_fine_tuned_model' --train_file_path "path_to_the_training_file"
```

An example for icews18 would be:
```
python training_LLaMA.py --dataset icews18 --trained_model_name 'llama-2-7b-icews18-ragtkgc' --train_file_path "ragtkgc/train/1024/history_modeling_train/icews18.json"
```

Parameters:
- **--dataset**, the name of the dataset, can be icews14 or icews18.
- **--trained_model_name**, name of the fine tuned model, use the same one when renaming the checkpoint folder.
- **output_dir**, path to where to save the model; default: "./models".
- **train_file_path**, path to the training file; by default it starts searching from "./data/processed_new/{dataset}/", only provide the rest of the path as in the example above.

This part is also provided as a Jupyter Notebook. The reason is that the training part was moved to Google Colab due to intensive resources needed to fine tune LLMs. On Colab, it is easier to work with notebooks than command lines. You can also run the notebook locally. All instructions are provided in the notebook, which is easy-to-follow.
For LLaMA2-7B, we have used A100 GPU setting, while for Flan-T5-Small we opted for T4 GPU.


## Testing models

Code based on https://github.com/usc-isi-i2/isi-tkg-icl. Many thanks for their great contribution!

To test any model, you need to run from RAGTKGC folder:
```
python run_hf.py --base_model "the base model" --finetuned_model "fine tuned version of base model" --dataset "dataset name" --dataset_path "path to test file"
```

An example for testing a LLaMA2-7B-ragtkgc LORA version on icews18 dataset:
```
python run_hf.py --base_model "TheBloke/Llama-2-7B-fp16" --finetuned_model "llama-2-7B-icews18-ragtkgc" --dataset "icews18" --dataset_path "ragtkgc/test/10000/history_modeling_test/icews18_test.json"
```

An example for testing a LLaMA2-7B-ragtkgc LORA version on icews18 dataset with RAG:
```
python run_hf.py --base_model "TheBloke/Llama-2-7B-fp16" --finetuned_model "llama-2-7B-icews18-ragtkgc" --dataset "icews18" --dataset_path "ragtkgc/test/10000/history_modeling_test/icews18_test.json" --dataset_rag_path "test_rag/icews18_gpt_given_rules.json"
```

Parameters:
- **--base_model**, the name of the base model, as it is on HuggingFace or folder name if saved locally; default: "google/flan-t5-small".
- **--finetuned_model**, name of the folder (or HuggingFace) which holds the fine tuned model version or peft adapter.
- **--dataset**, the name of the dataset; default: "icews18", other choice is "icews14".
- **--dataset_path**, the path to the test dataset; by default, it starts from "./data/processed_new/{args.dataset}/".
- **--dataset_rag_path**, the path to the rag test dataset; by default, it starts from "./data/processed_new/{args.dataset}/".

Results are saved in "./results/{dataset_name}/", with the name of "{finetuned_model}_{test_|rag|_dataset}.jsonl"
When testing with a rag dataset, we automatically load the original test file also, and take an input sample either from the original set or rag set, depending if it was modified by RAG. We do that by keeping a file with all modified indexes (more details on the RAG with GPT 4.1 section).

### Calculating BERTScore

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


### Compute metrics from results files

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

## RAG with GPT 4.1

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
