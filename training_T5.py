# import the required classes

import argparse
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--trained_model_name",type = str)
    parser.add_argument("--output_dir", default = './models', type = str)
    parser.add_argument("--train_file_path", type = str)

    args = parser.parse_args()
    return args
    
# tokenize the input prompt and target entity (as labels)

def process_function(examples):

  inputs = tokenizer(examples['context'], return_special_tokens_mask=True)
  labels = tokenizer(examples['target'], return_special_tokens_mask=True)
  inputs['labels'] = labels['input_ids']

  return inputs


if __name__ == "__main__":

    args = parser()

    # Feel free to set your own values.

    model = 'google/flan-t5-small' # the model to be finetuned
    trained_model_name = args.trained_model_name # the name of the trained model
    output_dir = args.output_dir #The output directory where checkpoints will be written.

    # warm up: nr of steps where the lr starts from 0 and goes to initial set, to prevent oferfitting on early data
    training_args = Seq2SeqTrainingArguments(output_dir = output_dir,
                                    auto_find_batch_size = True, # Find a correct bvatch size that fits the size of Data.
                                    per_device_train_batch_size=8, # set to 2 if ICEWS14
                                    #per_device_eval_batch_size = 1,
                                    learning_rate = 3e-4, # Higher learning rate than full fine-tuning.
                                    num_train_epochs =  1, # set to 3 if ICEWS14
                                    lr_scheduler_type = 'cosine', # It’s designed to lower the learning rate more gradually than step or exponential decay schedulers, and it often includes a restart mechanism, where the learning rate resets to its initial value at regular intervals before starting the next cycle of decay. This restart helps the model escape from potential local minima by periodically taking larger steps, enabling it to search more thoroughly across the loss landscape.
                                    #eval_strategy="steps",
                                    warmup_steps = 1000,
                                    logging_steps = 5000,
                                    #gradient_accumulation_steps=8,
                                    bf16 = True,
                                    weight_decay = 0.1, # penalizes weights to reduce overfitting
                                    report_to = 'none') # the list of integrations to report the results and logs to.
    
    # load the model and its tokenizer

    training_model = T5ForConditionalGeneration.from_pretrained(model, trust_remote_code = True,
                                                                device_map = 'auto',
                                                                )
    tokenizer = T5Tokenizer.from_pretrained(model)

    # set the model to train mode
    training_model.train()

    # load a dataset from the data files 
    # IMPORTANT: for ICEWS18, there are two training folders: 1024 and full. Use 1024 for finetuning LLaMA2-7B and full for Flan-T5-Small

    dataset = load_dataset('json', data_files=f'./data/processed_new/{args.dataset}/{args.train_file_path}', split = 'train')

    tokenized_input = dataset.map(process_function, batched = True, remove_columns=dataset.column_names)

    # filter out any input prompt that is longer than the models context size

    tokenized_input = tokenized_input.filter(lambda x: len(x['input_ids']) < tokenizer.max_len_single_sentence)

    trainer = Seq2SeqTrainer(model = training_model, # We pass in the PEFT version of the foundation model or the standard one if full finetuning is desired
                args = training_args, #The args for the training.
                train_dataset = tokenized_input, #The dataset used to to train the model.
                data_collator=DataCollatorForSeq2Seq(tokenizer, model = training_model), # mlm=False indicates not to use masked language modeling
                )
    
    trainer.train()