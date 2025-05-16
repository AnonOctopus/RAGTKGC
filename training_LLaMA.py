# import the required classes, feel free to add any other PeftConfig type

import argparse
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, DataCollatorForLanguageModeling, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from typing import Any, Dict, List, Union
import numpy as np

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--trained_model_name",type = str)
    parser.add_argument("--output_dir", default = './models', type = str)
    parser.add_argument("--train_file_path", type = str)

    args = parser.parse_args()
    return args

# tokenize the input prompt

def process_function(examples):

  texts = [f"{context} {target}] </s>" for context, target in zip(examples['context'], examples['target'])]

  inputs = tokenizer(texts, return_special_tokens_mask=True)


  return inputs


if __name__ == "__main__":

    args = parser()
    # we provide a customized DataCollatorForCompletionLM, to enable the model to only focus on the target entity, not the whole input text, while training
    # To train the model to predict only the response and ignore the prompt tokens,
    # it sets the label values before the response token to -100.
    # This ensures that those tokens are ignored by the PyTorch loss function during training.

    class DataCollatorForCompletionLM(DataCollatorForLanguageModeling):

        def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:

            # The torch_call method overrides the same method in the base class and
            # takes a list of examples as input.
            batch = super().torch_call(examples)

            labels = batch["labels"].clone()

            for i in range(len(examples)):

                # decode the labels tensor to get the original input prompt (for generative models, the labels are the inputs shifted by one to the right)
                # split the prompt to reach the target quadruple

                prompt = tokenizer.decode(batch['labels'][i][:-1])
                quad = prompt.split('\n')[-1].split(' ')[-1]
                quad = quad.strip()

                # remove any tokens that are not part of the target quad
                if quad.endswith(']'):
                    quad = quad.replace(']','')
                if '<s> [INST] ' in quad:
                    quad = quad.replace('<s> [INST] ','')
                if ' [/INST]' in quad:
                    quad = quad.replace('[/INST]','')
                if '\n' in quad:
                    quad = quad.replace('\n', '')

                # encode the target quad
                response_token_ids = self.tokenizer.encode(quad)[1:]

                # find where the target quad starts in the prompt and save its index
                response_token_ids_start_idx = None
                response_token_ids_start_idx = np.where(batch["labels"][i] == response_token_ids[0])[0][-1]
                
                if response_token_ids_start_idx is None:
                    # If the response token is not found in the sequence, it raises a RuntimeError.
                    
                    raise RuntimeError(
                        f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}')


                response_token_ids_end_idx = response_token_ids_start_idx

                # To train the model to predict only the response and ignore the prompt tokens,
                # it sets the label values before the response token to -100.
                # This ensures that those tokens are ignored by the PyTorch loss function during training.

                # set all labels for the input prompt until the target quad to -100
                labels[i, :response_token_ids_end_idx] = -100

                # there are a few tokens that need to be converted to another ones in order to correctly set the right label for the target quad
                # for example, the target quad may have an object like "Men_(Australia)"; the input prompt history (including the target quad) ends with "]", the tokenizer might encode ")]" together
                # and we are left with "Men_(Australia"; therefore, we change ")]" to ")", as "]" would have had the label -100 anyway and be ignored.

                if labels[i, len(response_token_ids)+response_token_ids_start_idx-1] == 4638: # ")]"
                    labels[i, len(response_token_ids)+response_token_ids_start_idx-1] = 29897 # ")"

                if labels[i, len(response_token_ids)+response_token_ids_start_idx-1] == 28166: # "))]"
                    labels[i, len(response_token_ids)+response_token_ids_start_idx-1] = 876 # "))"

                if labels[i, len(response_token_ids)+response_token_ids_start_idx-1] == 5586: # ".]"
                    labels[i, len(response_token_ids)+response_token_ids_start_idx-1] = 29889 # "."
                
                if args.dataset == 'icews14':
                    labels[i, len(response_token_ids)+response_token_ids_start_idx+1:] = -100
                else:
                    labels[i, len(response_token_ids)+response_token_ids_start_idx:] = -100
                
                # set the last label to 2, to mark the end of the input prompt
                labels[i,-1] = 2

            batch["labels"] = labels

            return batch
    
    # Feel free to set your own values.

    model = 'TheBloke/Llama-2-7B-fp16' # the model to be finetuned
    trained_model_name = args.trained_model_name # the name of the trained model
    output_dir = args.output_dir #The output directory where checkpoints will be written.

    # feel free to add any other parameter

    lora_config = LoraConfig(r = 8, #As bigger the R bigger the parameters to train.
                        lora_alpha=16, # a scaling factor that adjusts the magnitude of the weight matrix.
                        lora_dropout=0.05, #Helps to avoid Overfitting.
                        bias='lora_only', # this specifies if the bias parameter should be trained.
                        task_type='CAUSAL_LM',
                        use_rslora=True)
    
    # warm up: nr of steps where the lr starts from 0 and goes to initial set, to prevent oferfitting on early data
    training_args = TrainingArguments(output_dir = output_dir,
                                    auto_find_batch_size = True, # Find a correct bvatch size that fits the size of Data.
                                    per_device_train_batch_size=1,
                                    #per_device_eval_batch_size = 1,
                                    learning_rate = 3e-4, # Higher learning rate than full fine-tuning.
                                    num_train_epochs = 1,                                   
                                    warmup_steps = 20,
                                    logging_steps  = 20,
                                    gradient_accumulation_steps=8,
                                    #weight_decay = 0.1, # penalizes weights to reduce overfitting
                                    report_to = 'none') # the list of integrations to report the results and logs to.
    
    # load the model in a 4bit configuration to use QLora; this is how we trained the models, but you can also load them in 8bit or full if enough resources are available

    bnb4_config =  BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type='nf4', # precision of the stored weights
                                bnb_4bit_compute_dtype='bfloat16', # precision of computations
                                bnb_4bit_use_double_quant=True
                                )
    
    # set your own quantization_config if desired

    training_model = AutoModelForCausalLM.from_pretrained(model,
                                                            trust_remote_code = True,
                                                            quantization_config = bnb4_config
                                                            ) # for training, device_map does not have to be set. check -> https://huggingface.co/docs/transformers/v4.35.0/main_classes/quantization#bitsandbytes-integration
    tokenizer = AutoTokenizer.from_pretrained(model)

    tokenizer.pad_token = tokenizer.eos_token

    # If quantization is applied, enable the line, else disable it.
    training_model = prepare_model_for_kbit_training(training_model) # This method wraps the entire protocol for preparing a model before running a training.

    # If PEFT is desired, then get the peft version of the model, else disable it.
    training_model = get_peft_model(training_model, lora_config, low_cpu_mem_usage = False) # feel free to put any config file from above. low_cpu_mem_usage — Create empty adapter weights on meta device. Useful to speed up the loading process. Leave this setting as False if you intend on training the model -> https://huggingface.co/docs/peft/package_reference/peft_model

    print(training_model.print_trainable_parameters())

    # load a dataset from the data files
    # IMPORTANT: for ICEWS18, there are two training folders: 1024 and full. Use 1024 for finetuning LLaMA2-7B and full for Flan-T5-Small

    dataset = load_dataset('json', data_files=f'./data/processed_new/{args.dataset}/{args.train_file_path}', split = 'train')

    tokenized_input = dataset.map(process_function, batched = True, remove_columns=dataset.column_names)

    # filter out any input prompts longer than the model's context size; for LLaMA2-7B, the context size is 4096.
    tokenized_input = tokenized_input.filter(lambda x: len(x['input_ids']) < 4095)

    # !!! IMPORTANT !!! for icews18-ragtkgc, you already have the dataset of size 1024, for rest you have to select them

    length_ti = len(tokenized_input)
    
    if length_ti > 1024:
        step = round(length_ti / 1024)

        tokenized_input = tokenized_input.select(range(0, length_ti, step)[:1024])

    trainer = Trainer(model = training_model, # We pass in the PEFT version of the foundation model or the standard one if full finetuning is desired
                args = training_args, #The args for the training.
                train_dataset = tokenized_input, #The dataset used to to train the model.
                #eval_dataset = vd,
                data_collator=DataCollatorForCompletionLM(tokenizer, mlm=False), # mlm=False indicates not to use masked language modeling
                )

    trainer.train()
        
