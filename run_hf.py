from datasets.arrow_dataset import np
import torch
from tqdm import tqdm
from transformers import BitsAndBytesConfig,AutoModelForCausalLM, AutoTokenizer, logging as tf_logging
from peft import PeftModelForCausalLM
from model_utils import predict
from utils import (
    HitsMetric,
    get_args,
    get_filename,
    update_metric,
    write_results,
)

from datasets import load_dataset

tf_logging.set_verbosity_error()


if __name__ == "__main__":
    
    args = get_args()

    models_path = './models/'
    
    if 'google/flan-t5-small' == args.base_model:
      
      from transformers import T5Tokenizer, T5ForConditionalGeneration

      model = T5ForConditionalGeneration.from_pretrained(models_path + args.finetuned_model, trust_remote_code = True,
                                                            device_map = 'auto',
                                                            )
      
      tokenizer = T5Tokenizer.from_pretrained(models_path + args.finetuned_model)
      tokenizer.pad_token = tokenizer.eos_token

    elif 'TheBloke/Llama-2-7B-fp16' == args.base_model:

      bnb4_config =  BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_quant_type='nf4', # precision of the stored weights
                              bnb_4bit_compute_dtype='float16', # precision of computations
                              bnb_4bit_use_double_quant=True                 
                              )
      
      tokenizer = AutoTokenizer.from_pretrained(args.base_model)
      tokenizer.pad_token_id = tokenizer.eos_token_id

      training_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        quantization_config = bnb4_config
        )
      
      model = PeftModelForCausalLM.from_pretrained(training_model, models_path + args.finetuned_model)
      
    model.eval()
    print(f"model is loaded on device {model.device.type}")

    metric = HitsMetric()
    dataset_path = f'./data/processed_new/{args.dataset}/' + args.dataset_path

    test_set = load_dataset('json', data_files = dataset_path, split = 'train')
    if args.dataset == 'icews18': test_set = test_set.select(range(0,10000)) # select only the first 10k examples for icews18, due to resource limitations

    test_set_rag = []
    if args.dataset_rag_path:

      rag_dataset_path = f'./data/processed_new/{args.dataset}/' + args.dataset_rag_path
      test_set_rag = load_dataset('json', data_files = rag_dataset_path, split = 'train')
      
      # RAG with GPT 4.1 was run on a fixed number of test samples. However, those samples were not 'the first n' of the test set, but the ones where no model correctly predicted the target
      # therefore, we saved the indexes of those samples that benefit from the addition of extra information from our framework

      rag_indexes = 'icews18_gpt_index.txt' if args.dataset == 'icews18' else 'icews14_gpt_index.txt'
      f = open(f'./data/processed_new/{args.dataset}/test_rag/{rag_indexes}')
      indexes = eval(f.readline())
      f.close()

      filename = get_filename(args.dataset, dataset_path =  args.dataset_rag_path, model_name = args.finetuned_model)
    
    else:

      filename = get_filename(args.dataset, dataset_path = args.dataset_path, model_name = args.finetuned_model)

    with torch.no_grad(), open(filename, "w", encoding="utf-8") as writer, tqdm(test_set) as pbar:

      for i, x in enumerate(pbar):
          
          if test_set_rag:
            if i in indexes: # select the version of the prompt with extra info from rag, if in indexes
              j = indexes.index(i)
              x = test_set_rag[j]
              

          model_input = x['context']

          predictions = predict(tokenizer, model, model_input, args, output_text = True)

          text = model_input.split('\n')[-1] # get the target quadruple from input prompt
            
          # depending on the input prompt, the selection might need extra steps

          if test_set_rag and i in indexes:
            time = text.split(":")[1].strip()
            triple = text.split(":")[2].strip()
          else:
            time, triple = text.split(':')
          
          # get the relation and object; this is only needed to correctly write the results output file
          triple = triple.strip()
          obj, rel = triple.split(' ')
          obj = obj.strip()[1:-1] # get rid of paranthesis ("["") and comma
          rel = rel.strip()[:-1] # get rid of comma


          quad = [obj, rel, [x['target']], time.strip()]
          example = write_results(quad, predictions, 'tail', writer, args, []) # we predict the missing object, therefore "tail" prediction

          # update metrics with the new prediction
          update_metric(example, metric, args)
          pbar.set_postfix(metric.dump())
            
