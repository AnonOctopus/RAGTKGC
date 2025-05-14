import json
import math
import numpy as np
import torch


# function to decode permutations of logits

def permute(tokenizer, scores, cur_step, max_step, cur_seq, seqs, dec_cand, end_char):
    if cur_step == max_step or cur_step >= len(scores) or (len(cur_seq) > 0 and any(x in cur_seq[-1]["token"] for x in end_char)):
        _cur_seq = cur_seq[:-1].copy() if any(x in cur_seq[-1]["token"] for x in end_char) else cur_seq.copy()
        normalized_logit = (
            sum([x["logit"] for x in _cur_seq]) / len(_cur_seq) if len(_cur_seq) > 0 else -math.inf
        )
        seqs.append(
            {
                "tokens": [x["token"] for x in _cur_seq],
                "text": "".join([x["token"] for x in _cur_seq]).strip(),
                "probability": normalized_logit,
            }
        )
        return
    logits = scores[cur_step] 
    logits_indices = torch.argsort(logits, dim=-1, descending=True)
    for tok in logits_indices[0][:dec_cand]:
        #print(tok)
        cur_seq.append({"token": tokenizer.decode(tok), "logit": logits[0][tok].item()})
        permute(tokenizer, scores, cur_step + 1, max_step, cur_seq, seqs, dec_cand, end_char)
        cur_seq.pop()


def deduplicate(x):  # NOTE: assumes a sorted list based on probability
    f = {}
    z = []
    for y in x:
        if y[0] in f:
            continue
        f[y[0]] = True
        z.append(y)
    return z


def parse_results(results):
    #print('results',results)
    logprobs = [(x["text"], x["probability"]) for x in results]
    sorted_logprobs = sorted(logprobs, key=lambda tup: tup[1], reverse=True)
    dedup_sorted_logprobs = deduplicate(sorted_logprobs)

    probs = [x[1] for x in dedup_sorted_logprobs]
    softmax_probs = np.exp(probs) / np.sum(np.exp(probs), axis=0)

    to_return = [(x[0], p) for x, p in zip(dedup_sorted_logprobs, softmax_probs)]
    #print('to return', to_return)
    return to_return


def predict(tokenizer, model, prompt, args, output_text = False):
    
    tokenizer.pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    maxl = 45 # default; the max length of the output depends on the model and dataset; it is the maximum number of tokens that a target can have 

    if args.dataset == 'icews18':
        if args.base_model == 'TheBloke/Llama-2-7B-fp16': maxl = 35
        if args.base_model == 'google/flan-t5-small': maxl = 45
    
    if args.dataset == 'icews14':
        if args.base_model == 'TheBloke/Llama-2-7B-fp16': maxl = 26
        if args.base_model == 'google/flan-t5-small': maxl = 42

    if args.base_model == 'google/flan-t5-small':

        outputs = model.generate(
            **inputs,
            max_new_tokens=maxl,
            return_dict_in_generate=True,
            output_scores=True,
            renormalize_logits=True,
        )


        results = []

        permute(
                tokenizer,
                outputs.scores,
                0,
                maxl,
                [],
                results,
                1, # nr of logits to consider from each output score
                ["]","</s>"], # tokens when to stop decoding
                )
        
        results = list(sorted(results, key=lambda x: x["probability"], reverse=True))[:10] # keep the first 10 candidates
        
        if args.verbose:
                    
            for x in results:
                print(
                    f'| {json.dumps(x["tokens"]):30s} | {x["text"]:10s} | {x["probability"]:.4f} | {np.exp(x["probability"]):.2%}'
                    )

        predictions = parse_results(results)
        
        # method for decoding predictions for the second approach of calculating the metrics. Not reported
        # if used, make sure to also return pred and modify in run_hf.py to save it and pass it to write results; also, modify in utils.py (update_metrics) to calculate them using these predictions
        '''
        pred = ['','','']
        probability = [0,0,0]

        for i in range(len(outputs['scores'])):
          logits = outputs['scores'][i]
          logits_indices = torch.argsort(logits, dim=-1, descending=True)
          logits_values = torch.sort(logits, dim=-1, descending=True)
          for j in range(3):
    
            probability[j] += logits_values[0][0][j]
            pred[j] += tokenizer.decode(logits_indices[0][j])


        for i in range(len(probability)):

          probability[i] = np.exp(probability[i].item()/len(outputs['scores']))
        '''
    
    elif args.base_model == 'TheBloke/Llama-2-7B-fp16':

        beam_outputs = model.generate(
            **inputs,
            max_new_tokens=maxl,
            num_beams=3,
            num_return_sequences=3,
            early_stopping=True,
        )

    
        predictions = []
        for i, beam_output in enumerate(beam_outputs):
            beam_pred = tokenizer.decode(beam_output[len(inputs['input_ids'][0]):], skip_special_tokens=True).replace(']','').replace('</s>','').strip()
            
            if 'gtkg' in args.finetuned_model:
                predictions.append(beam_pred.split('\n')[0].strip()) # needs an extra processing step, as this variants predict more than the target
            else: predictions.append(beam_pred)

            if args.verbose:
                print("{}: {}".format(i,beam_pred ))
        
        

    return predictions
