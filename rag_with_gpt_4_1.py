import argparse
import json
import numpy as np
from openai import OpenAI
from collections import Counter
from math import sqrt
from tqdm import tqdm
from sentence_transformers import util
from sentence_transformers import SentenceTransformer


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='icews18', type=str)
    parser.add_argument("--rule_file", default='050525174831_r[1]_n200_exp_s1_rules.txt', type=str)
    parser.add_argument("--conf_treshold", default=0.5, type=float)
    parser.add_argument("--rag_version", choices=['gpt-given-rules','gpt-given-relations','gpt-rule-miner'], default='gpt-given-rules', type=str)
    parser.add_argument("--use_llm_similarity", default= False, type=bool)
    parser.add_argument("--no_similarity", default= False, type=bool)

    args = parser.parse_args()
    return args
    
# read all results file and count for each test sample how many times it was wrongly predicted
# we save everything in a dictionary where the key is the index of the sample, 
# while the value is a list, where the first element is the count of wrong predictions

def get_wrong_predictions_count(args):

    ff = [f'flan-t5-small-{args.dataset}-raw_{args.dataset}_raw_test',f'flan-t5-small-{args.dataset}-standard_{args.dataset}_standard_test', f'flan-t5-small-{args.dataset}-gtkg_{args.dataset}_gtkg_test', f'flan-t5-small-{args.dataset}-ragtkgc_{args.dataset}_ragtkgc_test',f'llama-2-7B-{args.dataset}-raw_{args.dataset}_raw_test', f'llama-2-7B-{args.dataset}-standard_{args.dataset}_standard_test',f'llama-2-7B-{args.dataset}-gtkg_{args.dataset}_gtkg_test',f'llama-2-7B-{args.dataset}-ragtkgc_{args.dataset}_ragtkgc_test']
    
    len_ff = len(ff)
    metrics = {}

    for f in ff:
        with open(f'./results/{args.dataset}/{f}.jsonl', 'r', encoding='utf-8') as results:
            lines = results.readlines()
            for i, line in enumerate(lines):
                
                d = eval(line)
                pred = d['predictions']
                if d['targets'][0] not in pred[0]:

                    if i not in metrics.keys():
                        metrics[i] = [1]
                    else:
                        metrics[i][0] += 1

    # sort them in descending order, starting from those test samples with lowest prediction power
    sorted_d = dict(sorted(metrics.items(),  key= lambda item: item[1][0], reverse = True))

    return sorted_d, len_ff

# we need to extract sample information such as target quad and associated history in order to generate additional data later with GPT 4.1

def get_sample_info(args):

    sorted_d, len_ff = get_wrong_predictions_count(args)
    
    # because the ragtkgc history modeling was the most efficient one, we build the new rag test prompts with this histories to which we'll append the ones obtained with rag
    fff = [f'./data/processed_new/{args.dataset}/ragtkgc/test/10000/history_modeling_test/{args.dataset}_ragtkgc_test.json' if args.dataset == 'icews18' else f'./data/processed_new/{args.dataset}/ragtkgc/test/history_modeling_test/{args.dataset}_ragtkgc_test.json']

    for fl in fff:

        with open(fl, 'r', encoding='utf-8') as test:

            lines = json.load(test)

            for i,line in enumerate(lines):
                
                if i in sorted_d.keys():
                    
                    quads = line['context'].split('\n')
                    prompt = ''
                    for quad in quads[:-1]:
                        prompt += f'{quad}\n'
                    input = quads[-1]
                    count = line['context'].count('[')
                    sorted_d[i].append(count) # add how many quads are in ith sample associated history
                    sorted_d[i].append(input) # target (incomplete) quad
                    sorted_d[i].append(prompt) # associated history
                    sorted_d[i].append(line['target']) # target value
    
    return sorted_d, len_ff

def group_predictions(args):
    
    sorted_d, len_ff = get_sample_info(args)

    # group together test samples based on the count of wrong predictions 
    groups = []
    i_d = {}
    i_v = 999 # dummy value to know when a group of test samples was processed and another one starts

    for i,(k, v) in enumerate(sorted_d.items()):
        
        if v[0] != i_v: # if true, then we have a new group to process
            
            i_v = v[0]
            if i_d:
                #print(len(i_d))
                groups.append(i_d) # save the already existing group
            i_d = {k:v} # start the new group

        else: # if false, continue adding the test samples to the group
            i_d[k] = v
        
        if i == len(sorted_d) - 1: # save the last group too
            #print(len(i_d))
            groups.append(i_d)


    return groups, len_ff, sorted_d

def read_json(json_dir):
    with open(json_dir, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    return json_data

def flip_dict(original_dict):
    return {v: k for k, v in original_dict.items()}

# retrieves all rules with a confidence greater or equal with a given treshold

def get_rules(args):

    final_rules = {}

    with open(f'./data/processed_new/{args.dataset}/output/{args.dataset}/{args.rule_file}', 'r', encoding='utf-8') as rules:

        lines = rules.readlines()
       
        for line in lines: 
            
            s_line = line.split(' ')

            s_line = [x for x in s_line if x]
  
            conf = s_line[0]
            rule = s_line[3] + s_line[4] + s_line[5].replace('\n','')
  
            
            if float(conf) >= args.conf_treshold:
                
                rel = rule.split('X0')[0][:-1]
                
                if rel in final_rules.keys():

                    final_rules[rel].append(rule)
                else:
                    final_rules[rel] = [rule]
            
    return final_rules

# create the prompt for each test sample that will retrieve extra information

def get_prompt(input_text, args, withRules = True, withRels = True):

    rels = read_json(f'./data/processed_new/{args.dataset}/relation2id.json') 
    timestamps = read_json(f'./data/processed_new/{args.dataset}/ts2id.json')
    timestamps = flip_dict(timestamps)

    quad = input_text.split(' ')
    time = timestamps[int(quad[0][:-3])*24]
    rel = quad[2][:-1]

    final_quad = f"{time}: {quad[1]} {quad[2]}"

    #print(final_quad)
    if withRules:
        dd_rules = get_rules(args)
        if rel in dd_rules.keys():
            d_rules = dd_rules[rel]
        else:
            d_rules = []
        rules = '\n'
        for rule in d_rules:

            rules += f'\t{rule}\n'

        return f'''
    You will work with dates, in the form of YYYY-MM-DD, starting from 2018-01-01 to 2018-10-31, with an interval of a day. For example, 2018-01-01, 2018-01-02 and so on.
    You will deal with quads in the form of "date: [subject, relationship, object]", where date follows the mentioned format, subject and object are entities, and relationship is from the set of rules. You will receive a target quad with a missing object.
    You will also receive a set of rules that tells you what other chained relationships might happen before the target one. For example: "Threaten(X0,X1,T3) <- Consult(X0,X1,T0), _Criticize_or_denounce(X1,X0,T1), Consult(X0,X1,T2)" translates into "before X0 threatened X1 on date T3, X0 consulted T1 on date T0, X1 was criticized or denounced by X0 on date T1 and X0 consulted X1 on date T2". Notice that one relationship has an underscore(_) at the beginning. This means that it is an inverse relationship that points from object to subject, so we can travel backwards and return to the subject. This means there were no relationships pointing from X1 to X0, but there was an inverse relationship before T2 that connected back to X0.
    Your task is to use the set of given rules and dates to construct a history of events in quad format that may lead to finding the missing object from the given quad. Base your response on facts you were trained on. All information must be factually correct.
    Make sure rules are followed. Try to find information in between the given dates interval. If you find information before 2018-01-01, you can use it, just output the date in the given format.
    Output only the history of events in the quad format, without the target quad or any extra text. Order them chronologically. If you find no information, output "None".
    ''',f'''
    Given rules: {rules}
    Target quad: {final_quad}
            '''
    
    elif withRels:
        return f'''
    You will work with dates, in the form of YYYY-MM-DD, starting from 2018-01-01 to 2018-10-31, with an interval of a day. For example, 2018-01-01, 2018-01-02 and so on.
    You will deal with quads in the form of "date: [subject, relationship, object]", where date follows the mentioned format, subject and object are entities, and relationship is from the given relationships' list. You will receive a target quad with a missing object.
    You will also receive a list of relationships that may have happened between the subject and the object. Use those names whenever you find information that fits or is close to any of the given relationship name.
    Your task is to use the list of relationships and dates to construct a history of events in quad format that may lead to finding the missing object from the given quad. Base your response on facts you were trained on. All information must be factually correct.
    Make sure relationship names are within the given ones. Try to find information in between the given dates interval. If you find information before 2018-01-01, you can use it, just output the date in the given format.
    Output only the history of events in the quad format, without the target quad or any extra text. Order them chronologically. If you find no information, output "None".
    ''',f'''
    Given relationships: {list(rels.keys())}
    Target quad: {final_quad}
            '''
    
    else:
        return f'''
    You will work with dates, in the form of YYYY-MM-DD, starting from 2018-01-01 to 2018-10-31, with an interval of a day. For example, 2018-01-01, 2018-01-02 and so on.
    You will deal with quads in the form of "date: [subject, relationship, object]", where date follows the mentioned format, subject and object are entities. You will receive a target quad with a missing object.
    Your task is to construct a history of events in quad format that may lead to finding the missing object from the given quad. Base your response on facts you were trained on. All information must be factually correct.
    Make sure relationship names are concised, and use an underscore between words forming the relationship name, not spaces. Try to find information in between the given dates interval. If you find information before 2018-01-01, you can use it, just output the date in the given format.
    Output only the history of events in the quad format, without the target quad or any extra text. Order them chronologically. If you find no information, output "None".
    ''',f'''
    Target quad: {final_quad}
            '''
        #    List of entities: {list(entities.keys())} Be sure to exactly use words as they were given in the list. You will receive a list of entities

# calculate cosine similarity using classical approach
def cosine_similarity(target, object):

    vec1 = Counter(target)
    vec2 = Counter(object)
    dot_product = sum(vec1[ch] * vec2[ch] for ch in vec1)
    magnitude1 = sqrt(sum(count ** 2 for count in vec1.values()))
    magnitude2 = sqrt(sum(count ** 2 for count in vec2.values()))
    return dot_product / (magnitude1 * magnitude2) if magnitude1 != 0 and magnitude2 != 0 else 0

# calculate cosine similarity using llms
def cosine_similarity_llm(target, object, model):

    
    #print('I am calculating embeddings')
    embeddings = model.encode(object, convert_to_tensor=True)
    #print('I am calculating prediction embeddings')
    prediction_embeddings =  model.encode([target], convert_to_tensor=True)
    #print("done")
    return util.pytorch_cos_sim(prediction_embeddings, embeddings)[0]

def get_gpt_response(args, prompt, user, model, useLLM = False):

    api_key = ''

    with open('api_key.txt', 'r') as ak:
        api_key = ak.readline()

    client = OpenAI(api_key = api_key) # load YOUR OWN KEY

    try:
        completion = client.chat.completions.with_raw_response.create(
                model='gpt-4.1-2025-04-14',
                messages=[{'role': 'system', 'content': prompt},{'role': 'user', 'content': user}]
            )
    except Exception as e:
        print(e)
        print(f"er1ror:") # if errors are generated from the server

    generated_response = completion.parse()

    resp = generated_response.choices[0].message.content # get the object that `chat.completions.create()` would have returned
    
    #print(gpt_response)

    rels = read_json(f'./data/processed_new/{args.dataset}/relation2id.json') 
    timestamps = read_json(f'./data/processed_new/{args.dataset}/ts2id.json')
    entities = read_json(f'./data/processed_new/{args.dataset}/entity2id.json') 

    quads = resp.splitlines() if resp.strip() != "None" else []
    history = ''
    history2 = ''
    history3 = ''

    for quad in quads:
        
        #print(quad)
        needS = False
        needO = False
        needR = False
        hasUnderscore = False

        #print(quad)
        try:
            time, q = quad.split(":")
        except:
            print(quad)
           #print(quad.split(":"))
            continue
        
        try:
            s, r, o = q.strip().split(', ')
        except:
            print(q)
            #print(q.strip().split(', '))
            continue

        s = s[1:].strip()

        if r.startswith('_'):
            r = r[1:]
            hasUnderscore = True

        o = o[:-1].strip()

        try:
            year = int(time.split('-')[0])
        except:
            continue
        
        target_year = 2014 if args.dataset == 'icews14' else 2018

        if year <= target_year:
            #print(year)
            if year == target_year: 
              if time in timestamps.keys():
                time = timestamps[time]/24
              else: 
                print('date not in ts')
                continue
            else: time = -1.0
        else:
            continue
        
        history3 += f'{time}: [{s}, {r}, {o}]\n'

        if s not in entities.keys():
            #print(f'{s} not in known entities')
            needS = True
        if o not in entities.keys():
            #print(f'{o} not in known entities')
            needO = True
        if r not in rels.keys():
            #print(f'{r} not in known entities')
            needR = True

        llm_s = s
        llm_r = r
        llm_o = o

        if needR:

            original_r = r

            if useLLM:
                cs = cosine_similarity_llm(r, list(rels.keys()), model)
                llm_r = list(rels.keys())[np.argmax(cs.cpu())]

            max_cs = 0 
            
            for rel in rels.keys():
                cs = cosine_similarity(original_r, rel)
                if cs > max_cs:
                    max_cs = cs
                    r = rel

        if needS:

            original_s = s

            if useLLM:
                cs = cosine_similarity_llm(s, list(entities.keys()), model)
                llm_s = list(entities.keys())[np.argmax(cs.cpu())]

            max_cs_s = 0 
            
            for e in entities.keys():
                cs = cosine_similarity(original_s, e)
                if cs > max_cs_s:
                    max_cs_s = cs
                    s = e

        if needO:

            original_o = o

            if useLLM:
                cs = cosine_similarity_llm(o, list(entities.keys()), model)
                llm_o = list(entities.keys())[np.argmax(cs.cpu())]

            max_cs_o = 0 

            for e in entities.keys():
                cs = cosine_similarity(original_o, e)
                if cs > max_cs_o:
                    max_cs_o = cs
                    o = e

        if hasUnderscore: r = '_' + r
        #print(time, s, r, o)
        history += f'{time}: [{s}, {r}, {o}]\n'
        history2 += f'{time}: [{llm_s}, {llm_r}, {llm_o}]\n'
        #print(history)
    return history, history2, history3


if __name__ == "__main__":

    args = parser()

    groups, nr_of_results_files, sorted_d = group_predictions(args)

    total_length = 0

    for g in groups:
        print(f'There are {len(g.items())} samples where they were wrongly predicted {list(g.values())[0][0]}/{nr_of_results_files} times.')
        total_length += len(g.items())
    
    # we continue taking from the next group if one is less than the requested number
    nr_of_samples = input(f'How many samples do you want to extend with RAGTKGC with GPT 4.1? (can be max dataset length = {total_length}): ')
    try:
        nr_of_samples = int(nr_of_samples)
    except:
        print("Please input an integer number")

    if nr_of_samples > total_length:
        print('The number of samples is beyond the length of the dataset. Choose a smaller number.')
    
    else:


        prompts = []
        prompts2 = []
        prompts3 = []

        

        for i, (k,v) in tqdm(enumerate(sorted_d.items())):
            
            #./data/processed_mew/{args.dataset}/test_rag/
            file_name = f'{args.dataset}_{args.rag_version}'

            if args.rag_version == 'gpt-given-rules':

                prompt, user = get_prompt(v[2], args, True, False)

            elif args.rag_version == 'gpt-given-relations':

                prompt, user = get_prompt(v[2], args, False, True)

            elif args.rag_version == 'gpt-rule-miner':

                prompt, user = get_prompt(v[2], args, False, False)
            

            if args.use_llm_similarity: model = SentenceTransformer('all-MiniLM-L6-v2')
            else: model = False

            # first is using cosine similarity, second is using LLM similarity (if set to False, it returns the same as history3), and third leaves the entities/relationships untouched, as output by gpt 4.1
            history, history2, history3 = get_gpt_response(args, prompt, user, model, args.use_llm_similarity) 
            
            # first we save everything in a temporary file, in case any error stops the process; if True, restart the process but ignore the first i already done ones
            to_write = {"context" : f"{v[3]}The above set of quadruples are factually correct. Firstly try to formulate your answer on them. If you are unsure of your answer, here are additional quadruples:\n{history}Input quadruple: {v[2]}",
                                                "target" : f"{v[4]}"}
            with open(f'{file_name}_temp.json', 'a', encoding='utf8') as jsontext:
                jsontext.write(f"{json.dumps(to_write)},")
            prompts.append(to_write)

            if args.use_llm_similarity:
                to_write2 = {"context" : f"{v[3]}The above set of quadruples are factually correct. Firstly try to formulate your answer on them. If you are unsure of your answer, here are additional quadruples:\n{history2}Input quadruple: {v[2]}",
                                                    "target" : f"{v[4]}"}
                with open(f'{file_name}_llm_sim_temp.json', 'a', encoding='utf8') as jsontext:
                    jsontext.write(f"{json.dumps(to_write2)},")
                prompts2.append(to_write2)

            if args.no_similarity:
                to_write3 = {"context" : f"{v[3]}The above set of quadruples are factually correct. Firstly try to formulate your answer on them. If you are unsure of your answer, here are additional quadruples:\n{history3}Input quadruple: {v[2]}",
                                                    "target" : f"{v[4]}"}
                with open(f'{file_name}_no_sim_temp.json', 'a', encoding='utf8') as jsontext:
                    jsontext.write(f"{json.dumps(to_write3)},")
                prompts3.append(to_write3)
                                                
            if i == nr_of_samples - 1: break
        
        # save the final new prompts
        with open(f'{file_name}.json', 'w', encoding='utf8') as jsontext:
            json.dump(prompts, jsontext)
        
        if args.use_llm_similarity:
            with open(f'{file_name}_llm_sim.json', 'w', encoding='utf8') as jsontext:
                json.dump(prompts2, jsontext)
        
        if args.no_similarity:
            with open(f'{file_name}_no_sim.json', 'w', encoding='utf8') as jsontext:
                json.dump(prompts3, jsontext)

        # save the index list of those samples that benefited from extra information, it will be needed in the testing part as we select samples from their normal test file as well as this ones
        with open(f'{args.dataset}_gpt_index.json', 'w', encoding='utf8') as jsontext:
            jsontext.write(str(list(sorted_d.keys())[:nr_of_samples]))