import json
import numpy as np
from openai import OpenAI
from collections import Counter
from math import sqrt
from tqdm import tqdm
from sentence_transformers import util
from sentence_transformers import SentenceTransformer

ff = ['results7.jsonl','results8.jsonl','results9.jsonl','results10.jsonl', 'results11.jsonl', 'results12.jsonl']
metrics = {}

for f in ff:
    with open(f'./{f}', 'r', encoding='utf-8') as results:
        lines = results.readlines()
        for i, line in enumerate(lines):
            
            d = eval(line)
            pred = d['predictions']
            if d['targets'][0] not in pred[0]:
                acc += 1
                if i not in metrics.keys():
                    metrics[i] = [1]
                else:
                    metrics[i][0] += 1
            count += 1

sorted_d = dict(sorted(metrics.items(),  key= lambda item: item[1][0], reverse = True))

#print(groups)
fff = ['../data/processed_new_ici/icews18/splits_rl_1/uw-text/test/10000/lora_test/icews18_uw_test.json']
       #,'../data/processed_new_ici/icews14/splits_rl_1/u-text/test/lora_test/icews14_test.json']

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
                sorted_d[i].append(count)
                sorted_d[i].append(input)
                sorted_d[i].append(prompt)
                sorted_d[i].append(line['target'])
                
                

groups = []
i_d = {}
i_v = 999
for i,(k, v) in enumerate(sorted_d.items()):
    
    if v[0] != i_v:
        
        i_v = v[0]
        if i_d:
            print(len(i_d))
            groups.append(i_d)
        i_d = {k:v}
    else:
        i_d[k] = v
    
    if i == len(sorted_d) - 1:
        print(len(i_d))
        groups.append(i_d)

#print(groups)

for d in groups:
    val = list(d.values())
    #print(val)
    #print(np.average(val[:][1]))

def read_json(json_dir):
    with open(json_dir, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    return json_data

def flip_dict(original_dict):
    return {v: k for k, v in original_dict.items()}

def get_rules(file, conf_treshold = 0.5):

    final_rules = {}

    with open(f'../data/processed_new_ici/icews18/output/icews18/{file}', 'r', encoding='utf-8') as rules:

        lines = rules.readlines()
       
        for line in lines: 
            
            s_line = line.split(' ')
            #print(s_line)
            s_line = [x for x in s_line if x]
            #print(s_line)
            conf = s_line[0]
            rule = s_line[3] + s_line[4] + s_line[5].replace('\n','')
            #print(conf, rule)
            
            if float(conf) >= conf_treshold:
                
                rel = rule.split('X0')[0][:-1]
                
                if rel in final_rules.keys():

                    final_rules[rel].append(rule)
                else:
                    final_rules[rel] = [rule]
            
    return final_rules

def get_prompt(input_text, withRules = True, withRels = True):

    #entities = read_json('../data/processed_new_ici/icews14/entity2id.json') 
    rels = read_json('../data/processed_new_ici/icews18/relation2id.json') 
    timestamps = read_json('../data/processed_new_ici/icews18/ts2id.json')
    timestamps = flip_dict(timestamps)

    quad = input_text.split(' ')
    time = timestamps[int(quad[0][:-3])*24]
    rel = quad[2][:-1]

    final_quad = f"{time}: {quad[1]} {quad[2]}"

    #print(final_quad)
    if withRules:
        dd_rules = get_rules('050525174831_r[1]_n200_exp_s1_rules.txt')
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

def cosine_similarity(target, object):

    vec1 = Counter(target)
    vec2 = Counter(object)
    dot_product = sum(vec1[ch] * vec2[ch] for ch in vec1)
    magnitude1 = sqrt(sum(count ** 2 for count in vec1.values()))
    magnitude2 = sqrt(sum(count ** 2 for count in vec2.values()))
    return dot_product / (magnitude1 * magnitude2) if magnitude1 != 0 and magnitude2 != 0 else 0

def cosine_similarity_llm(target, object, model):

    
    #print('I am calculating embeddings')
    embeddings = model.encode(object, convert_to_tensor=True)
    #print('I am calculating prediction embeddings')
    prediction_embeddings =  model.encode([target], convert_to_tensor=True)
    #print("done")
    return util.pytorch_cos_sim(prediction_embeddings, embeddings)[0]

def get_gpt_response(prompt, user, model, useLLM = False):

    client = OpenAI(api_key = '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') # load YOUR OWN KEY; here stored as a secret in the google colab
        
    req_headers = ['openai-model', 'openai-processing-ms', 'x-ratelimit-remaining-requests', 'x-ratelimit-remaining-tokens', 'x-ratelimit-reset-requests']


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

    entities = read_json('../data/processed_new_ici/icews18/entity2id.json') 
    rels = read_json('../data/processed_new_ici/icews18/relation2id.json') 
    timestamps = read_json('../data/processed_new_ici/icews18/ts2id.json')

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
        
        if year <= 2018:
            #print(year)
            if year == 2018: 
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

#prompt, user = get_prompt(groups[0])

prompts = []
prompts2 = []
prompts3 = []

model = SentenceTransformer('all-MiniLM-L6-v2')

for i, (k,v) in tqdm(enumerate(groups[0].items())):
    
    prompt, user = get_prompt(v[2], False, False) # True/False pt reguli, False/True pt rel, False/False pt nimic
    #print(prompt, user)
    history, history2, history3 = get_gpt_response(prompt, user, model, False) 
    #print(history, history2, history3)
    to_write = {"context" : f"{v[3]}The above set of quadruples are factually correct. Firstly try to formulate your answer on them. If you are unsure of your answer, here are additional quadruples:\n{history}Input quadruple: {v[2]}",
                                        "target" : f"{v[4]}"
                                        }
    #to_write2 = {"context" : f"{v[3]}The above set of quadruples are factually correct. Firstly try to formulate your answer on them. If you are unsure of your answer, here are additional quadruples:\n{history2}Input quadruple: {v[2]}",
    #                                    "target" : f"{v[4]}"
    #                                    }
    to_write3 = {"context" : f"{v[3]}The above set of quadruples are factually correct. Firstly try to formulate your answer on them. If you are unsure of your answer, here are additional quadruples:\n{history3}Input quadruple: {v[2]}",
                                        "target" : f"{v[4]}"}
      
    with open('icews18_test_free_cs_4000_temp.json', 'a', encoding='utf8') as jsontext:
        jsontext.write(f"{json.dumps(to_write)},")

    #with open('lora_test_gpt_5_csllm_temp.json', 'a', encoding='utf8') as jsontext:
    #    jsontext.write(f"{json.dumps(to_write2)},")

    with open('icews18_test_free_nocs_4000_temp.json', 'a', encoding='utf8') as jsontext:
        jsontext.write(f"{json.dumps(to_write3)},")

    prompts.append(to_write)
    #prompts2.append(to_write2)
    prompts3.append(to_write3)
                                        
    if i == 3999: break
    #break
with open('icews18_test_free_cs_4000.json', 'w', encoding='utf8') as jsontext:
    json.dump(prompts, jsontext)
#with open('lora_test_gpt_5_csllm.json', 'w', encoding='utf8') as jsontext:
#    json.dump(prompts2, jsontext)
with open('icews18_test_free_nocs_4000.json', 'w', encoding='utf8') as jsontext:
    json.dump(prompts3, jsontext)

#print(list(groups[0].keys())[:4000])