import json
import csv
import os
import pickle
import random
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np


def get_rels_per_node(path_file, rel2id_file):
    
    f = open(path_file, encoding = 'utf8')
    lines = f.readlines()
    nodes = {}
   
    for l in lines:
        quad = [int(ll.strip('\n').strip()) for ll in l.split('\t')]
        

        q0 = quad[0]
        q2 = quad[2]

        if q0 in nodes.keys():
            if quad[1] not in nodes[q0]:
                nodes[q0].append(quad[1])
        else:
            nodes[q0] = [quad[1]]
        
        
        reverse_rel = quad[1] + len(rel2id_file.keys())
        
        if q2 in nodes.keys():
            
            if reverse_rel not in nodes[q2]:
                nodes[q2].append(reverse_rel)
        else:
            nodes[q2] = [reverse_rel]
    
    f.close()
    
    return dict(sorted(nodes.items(), key=lambda item: item[1], reverse = True))

def get_unique_quads(path, rel2id_file):
    
    nodes_rels = get_rels_per_node(path, rel2id_file)

    for v in nodes_rels.values():
        v.sort()

    values = nodes_rels.values()
    values = list(values)
    #print(len(values))
    values = [list(item) for item in set(tuple(row) for row in values)]
    #print(len(values))
    #print(values)
    for k,v in nodes_rels.items():
        if v in values:
            nodes_rels[k] = values.index(v)

    return dict(sorted(nodes_rels.items(), key=lambda item: item[1], reverse = False)), values

def get_unique_quads_per_rels(dataset, path, period = 24, infer_from_type = False):

    def add_quad(quad, quads_all, quads, unique_nodes):

        quad = [quad[0],quad[1],quad[2],quad[3] * period]
        
        if quad[1] in quads_all.keys():
            
            quads_all[quad[1]].append(quad)

        else:
            quads_all[quad[1]] = [quad]

        if infer_from_type:
            
            def infer_quad(quad):
                    
                    type_s = unique_nodes[quad[0]]
                    type_o = unique_nodes[quad[2]]
                    
                    if quad[1] in already_inferred.keys():
                        already_inferred[quad[1]].append(np.array([type_s,type_o]))
                    else:
                        already_inferred[quad[1]] = [np.array([type_s,type_o])]
                    
                    for st in types[type_s]:
                        if st + len(id2rel.keys()) in types[type_o]:
                            if quad[1] in quads.keys():
                                quads[quad[1]].add(st)
                            else:
                                quads[quad[1]] = set([st])

                        elif st - len(id2rel.keys()) in types[type_o]:
                            if quad[1] in quads.keys():
                                quads[quad[1]].add(st)
                            else:
                                quads[quad[1]] = set([st])

            if quad[1] in already_inferred.keys():
                
                flag = False

                for q in already_inferred[quad[1]]:
                    
                    if q[0] == unique_nodes[quad[0]] and q[1] == unique_nodes[quad[2]]:
                        flag = True
                        break
                
                if flag == False:
                    infer_quad(quad)
            
            else:
                infer_quad(quad)
            
        
        else: 
            if quad[1] in quads.keys():
                
                flag = False

                for q in quads[quad[1]]:
                    #print(q)
                    if unique_nodes[q[0]] == unique_nodes[quad[0]] and q[1] == quad[1] and unique_nodes[q[2]] == unique_nodes[quad[2]]:
                        flag = True
                        break
                if flag == False:
                        quads[quad[1]].append(np.array(quad))

            else:
                quads[quad[1]] = [np.array(quad)]

   
    #print(unique_nodes)
    f = open(path, encoding = 'utf8')
    rel2id_file = open(f'../../data/original/{dataset}/relation2id.json', encoding = 'utf8')
    id2rel = dict([(v, k) for k, v in json.load(rel2id_file).items()])
    unique_nodes, types = get_unique_quads(path, id2rel)
    #print(unique_nodes)
    output = open(f'../../data/processed_new/{dataset}/node_labelling_stats_{dataset}.txt', 'w')

    
    lines = f.readlines()
    quads = {}
    quads_all = {}
    already_inferred = {}
    total_unique_quads = 0
    total_quads = 0


    for l in tqdm(lines):

        quad = [int(ll.strip('\n').strip()) for ll in l.split('\t')]
        
        reverse_quad_split = l.split('\t')
        reverse_quad = [int(reverse_quad_split[2].strip('\n').strip()),int(reverse_quad_split[1].strip('\n').strip()) + len(id2rel.keys()),int(reverse_quad_split[0].strip('\n').strip()),int(reverse_quad_split[3].strip('\n').strip()) ]
    
        add_quad(quad, quads_all, quads, unique_nodes)
        add_quad(reverse_quad, quads_all, quads, unique_nodes)
            

    # zip pairs the two dicts by insertion order, not by key: the relation name
    # comes from `quads` while the total count comes from `quads_all`. This is
    # only correct because a head relation is always its own type-compatible
    # body, so every key entering quads_all also enters quads at the same moment.
    # If the admission test changes, iterate one dict and look the other up.
    for (k, v),(kk,vv) in zip(quads.items(),quads_all.items()):

        message = f'Relationship {id2rel[k % len(id2rel.keys())]} ({k}) has total quads - {len(vv)} and total unique quads - {len(v)}\n' # Heterogeneous rate: {heterogeneous_rate}%\n'

        output.write(message)


    f.close()
    rel2id_file.close()
    output.close()

    return quads


def get_unique_quads_per_rels_cached(dataset: str, path: str, period: int = 24,
                                     infer_from_type: bool = False) -> dict:
    """Admitted bodies per relation, reused from disk when already computed.

    Args:
        dataset: dataset name, e.g. "icews14".
        path: quad file to read.
        period: multiplier applied to each quad's timestamp. Names the node-mode
            cache, because changing it changes every timestamp in the result.
        infer_from_type: admit bodies by relation-signature compatibility
            rather than by node identity. Also part of the cache name, because
            it changes the type of the values, not only their content.

    Returns:
        dict: relation id to admitted bodies — a list of quad arrays when
            infer_from_type is False, a set of relation ids when it is True.
    """
    # period scales the timestamp column, which the type mode never reads, so
    # its result is invariant to period and the name leaves it out.
    stamp = "type" if infer_from_type else f"p{period}_node"
    cache_dir = f"../../data/processed_new/{dataset}/cache"
    cache_path = os.path.join(cache_dir, f"unique_quads_{dataset}_{stamp}.pkl")

    # pickle rather than JSON: the two modes return different value types
    # (list of arrays, set of ints) and the keys are ints, none of which JSON
    # represents. Delete the file to force a recomputation.
    try:
        with open(cache_path, "rb") as handle:
            quads = pickle.load(handle)
        print(f"unique_quads loaded from cache: {cache_path}")
        return quads
    except FileNotFoundError:
        pass
    except (OSError, pickle.UnpicklingError, EOFError) as exc:
        # An interrupted write leaves a truncated file. Recomputing is always
        # correct, so a damaged cache must not be fatal.
        print(f"ignoring unreadable unique_quads cache ({exc}); recomputing")

    quads = get_unique_quads_per_rels(dataset, path, period, infer_from_type)

    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "wb") as handle:
            pickle.dump(quads, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"unique_quads cached to: {cache_path}")
    except OSError as exc:
        print(f"could not write unique_quads cache ({exc})")

    return quads


if __name__ == "__main__":
    dataset = 'icews14'
    with open(f'../../data/original/{dataset}/relation2id.json', encoding = 'utf8') as f:
        rel2id_file = json.load(f)
    path = f'../../data/original/{dataset}/train.txt'
    #print(get_unique_quads(path, rel2id_file))
    #print(get_unique_quads_per_rels(dataset, path, infer_from_type=True))