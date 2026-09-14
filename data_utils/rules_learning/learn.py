import json
import os
import time
import argparse
import numpy as np
from collections import Counter
from datetime import datetime
from joblib import Parallel, delayed

from grapher import Grapher
from temporal_walk import Temporal_Walk
from rule_learning import Rule_Learner, rules_statistics
from basic import get_unique_quads_per_rels_cached

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", default="", type=str)
parser.add_argument("--rule_lengths", "-l", default="1", type=int, nargs="+")
parser.add_argument("--num_walks", "-n", default="200", type=int)
# exp is the published behaviour and stays the default; exp_scaled normalises the
# time gap by the timestamp spacing in the data. The value is recorded in the
# bank filename, so a bank says which transition produced it.
parser.add_argument("--transition_distr", default="exp", type=str,
                    choices=["unif", "exp", "exp_scaled"])
parser.add_argument("--num_processes", "-p", default=1, type=int)
parser.add_argument("--seed", "-s", default=None, type=int)
# choices, not free text: an unrecognised name matches none of the branches in
# learn_rules, so the run would mine nothing and still save an empty bank.
parser.add_argument("--mining", "-m", default="ragtkgc", type=str,
                    choices=["gtkg", "exhaustive", "ragtkgc", "ragtkgc_no_walks"])
parsed = vars(parser.parse_args())

dataset = parsed["dataset"]
rule_lengths = parsed["rule_lengths"]
rule_lengths = [rule_lengths] if (type(rule_lengths) == int) else rule_lengths
num_walks = parsed["num_walks"]
transition_distr = parsed["transition_distr"]
num_processes = parsed["num_processes"]
seed = parsed["seed"]
mining_alg = parsed['mining']

dataset_dir = "../../data/processed_new/" + dataset + "/"

# Phase origins for the timing breakdown. The mining timer alone hides the graph
# load and the candidate-set build, and the latter dominates for exhaustive.
t_start = time.time()
data = Grapher(dataset_dir)
t_graph = time.time()
temporal_walk = Temporal_Walk(data.train_idx, data.inv_relation_id, transition_distr)
rl = Rule_Learner(temporal_walk.edges, data.id2relation, data.inv_relation_id, dataset, mining_alg=mining_alg)
all_relations = sorted(temporal_walk.edges)  # Learn for all relations
t_init = time.time()

if mining_alg in ['ragtkgc','ragtkgc_no_walks', 'exhaustive']:
    # Cached on disk under data/processed_new/<dataset>/cache/, keyed by dataset,
    # period and infer_from_type. Delete that file to force a recomputation.
    # period=1: the quad file and ts2id.json already express time in the same
    # unit (day x 24), so the default period=24 multiplied it a second time.
    # sample_walk takes the start edge's timestamp straight from here and
    # compares it against Grapher's un-inflated timestamps, so an inflated value
    # makes the "strictly earlier" constraint admit later facts.
    if mining_alg in ['ragtkgc','ragtkgc_no_walks']:
        unique_quads = get_unique_quads_per_rels_cached(dataset, f'../../data/original/{dataset}/train.txt', period = 1)
    elif mining_alg == 'exhaustive':
        unique_quads = get_unique_quads_per_rels_cached(dataset, f'../../data/original/{dataset}/train.txt', period = 1, infer_from_type = True)

    unique_quads = dict(sorted(unique_quads.items(), key=lambda item: item[0], reverse = False))
t_quads = time.time()

#print(temporal_walk.edges[160])
#print(unique_quads[0])

def learn_rules(i, num_relations):
    """
    Learn rules (multiprocessing possible).

    Parameters:
        i (int): process number
        num_relations (int): minimum number of relations for each process

    Returns:
        rl.rules_dict (dict): rules dictionary
    """

    # `is not None`, not truthiness: 0 is a valid seed, and `if seed` would
    # silently leave the RNG unseeded for `--seed 0`.
    #
    # seed + i, not seed: joblib runs each worker in its own process and numpy's
    # global RNG is per-process, so a shared seed gives every worker the same
    # stream. Two relations with equal candidate counts then draw the same index
    # in lockstep, which leaves one independent stream rather than one per
    # worker. Upstream shares the seed; this deviates deliberately, because
    # bit-exact reproduction of the published bank is not attempted anyway.
    #
    # It does not make the bank independent of --num_processes: which relations a
    # worker handles still depends on the process count, so reproduction needs
    # the seed and the process count together.
    if seed is not None:
        np.random.seed(seed + i)

    nr_of_walks = 0
    nr_of_success_walks = 0

    num_rest_relations = len(all_relations) - (i + 1) * num_relations
    if num_rest_relations >= num_relations:
        relations_idx = range(i * num_relations, (i + 1) * num_relations)
    else:
        relations_idx = range(i * num_relations, len(all_relations))

    num_rules = [0]

    for k in relations_idx:
        
        rel = all_relations[k]

        for length in rule_lengths:
            
            it_start = time.time()

            if mining_alg == 'exhaustive':
                if length > 1:
                    print("EXHAUSTIVE only mines rules of length 1. Skipping relation {} for length {}.".format(rel, length))
                    break

                for st in unique_quads[rel]:
                    
                    walk = dict()
                    walk["head_rel"] = int(rel)
                    walk["body_rels"] = [int(st)]
                    walk["var_constraints"] = []
                    rl.create_rule(walk, custom_generated = True)


            if mining_alg == 'ragtkgc':
                
                len_unq_quads = len(unique_quads[rel])
                

                for _ in range(num_walks):
                    for q in range(len_unq_quads):
                        walk_successful, walk = temporal_walk.sample_walk(length + 1, rel, q, unique_quads)
                        nr_of_walks += 1
                        if walk_successful:
                            nr_of_success_walks += 1
                            rl.create_rule(walk)
            
            elif mining_alg == 'ragtkgc_no_walks':

                len_unq_quads = len(unique_quads[rel])
                

                for q in range(len_unq_quads):
                    walk_successful, walk = temporal_walk.sample_walk(length + 1, rel, q, unique_quads)
                    nr_of_walks += 1
                    if walk_successful:
                        nr_of_success_walks += 1
                        rl.create_rule(walk)

            elif mining_alg == 'gtkg': 

                for _ in range(num_walks):
                    walk_successful, walk = temporal_walk.sample_walk(length + 1, rel)
                    nr_of_walks += 1
                    if walk_successful:
                        nr_of_success_walks += 1
                        rl.create_rule(walk)

            it_end = time.time()
            it_time = round(it_end - it_start, 6)
            num_rules.append(sum([len(v) for k, v in rl.rules_dict.items()]) // 2)
            num_new_rules = num_rules[-1] - num_rules[-2]
            print(
                "Process {0}: relation {1}/{2}, length {3}: {4} sec, {5} rules".format(
                    i,
                    k - relations_idx[0] + 1,
                    len(relations_idx),
                    length,
                    it_time,
                    num_new_rules,
                )
            )

    #print(f'Number of total walks is {nr_of_walks}')
    #print(f'Number of total success walks is {nr_of_success_walks}\n')

    return rl.rules_dict


start = time.time()
num_relations = len(all_relations) // num_processes
output = Parallel(n_jobs=num_processes)(
    delayed(learn_rules)(i, num_relations) for i in range(num_processes))
end = time.time()

all_rules = output[0]
for i in range(1, num_processes):
    all_rules.update(output[i])

total_time = round(end - start, 6)
print("Learning finished in {} seconds.".format(total_time))

# The line above reports mining only, which is what earlier runs reported. The
# breakdown covers the whole invocation, so the reported cost can include the
# phases mining excludes.
timings = {
    "graph_load": round(t_graph - t_start, 2),
    "walker_and_learner_init": round(t_init - t_graph, 2),
    "unique_quads": round(t_quads - t_init, 2),
    "rule_mining": round(end - start, 2),
    "total": round(end - t_start, 2),
}
print("Phase timings (seconds):")
for _phase, _secs in timings.items():
    print("  {0:24s}: {1:.2f}".format(_phase, _secs))

rl.rules_dict = all_rules
rl.sort_rules_dict()
dt = datetime.now()
dt = dt.strftime("%d%m%y%H%M%S")
rules_file = rl.save_rules(dt, rule_lengths, num_walks, transition_distr, seed)
rl.save_rules_verbalized(dt, rule_lengths, num_walks, transition_distr, seed)
rules_statistics(rl.rules_dict)

# Persist the run alongside the bank it produced: the printed timings are the
# only record of what mining cost, and stdout does not survive the run.
lengths_mined = Counter(
    len(rule["body_rels"]) for rules in rl.rules_dict.values() for rule in rules
)
run_record = {
    "rules_file": rules_file,
    "dataset": dataset,
    "mining": mining_alg,
    "rule_lengths": rule_lengths,
    "num_walks": num_walks,
    "transition_distr": transition_distr,
    "num_processes": num_processes,
    # The bank is a function of the seed and the process count together: each
    # worker seeds with seed + its index, and which relations it handles depends
    # on how many workers there are.
    "seed": seed,
    "timings_seconds": timings,
    "relations_with_rules": len(rl.rules_dict),
    "total_rules": sum(len(v) for v in rl.rules_dict.values()),
    "rules_by_length": {str(k): v for k, v in sorted(lengths_mined.items())},
}
stats_dir = "../../logs"
stats_path = os.path.join(stats_dir, "learn_stats_{0}_{1}_{2}.json".format(dataset, mining_alg, dt))
try:
    os.makedirs(stats_dir, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as fout:
        json.dump(run_record, fout, indent=2)
    print("Run record written to: {0}".format(stats_path))
except OSError as exc:
    print("Could not write the run record ({0})".format(exc))
