import argparse
from dataclasses import dataclass
import json
import os
import random
from typing import Any, Dict, List, Optional
from collections import Counter
from math import sqrt
import numpy as np
from numpy import False_

MAX_HITS = 10


@dataclass
class HitsMetric:
    
    total: int = 0
    
    # prediction needs to exactly match the target
    hit1: int = 0
    hit3: int = 0
    hit10: int = 0
    
    
    
    # target needs to just be in the prediction, e.g. prediction = "China]", target = "China". 
    # However, this metric is tricky, as prediction = "Chinatown" and target = "China" would count as correct. Therefore, we use the above metrics
    total2: int = 0
    hit1p: int = 0
    hit3p: int = 0
    hit10p: int = 0

    cosine_sim: float = 0.0


    def update(self, rank):
        if rank <= 1:
            self.hit1 += 1
        if rank <= 3:
            self.hit3 += 1
        if rank <= 10:
            self.hit10 += 1
    
    def update2(self, rank):

        if rank <= 1:
            self.hit1p += 1
        if rank <= 3:
            self.hit3p += 1
        if rank <= 10:
            self.hit10p += 1

    def update3(self, cs):
      self.cosine_sim += cs


    def dump(self):
        return {
            "t": self.total,
            "hit1": self.hit1 / self.total,
            "hit3": self.hit3 / self.total,
            "hit10": self.hit10 / self.total,
            #"t2": self.total,
            #"hit1p": self.hit1p / self.total2,
            #"hit3p": self.hit3p / self.total2,
            #"hit10p": self.hit10p / self.total2,
            "cs": self.cosine_sim / self.total
        }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="google/flan-t5-small", type=str) # the base (foundation) model that the fine tuned version is based on
    parser.add_argument("--finetuned_model", type=str) # name of the folder which holds the fine tuned model version or peft adapter
    parser.add_argument(
        "--dataset",
        choices=["icews14", "icews18"],
        default="icews18",
        type=str,
    )
    parser.add_argument("--dataset_path", default="ragtkgc/test/10000/history_modeling_test/icews18_test.json", type=str) # path to the test set
    parser.add_argument("--dataset_rag_path", default="test_rag/icews18_gpt_rule_miner.json", type=str) # path to the rag test set
    parser.add_argument("--verbose", default=False, action="store_true")  # print extra information

    args = parser.parse_args()

    return args

def get_filename(dataset, dataset_path = '', model_name = '',):
    filename_args = "_".join(
        [
            model_name,
            dataset_path.split('/')[-1],
        ]
    )
    filename = f"results/{dataset}/{filename_args}.jsonl"
    print(f"output file: {filename}")
    return filename

def write_results(x, predictions, direction, writer, args, pred):

    entity, relation, targets, time = x[0], x[1], x[2], x[3]
    example = {
        "timestamp": time,
        "entity": entity,
        "relation": relation,
        "targets": targets,
        "direction": direction,
        "predictions": [x[0] for x in predictions],
        "output_text": pred
    }
    writer.write(json.dumps(example) + "\n")

    if args.verbose:
        print(f"example:\n{json.dumps(example, indent=2)}")

    return example


def update_metric(example, metric, args):


    def cosine_similarity(target, prediction):

        vec1 = Counter(target)
        vec2 = Counter(prediction.replace(']','').replace('</s>','').strip()) # Flan-T5-small predictions do contain extra tokens that need to be cleaned.
        dot_product = sum(vec1[ch] * vec2[ch] for ch in vec1)
        magnitude1 = sqrt(sum(count ** 2 for count in vec1.values()))
        magnitude2 = sqrt(sum(count ** 2 for count in vec2.values()))

        return dot_product / (magnitude1 * magnitude2) if magnitude1 != 0 and magnitude2 != 0 else 0

    if args.verbose:
        print(f'predictions: {example["predictions"]}')

    for target in example["targets"]:
        
        metric.total += 1
        metric.total2 += 1

        # the other approach of calculating metrics
        '''for i, pred in enumerate(example['output_text']): 
          if target in pred:
            metric.update2(i+1)'''
        
        # calculate cosine similarity
    
        cs = cosine_similarity(target, example['predictions'][0])
        metric.update3(cs)

        # standard and reported approach for calculating metrics
        # verify if the target is among the predictions, and if yes which is its rank

        index = example["predictions"].index(target) if target in example["predictions"] else -1
        if index >= 0:
            _predictions = [
                x for x in example["predictions"][:index] if x not in example["targets"]
            ]
            rank = len(_predictions) + 1
            if args.verbose:
                print(f"target: {target} --> rank: {rank}")
            metric.update(rank)
