import argparse
from evaluate import load
import numpy as np
from datasets import load_dataset

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", "-rf", type=str)

    args = parser.parse_args()
    return args
    
if __name__ == "__main__":

    print('If any errors regarding the loading of the dataset appears, please run "pip install -U datasets"')

    args = parser()

    bertscore = load("bertscore")
    dataset = load_dataset('json', data_files=f'./results/{args.results_file}', split = 'train')
    targets = [x[0] for x in dataset[:]['targets']]
    predictions = [x[0] for x in dataset[:]['predictions']]
    
    bertscore = load("bertscore")
    results = bertscore.compute(predictions=predictions, references=targets, lang="en")
        
    bs = np.mean(results['f1'])  

    print(f'For {args.results_file}, BERTScore is: {bs}')