import argparse
import os

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default = 'icews14',type = str)
    parser.add_argument("--file_name", default = 'all', type = str)

    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    
    args = parser()
    
    path = f"./results/{args.dataset}"
    dir_list = os.listdir(path)

    

    if args.file_name != 'all':
      results_files = [args.file_name]
    else:
      results_files = [f for f in dir_list if f.endswith('.jsonl')]

    h1 = 0
    h3 = 0
    count = 0

    limit = 10000 if args.dataset == 'icews18' else 7371

    for f in results_files:
        
        with open(f'{path}/{f}', 'r', encoding='utf-8') as results:
        
            lines = results.readlines()
            
            for line in lines[:limit]:
                d = eval(line)
    
                pred = d['predictions']

                index = pred.index(d['targets'][0]) if d['targets'][0] in pred else -1
                rank = 5
                if index >= 0:
                    _predictions = [
                        x for x in pred[:index] if x not in d['targets']
                    ]
                    rank = len(_predictions) + 1

                if rank == 1:
                    h1 += 1
                    h3 += 1

                elif rank <=3:
                    h3 += 1


                count += 1

        print(f"For {f.replace('.jsonl','')}, H@1: {round(h1/count,3)}, H@3: {round(h3/count,3)}")
        count = 0
        h1 = 0
        h3 = 0
