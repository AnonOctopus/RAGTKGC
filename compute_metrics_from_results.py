import os



path = "./results/icews14"
dir_list = os.listdir(path)

results_files = [f for f in dir_list if f.endswith('.jsonl')]

acc1l = 0
acc3l = 0
acc1s = 0
acc3s = 0
count = 0
for f in results_files:
    with open(f'./results/icews14/{f}', 'r', encoding='utf-8') as results:
        lines = results.readlines()
        for line in lines[:10000]:
            isC = False
            d = eval(line)
            if d['output_text']:
              pred = [x.replace('</s>','').replace(']','').strip() for x in d['output_text']]
            else:
              pred = d['predictions']

            index = pred.index(d['targets'][0]) if d['targets'][0] in pred else -1
            rank = 5
            if index >= 0:
              _predictions = [
                  x for x in pred[:index] if x not in d['targets']
              ]
              rank = len(_predictions) + 1

            if rank == 1:
              acc1s += 1
              acc3s += 1

            elif rank <=3:
              acc3s += 1


            for i,p in enumerate(pred):
              if d['targets'][0] in p and isC == False:

                  isC = True

                  if i == 0:
                    acc1l += 1
                    acc3l += 1
                  elif i <= 2:
                    acc3l += 1



            count += 1
    print(count,round(acc1l/count,3),round(acc3l/count,3),round(acc1s/count,3),round(acc3s/count,3))
    count = 0
    acc1l = 0
    acc3l = 0
    acc1s = 0
    acc3s = 0
#print(len(metrics))
