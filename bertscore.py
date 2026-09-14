import argparse
from evaluate import load
import numpy as np
from datasets import load_dataset

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", "-rf", type=str)
    parser.add_argument(
        "--no_rescale",
        dest="rescale",
        action="store_false",
        help=(
            "Report raw BERTScore instead of rescaling against the baseline. Raw values "
            "sit in a narrow high band and are not comparable to published numbers."
        ),
    )

    args = parser.parse_args()
    return args
    
if __name__ == "__main__":

    print('If any errors regarding the loading of the dataset appears, please run "pip install -U datasets"')

    args = parser()

    # force_redownload re-reads the results file instead of trusting the
    # loader's cache; results are rewritten in place by each evaluation run.
    dataset = load_dataset('json', data_files=f'./results/{args.results_file}', split = 'train',
                           download_mode='force_redownload')
    # Only the top-1 prediction is scored. The candidate list beam search
    # produces could be scored in full, but a top-k BERTScore is not reported.
    targets = [x[0] for x in dataset[:]['targets']]
    predictions = [x[0] for x in dataset[:]['predictions']]

    bertscore = load("bertscore")
    # rescale_with_baseline subtracts the score of randomly paired text, so 0
    # means "no better than random". Raw values sit near a high floor and are
    # not comparable across papers. Downloads roberta-large on first use.
    results = bertscore.compute(
        predictions=predictions,
        references=targets,
        lang="en",
        rescale_with_baseline=args.rescale,
    )

    bs = np.mean(results['f1'])
    scale = "rescaled" if args.rescale else "raw"

    print(f'For {args.results_file}, BERTScore F1 ({scale}, top-1, '
          f'n={len(predictions)}): {bs}')