from .syntaxgym.agg_surprisals import *
from .syntaxgym import _load_suite
import numpy as np


# parser = argparse.ArgumentParser(description='aggregate surprisal')
# parser.add_argument('--surprisal', type=Path,
#                     help='path to file containing token-based surprisals')
# parser.add_argument('--sentences', type=Path,
#                     help='path to file containing pre-tokenized sentences')
# parser.add_argument('--spec', type=Path, help='Model specification file')
# parser.add_argument('--input', type=Path,
#                     help='path to JSON file with input data')
# parser.add_argument('--output', '-o', type=Path,
#                     help='path to JSON file to write output data')
# args = parser.parse_args()


def get_accuracy(input_path, surprisal_path, sentence_path, spec_path):
    # read input test suite and token-level surprisals
    in_data = _load_suite(input_path)
    surprisals = pd.read_csv(surprisal_path, delim_whitespace=True)

    # obtain spec for model
    spec = utils.load_json(spec_path)

    # obtain tokens and unk mask for sentences
    tokens = utils.tokenize_file(sentence_path, None)
    unks = utils.unkify_file(sentence_path, None)

    # aggregate token-level --> region-level surprisals
    suite = aggregate_surprisals(surprisals, tokens, unks, in_data, spec)

    results = suite.evaluate_predictions()

    # print(results)

    results_data = [(suite.meta["name"], pred.idx, item_number, result)
                    for item_number, preds in results.items()
                    for pred, result in preds.items()]

    # for rs_item in results_data:
    #     print('\t'.join([str(x) for x in rs_item]), file=sys.stderr)

    results_summary  = []

    for item_number, preds in results.items():
        item_result = True
        for pred, result in preds.items():
            if pred.idx == 0:
                item_result = result
                break
        results_summary.append([item_number, item_result])
        
    # print(suite.meta["name"], np.mean([x[-1] for x in results_summary]))
    return suite.meta["name"], np.mean([x[-1] for x in results_summary])


# get_accuracy(args.input, args.surprisal, args.sentences, args.spec)