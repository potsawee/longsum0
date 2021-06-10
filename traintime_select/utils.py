import random
import numpy as np
from rouge import Rouge
rouge_pltrdy = Rouge()

def get_rouge2recall_scores(sentences, reference, oracle_type):
    if oracle_type not in ['padrand', 'padlead']:
        raise Exception("oracle_type must be padrand or padlead")

    # rouge_pltrdy is case sensitive
    reference = reference.lower()
    scores = [None for _ in range(len(sentences))]
    count_nonzero_rouge2recall = 0
    for i, sent in enumerate(sentences):
        sent = sent.lower()
        try:
            rouge_scores = rouge_pltrdy.get_scores(sent, reference)
            rouge2recall = rouge_scores[0]['rouge-2']['r']
            scores[i] = rouge2recall
        except ValueError:
            scores[i] = 0.0
        except RecursionError:
            scores[i] = 0.5 # just assign 0.5 as this sentence is simply too long
        if scores[i] > 0.0: count_nonzero_rouge2recall += 1
    scores = np.array(scores)
    N = len(scores)

    if oracle_type == 'padlead':
        biases = np.array([(N-i)*1e-12 for i in range(N)])
    elif oracle_type == 'padrand':
        biases = np.random.normal(scale=1e-10,size=(N,))
    else:
        raise ValueError("this oracle method not supported")
    return scores + biases

def get_rouge2recall_scores_nopad(sentences, reference):
    # rouge_pltrdy is case sensitive
    reference = reference.lower()
    scores = [None for _ in range(len(sentences))]
    count_nonzero_rouge2recall = 0
    for i, sent in enumerate(sentences):
        sent = sent.lower()
        try:
            rouge_scores = rouge_pltrdy.get_scores(sent, reference)
            rouge2recall = rouge_scores[0]['rouge-2']['r']
            scores[i] = rouge2recall
        except ValueError:
            scores[i] = 0.0
        except RecursionError:
            scores[i] = 0.5 # just assign 0.5 as this sentence is simply too long
        if scores[i] > 0.0: count_nonzero_rouge2recall += 1
    scores = np.array(scores)
    return scores
