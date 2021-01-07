from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from datasets import load_metric

def compute_metric(arg, target_sentences, generated_sentences):
    avg_score = None
    if (arg.metric == 'BLEU'):
        avg_score = compute_bleu_scores(target_sentences, generated_sentences)
    elif (arg.metric == 'ROUGE'):
        avg_score = compute_rogue_scores(target_sentences, generated_sentences)
    else:
        assert False, f'{arg.metric} not defined'
    return avg_score

def compute_bleu_scores(target_sentences, generated_sentences):
    bleu_scores = [sentence_bleu([target_sentences[i].split()], generated_sentences[i].split()) for i, sen in enumerate(generated_sentences)]
    return np.mean(bleu_scores)

def compute_rogue_scores(target_sentences, generated_sentences):
    metric = load_metric('rouge')
    metric.add_batch(predictions=generated_sentences, references=target_sentences)
    score = metric.compute()
    rougeL_f = score['rougeL'].mid.fmeasure
    return rougeL_f