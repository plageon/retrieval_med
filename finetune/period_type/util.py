from transformers import EvalPrediction
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def compute_metrics(p: EvalPrediction):
    labels, preds = {}, {}
    labels['T_period'], labels['N_period'], labels['cancer_type'] = p.label_ids
    preds['T_period'], preds['N_period'], preds['cancer_type'] = p.predictions

    res = {'acc': 0.0, 'f1': 0.0}
    for field in ['T_period', 'N_period', 'cancer_type']:
        preds[field] = np.argmax(preds[field], axis=1)
        res['{}_f1'.format(field)] = f1_score(y_true=labels[field], y_pred=preds[field], average='weighted')
        res['{}_acc'.format(field)] = accuracy_score(y_true=labels[field], y_pred=preds[field])

        res['acc'] += res['{}_acc'.format(field)]
        res['f1'] += res['{}_f1'.format(field)]
    res['acc'] /= 3
    res['f1'] /= 3

    return res
