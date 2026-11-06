"""Recalculates the ROC AUC of the detector for each prediction file in this folder.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

from glob import glob
import pandas as pd
import torch
import warnings


def calculate_rocauc(y, scores, num_steps: int = 100):
    """ROC AUC over quantile thresholds."""
    # pre-check class availability
    if (y == 1).sum() == 0:
        warnings.warn("no positive (stego) samples, TPR undefined")
    if (y == 0).sum() == 0:
        warnings.warn("no negative (cover) samples, FPR undefined")
    #
    steps = torch.linspace(0.0, 1.0, num_steps + 1, device=y.device, dtype=scores.dtype)
    taus = torch.quantile(scores, steps)

    # compute tpr, fpr for each tau
    cmp = scores[:, None] >= taus[None, :]

    is_pos = (y == 1)[:, None]
    is_neg = (y == 0)[:, None]

    tpr = (cmp & is_pos).float().sum(dim=0) / is_pos.float().sum()
    fpr = (cmp & is_neg).float().sum(dim=0) / is_neg.float().sum()

    # trapezoidal integration
    d_fpr = torch.abs(fpr[1:] - fpr[:-1])
    avg_tpr = (tpr[1:] + tpr[:-1]) / 2
    auc = torch.sum(d_fpr * avg_tpr)
    return auc.item()


# iterate model predictions
for model_file in sorted(glob('*.csv')):
    predictions = pd.read_csv(model_file)
    #
    y = torch.from_numpy(predictions['y'].to_numpy(copy=True))
    logits = torch.from_numpy(predictions[['logit0', 'logit1']].to_numpy(copy=True))
    scores = torch.nn.functional.softmax(logits, dim=1)[:, 1]
    #
    auc = calculate_rocauc(y=y, scores=scores)
    print(model_file, auc)