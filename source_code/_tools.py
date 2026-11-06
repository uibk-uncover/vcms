"""Training utilities: running metrics (loss, P_E, ROC AUC), logging, and seeding.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

from abc import ABC, abstractmethod
import logging
import numpy as np
from pathlib import Path
import sys
import torch
from typing import Optional, TextIO, Dict, Tuple
import warnings


class Meter(ABC, torch.nn.Module):
    """Interface for meter classes."""
    name: str

    def __init__(self, fmt: str = ':4.3f', device: torch.device = None):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.fmt = fmt
        self.reset()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, *args, **kw):
        pass

    @property
    @abstractmethod
    def avg(self) -> float:
        pass

    def __str__(self):
        fmtstr = '{name}={avg' + self.fmt + '}'
        return fmtstr.format(name=self.name, avg=self.avg)

    def to_dict(self) -> Dict[str, float]:
        return {self.name: self.avg}

    def to(self, *args, device, **kw):
        super().to(*args, **kw)
        self.device = device


class DetectionMeter(Meter):
    """Abstract class for detection meter.

    Can be used for metrics that need all the labels and predictions.
    Such metrics are typically derived from the ROC curve or confusion matrix.
    """

    def __init__(self, *args, y_true_idx: int = 0, y_pred_idx: int = 1, history: int = None, device: torch.device = None, **kw):
        super().__init__(*args, **kw)
        self.y_true_idx = y_true_idx
        self.y_pred_idx = y_pred_idx
        self.history = history
        self.device = device or torch.device("cpu")
        self.reset()

    def reset(self):
        self.y_pred = torch.empty(0, device=self.device)
        self.y_true = torch.empty(0, device=self.device)

    def update(self, *ys):
        y_true = ys[self.y_true_idx]
        y_pred = ys[self.y_pred_idx]

        # flatten and detach but KEEP ON GPU
        y_true = y_true.detach().flatten().to(self.device)
        y_pred = y_pred.detach().flatten().to(self.device)

        self.y_pred = torch.cat([self.y_pred, y_pred])
        self.y_true = torch.cat([self.y_true, y_true])
        #
        if self.history:
            self.y_pred = self.y_pred[-self.history:]
            self.y_true = self.y_true[-self.history:]

    def to(self, *args, **kw):
        super().to(*args, **kw)
        self.y_pred = self.y_pred.to(self.device)
        self.y_true = self.y_true.to(self.device)
        return self


class ROCMeter(DetectionMeter):
    name = 'roc'
    @staticmethod
    def roc(labels, scores, num_steps=100) -> Tuple[torch.Tensor]:
        """ROC over quantile thresholds."""
        # pre-check class availability
        if (labels == 1).sum() == 0:
            warnings.warn("no positive (stego) samples, TPR undefined")
        if (labels == 0).sum() == 0:
            warnings.warn("no negative (cover) samples, FPR undefined")
        #
        steps = torch.linspace(0.0, 1.0, num_steps + 1, device=labels.device, dtype=scores.dtype)
        taus = torch.quantile(scores, steps)

        # compute tpr, fpr for each tau
        # Broadcast taus to compare: scores[..., None] >= taus
        cmp = scores[:, None] >= taus[None, :]   # shape: [N, num_steps]

        is_pos = (labels == 1)[:, None]
        is_neg = (labels == 0)[:, None]

        tpr = (cmp & is_pos).float().sum(dim=0) / is_pos.float().sum()
        fpr = (cmp & is_neg).float().sum(dim=0) / is_neg.float().sum()

        return fpr, tpr, taus


class PEMeter(ROCMeter):
    """P_E metric, as used in the steganalysis literature."""
    name = 'p_e'
    @property
    def avg(self) -> float:
        labels, scores = self.y_true, self.y_pred
        # NaN guard
        if torch.isnan(labels).any() or torch.isnan(scores).any() or len(labels) == 0 or len(scores) == 0:
            return float('nan')
        #
        fpr, tpr, taus = self.roc(labels=labels, scores=scores)
        # P_E
        p_e = 0.5 * ((1 - tpr) + fpr)
        # best index
        idx_best = torch.argmin(p_e)
        return p_e[idx_best].item()  # , taus[idx_best].item()


class AUCMeter(ROCMeter):
    name = 'roc_auc'
    @property
    def avg(self) -> float:
        labels, scores = self.y_true, self.y_pred
        #
        fpr, tpr, taus = self.roc(labels=labels, scores=scores)
        # trapezoidal integration
        d_fpr = torch.abs(fpr[1:] - fpr[:-1])
        avg_tpr = (tpr[1:] + tpr[:-1]) / 2
        auc = torch.sum(d_fpr * avg_tpr)
        return auc.item()



class PredictionMeter(Meter):
    """Abstract class to calculate the rolling average value.

    Can be used for additive metrics, such as MSE.
    """
    def __init__(self, *args, gamma: float = .9, **kw):
        super().__init__(*args, **kw)
        self.gamma = gamma
        self.reset()

    def reset(self):
        self._mean = None
        self._mean2 = None

    @abstractmethod
    def compute(self, *args, **kw):
        ...

    def update(self, val):
        if self._mean is None:
            self._mean = val
            self._mean2 = val ** 2
        else:
            self._mean = (self.gamma) * self._mean + (1 - self.gamma) * val
            self._mean2 = (self.gamma) * self._mean2 + (1 - self.gamma) * val ** 2

    @property
    def avg(self):
        return self._mean

    @property
    def var(self):
        if self._mean is None:
            return None
        return self._mean2 - (self._mean**2)

    def to(self, *args, **kw):
        super().to(*args, **kw)
        self._mean = self._mean.to(self.device)
        self._mean2 = self._mean2.to(self.device)


class LossMeter(PredictionMeter):
    name = 'loss'
    """Computes and stores the average loss value"""
    def compute(self, *args, **kw) -> float:
        ...  # placeholder





class StdoutLogger:
    """Logger writing the same lines to several streams (e.g., console and log file)."""
    def __init__(
        self,
        files: Tuple[str | Path | TextIO, ...] = (sys.stdout,),
        module: str = 'train_proposed.py',
    ):
        # open paths, keep streams; remember which files we own
        self.files = []
        self._owned_files = []
        for file in files:
            if isinstance(file, (str, Path)):
                file = open(str(file), 'a')
                self._owned_files.append(file)
            self.files.append(file)

        # Configure a logger for pretty output
        self._log = logging.getLogger('StdoutLogger')  # global singleton
        self._log.handlers.clear()  # avoid duplicate lines if instantiated twice
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        for file in self.files:
            handler = logging.StreamHandler(file)  # one handler per stream
            handler.setFormatter(formatter)
            self._log.addHandler(handler)
        self._log.propagate = False
        self._log.setLevel(logging.INFO)
        self.module = module

    def log_scalars(
        self,
        metrics: Dict[str, float],
        split: str,
        epoch: int,
        num_batches: int,
        batch_idx: int = None,
        step: Optional[int] = None,
        fmt: str = '%.04f',
        **kw,
    ):
        # default batch index is the number of batches
        batch_s = f'[{batch_idx:4d}/{num_batches}]' if batch_idx is not None else ''
        step_s = f'[{step:d}]' if step is not None else ''
        # Prefix
        prefix = f"{self.module} - {split}: [{epoch:2d}]{step_s}{batch_s}"

        # Format metrics
        parts = []
        for k, v in metrics.items():
            parts.append(f'{k} {fmt % v}')
        metrics_str = '  '.join(parts)

        msg = f'{prefix}  {metrics_str}'
        self._log.info(msg)

    def debug(self, *args, **kw):
        self._log.debug(*args, **kw)

    def info(self, *args, **kw):
        self._log.info(*args, **kw)

    def warn(self, *args, **kw):
        self._log.warning(*args, **kw)

    def error(self, *args, **kw):
        self._log.error(*args, **kw)

    def close(self):
        """Closes only the files opened by the logger (not, e.g., sys.stdout)."""
        for handler in self._log.handlers:
            handler.flush()
        for file in self._owned_files:
            file.close()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(seed)
    return rng, g