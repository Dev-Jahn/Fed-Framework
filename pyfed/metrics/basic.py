from functools import cache, _lru_cache_wrapper
import warnings

import numpy as np
import torch
import sklearn
from sklearn.metrics._classification import UndefinedMetricWarning
import plotly.graph_objects as go


# TODO: Add multi-label classification support
class ClassificationMetrics:
    """
    Continuously stores and computes metrics for classification task
    Efficient implementation for n>>d case, when n is n_samples and d is n_classes
    """

    def __init__(self, n_classes, pos_label=1, average='binary', warn=False):
        self.warn=warn
        if n_classes > 2:
            assert average in [None, 'micro', 'macro', 'weighted']
        self.n_classes, self.pos_label, self.average = n_classes, pos_label, average
        self.reset()

    def clear_caches(self):
        props = list(map(lambda x: x.fget,
                         filter(lambda attr: isinstance(attr, property), ClassificationMetrics.__dict__.values())))
        attrs = list(ClassificationMetrics.__dict__.values())
        for wrapper in filter(lambda attr: isinstance(attr, _lru_cache_wrapper), attrs + props):
            wrapper.cache_clear()

    def reset(self):
        self.confusion = np.zeros((self.n_classes, self.n_classes))
        self.clear_caches()

    def update(self, true, pred):
        self.confusion += sklearn.metrics.confusion_matrix(true, pred, labels=range(self.n_classes))
        self.clear_caches()

    @cache
    def accuracy(self):
        return self.confusion.trace() / self.confusion.sum()

    @property
    @cache
    def _pred(self):
        return self.confusion.sum(axis=0)

    @property
    @cache
    def _true(self):
        # can be viewed as 'support'
        return self.confusion.sum(axis=1)

    @property
    @cache
    def _tp(self):
        return np.diag(self.confusion)

    @property
    @cache
    def _fp(self):
        return self._pred - self._tp

    @property
    @cache
    def _fn(self):
        return self._true - self._tp

    def _metrics_impl(self, op):
        assert op in ['prec', 'recl', 'f1']
        if op == 'prec':
            denom = self._pred
        if op == 'recl':
            denom = self._true
        if op == 'f1':
            denom = (self._pred + self._true) / 2
        if np.any(denom == 0):
            if self.warn: warnings.warn('Precision, recall and F-1 score when denominator is zero is not defined.', UndefinedMetricWarning)
            denom = denom+1e-15

        raw = self._tp / denom
        if self.average is None:
            return raw
        elif self.average == 'binary':
            return raw[self.pos_label]
        elif self.average == 'micro':
            return self.accuracy()  # may differ when multi-label
        elif self.average == 'macro':
            return raw.mean()
        elif self.average == 'weighted':
            return np.average(raw, weights=self._true)

    @cache
    def precision(self):
        return self._metrics_impl('prec')

    @cache
    def recall(self):
        return self._metrics_impl('recl')

    @cache
    def f1(self):
        return self._metrics_impl('f1')

    def summary(self, prefix=None):
        prefix = '' if prefix is None else prefix + ' '
        return {
            prefix + 'Accuracy': self.accuracy(),
            prefix + 'Precision': self.precision(),
            prefix + 'Recall': self.recall(),
            prefix + 'F-1 Score': self.f1()
        }

    @staticmethod
    def _test():
        true, pred = np.random.randint(0, 10, (10000)), np.random.randint(0, 10, (10000))
        metrics = ClassificationMetrics(10, average=None)
        metrics.update(true, pred)
        return np.allclose(sklearn.metrics.accuracy_score(true, pred), metrics.accuracy()) and \
            np.allclose(sklearn.metrics.precision_score(true, pred, average=None), metrics.precision()) and \
            np.allclose(sklearn.metrics.recall_score(true, pred, average=None), metrics.recall()) and \
            np.allclose(sklearn.metrics.f1_score(true, pred, average=None), metrics.f1())


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_accuracy(model, dataloader, get_confusion_matrix=False, device='cuda'):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.to(device), target.to(device, dtype=torch.int64)
            out = model(x)
            _, pred_label = torch.max(out.data, 1)

            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

            if device == "cpu":
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())
            else:
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if get_confusion_matrix:
        conf_matrix = sklearn.metrics.confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct / float(total), conf_matrix

    return correct / float(total)
