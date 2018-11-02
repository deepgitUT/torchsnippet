from sklearn.metrics.scorer import make_scorer
import torch
import torch.nn


class EvaluationMetrics(object):
    """
    Evaluation metrics for model.
    """
    def convert_sklearn_metric_function(self, scoring):
        if callable(scoring):
            module = getattr(scoring, '__module__', None)
            if (
                    hasattr(module, 'startswith') and
                    module.startwith('sklearn.metrics.') and
                    not module.startwith('sklearn.metrics.scorer') and
                    not module.startwith('sklearn.metrics.tests')
            ):
                return make_scorer(scoring)
        return scoring

    def image_classification_accuracy(self, output, target):
        """
        Image classification accuracy calculator.
        Args:
            output(Tensor): shape [batch, ], 1 or 0 for every batch sample.
            target(Tensor): save as output.
        """
        output = output.long()
        target = target.long()
        with torch.no_grad():
            total = target.size(0)
            correct = (output == target).sum().item()
        percent_acc = 100 * correct / total
        return percent_acc