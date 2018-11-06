from sklearn.metrics.scorer import make_scorer
import torch
import torch.nn
from sklearn.metrics import roc_auc_score


class EvaluationMetrics(object):
    class ImageClassification(object):
        """
        For image classification tasks.
        """
        @staticmethod
        def convert_sklearn_metric_function(scoring):
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

        @staticmethod
        def image_classification_f1(output, target):
            return None

        @staticmethod
        def image_classification_precision(output, target):
            with torch.no_grad():
                tp = (output == target).sum().item()
                fp = (output != target).sum().item()
                recall = tp / tp + fp
            return recall

        @staticmethod
        def image_classification_recall(output, target):
            with torch.no_grad():
                tp = (output == target).sum().item()
                fp = (output != target).sum().item()
                recall = tp / tp + fp
            return recall

        @staticmethod
        def image_classification_auc(output, target):
            output = output.data.numpy()
            target = target.data.numpy()
            with torch.no_grad():
                auc = roc_auc_score(target, output)
            return auc

        @staticmethod
        def image_classification_falsepostive(output, target):
            with torch.no_grad():
                incorrect = (output != target).sum().item()
            return incorrect

        def accuracy(output, target):
            total = 0
            correct = 0
            with torch.no_grad():
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            percent_acc = 100 * correct / total
            return percent_acc

        @staticmethod
        def image_classification_accuracy(output, target):
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