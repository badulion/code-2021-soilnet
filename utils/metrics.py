import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy


class Metric:
    def __init__(self):
        super().__init__()

    def reset(self):
        raise NotImplementedError

    def update(self, pred, target):
        raise NotImplementedError

    def calculate(self):
        raise NotImplementedError


class MSE(Metric):
    def __init__(self):
        super().__init__()

        self.length = 0
        self.sum = 0

    def reset(self):
        self.__init__()

    def update(self, pred, target):
        self.length += target.shape[0]
        self.sum += np.square(pred - target).sum()

    def calculate(self):
        return self.sum / self.length


class MAE(Metric):
    def __init__(self):
        super().__init__()

        self.length = 0
        self.sum = 0

    def reset(self):
        self.__init__()

    def update(self, pred, target):
        self.length += target.shape[0]
        self.sum += np.abs(pred - target).sum()

    def calculate(self):
        return self.sum / self.length


class MAEClass(Metric):
    def __init__(self, index=0):
        super().__init__()

        self.length = 0
        self.sum = 0
        self.index = index

    def reset(self):
        self.__init__()

    def update(self, pred, target):
        self.length += target.shape[0]
        self.sum += np.abs(pred[:, self.index] - target[:, self.index]).sum()

    def calculate(self):
        return self.sum / self.length


class MSEClass(Metric):
    def __init__(self, index=0):
        super().__init__()

        self.length = 0
        self.sum = 0
        self.index = index

    def reset(self):
        self.__init__()

    def update(self, pred, target):
        self.length += target.shape[0]
        self.sum += np.square(pred[:, self.index] - target[:, self.index]).sum()

    def calculate(self):
        return self.sum / self.length


class JSDivergence(Metric):
    """Jensen-Shannon Divergence"""

    def __init__(self):
        super().__init__()

        self.length = 0
        self.sum = 0

    def reset(self):
        self.__init__()

    def update(self, pred, target):
        self.length += target.shape[0]
        self.sum += np.sum([np.square(jensenshannon(p, t)) for p, t in zip(pred, target)])

    def calculate(self):
        return self.sum / self.length


class KLDivergence(Metric):
    """Kullback-Leibler Divergence"""

    def __init__(self):
        super().__init__()

        self.length = 0
        self.sum = 0

    def reset(self):
        self.__init__()

    def update(self, pred, target):
        self.length += target.shape[0]
        self.sum += np.sum(entropy(target, pred, axis=1))

    def calculate(self):
        return self.sum / self.length


class Accuracy(Metric):
    """Accuracy"""

    def __init__(self):
        super().__init__()

        self.total = 0
        self.correct = 0

    def reset(self):
        self.__init__()

    def update(self, pred, target):
        self.total += target.shape[0]
        pred_class = soil_classification(pred)
        target_class = soil_classification(target)

        self.correct += np.sum(target_class == pred_class)

    def calculate(self):
        return self.correct / self.total


class Recall(Metric):
    """Accuracy"""

    def __init__(self):
        super().__init__()

        self.true_positives = 0
        self.false_negatives = 0
        self.total = 0
        self.num_classes = 12

    def reset(self):
        self.__init__()

    def update(self, pred, target):
        pred_class = soil_classification(pred)
        target_class = soil_classification(target)

        pred_onehot = np.eye(self.num_classes)[pred_class]
        target_onehot = np.eye(self.num_classes)[target_class]

        self.true_positives += np.sum(pred_onehot*target_onehot == 1, axis=0)
        self.false_negatives += np.sum((pred_onehot == 0)*(target_onehot == 1), axis=0)
        self.total += np.sum(target_onehot, axis=0)

    def calculate(self):
        denominator = self.true_positives+self.false_negatives
        denominator = np.maximum(denominator, 1e-1)
        return np.mean(self.true_positives / denominator)


class Precision(Metric):
    """Accuracy"""

    def __init__(self):
        super().__init__()

        self.true_positives = 0
        self.false_positives = 0
        self.total = 0
        self.num_classes = 12

    def reset(self):
        self.__init__()

    def update(self, pred, target):
        pred_class = soil_classification(pred)
        target_class = soil_classification(target)

        pred_onehot = np.eye(self.num_classes)[pred_class]
        target_onehot = np.eye(self.num_classes)[target_class]

        self.true_positives += np.sum(pred_onehot*target_onehot == 1, axis=0)
        self.false_positives += np.sum((pred_onehot == 1)*(target_onehot == 0), axis=0)
        self.total += np.sum(target_onehot, axis=0)

    def calculate(self):
        denominator = self.true_positives+self.false_positives
        denominator = np.maximum(denominator, 1e-1)
        return np.mean(self.true_positives / denominator)


class F1(Metric):
    """Accuracy"""

    def __init__(self):
        super().__init__()

        self.true_positives = 0
        self.false_negatives = 0
        self.false_positives = 0
        self.total = 0
        self.num_classes = 12

    def reset(self):
        self.__init__()

    def update(self, pred, target):
        pred_class = soil_classification(pred)
        target_class = soil_classification(target)

        pred_onehot = np.eye(self.num_classes)[pred_class]
        target_onehot = np.eye(self.num_classes)[target_class]

        self.true_positives += np.sum(pred_onehot*target_onehot == 1, axis=0)
        self.false_negatives += np.sum((pred_onehot == 0)*(target_onehot == 1), axis=0)
        self.false_positives += np.sum((pred_onehot == 1)*(target_onehot == 0), axis=0)
        self.total += np.sum(target_onehot, axis=0)

    def calculate(self):
        denominator = self.true_positives+0.5*(self.false_negatives+self.false_positives)
        denominator = np.maximum(denominator, 1e-1)
        return np.mean(self.true_positives / denominator)


class AllMetrics(Metric):
    def __init__(self):
        super().__init__()

        self.metrics = {
            "mae": MAE(),
            "mse": MSE(),
            "mae_sand": MAEClass(index=0),
            "mae_silt": MAEClass(index=1),
            "mae_clay": MAEClass(index=2),
            "mse_sand": MSEClass(index=0),
            "mse_silt": MSEClass(index=1),
            "mse_clay": MSEClass(index=2),
            "js_divergence": JSDivergence(),
            "kl_divergence": KLDivergence(),
            "accuracy": Accuracy(),
            "recall": Recall(),
            "precision": Precision(),
            "f1": F1(),
            }

    def reset(self):
        for _, metric in self.metrics.items():
            metric.reset()

    def update(self, pred, target):
        for _, metric in self.metrics.items():
            metric.update(pred, target)

    def calculate(self, prefix=""):
        output = {}
        for name, metric in self.metrics.items():
            output[prefix + name] = metric.calculate()

        return output

    def calculate_string(self):
        output = ""
        for name, metric in self.metrics.items():
            output += f"{name}: {metric.calculate()}\n"

        return output


def soil_classification(grain_distribution, sand_idx=0, silt_idx=1, clay_idx=2):
    soil_class = np.zeros(len(grain_distribution), dtype=np.int)
    soil_class[(grain_distribution[:, clay_idx] > 0.4) & (grain_distribution[:, sand_idx] <= 0.45)
               & (grain_distribution[:, silt_idx] <= 0.4)] = 0  # clay
    soil_class[(grain_distribution[:, clay_idx] > 0.4) & (grain_distribution[:, silt_idx] > 0.4)] = 1  # silty clay
    soil_class[(grain_distribution[:, clay_idx] > 0.35) & (grain_distribution[:, sand_idx] > 0.45)] = 2  # sandy clay
    soil_class[(grain_distribution[:, clay_idx] > 0.275) & (grain_distribution[:, clay_idx] <= 0.4) & (
        grain_distribution[:, sand_idx] <= 0.45) & (grain_distribution[:, sand_idx] > 0.2)] = 3  # clay loam
    soil_class[(grain_distribution[:, clay_idx] > 0.275) & (grain_distribution[:, clay_idx] <= 0.4)
               & (grain_distribution[:, sand_idx] <= 0.2)] = 4  # silty clay loam
    soil_class[(grain_distribution[:, clay_idx] > 0.2) & (grain_distribution[:, clay_idx] <= 0.35) & (
        grain_distribution[:, sand_idx] > 0.45) & (grain_distribution[:, silt_idx] <= 0.275)] = 5  # sandy clay loam
    soil_class[(grain_distribution[:, clay_idx] > 0.075) & (grain_distribution[:, clay_idx] <= 0.275) & (grain_distribution[:,
                                                                                                                            sand_idx] <= 0.525) & (grain_distribution[:, silt_idx] > 0.275) & (grain_distribution[:, silt_idx] <= 0.5)] = 6  # loam
    soil_class[(grain_distribution[:, clay_idx] <= 0.275) & (grain_distribution[:, silt_idx] > 0.5) & (
        (grain_distribution[:, silt_idx] <= 0.8) | (grain_distribution[:, clay_idx] > 0.125))] = 7  # silt loam
    soil_class[(grain_distribution[:, silt_idx] > 0.8) & (grain_distribution[:, clay_idx] <= 0.125)] = 8  # silt
    soil_class[(grain_distribution[:, sand_idx]-grain_distribution[:, clay_idx] <= 0.7) & (grain_distribution[:, clay_idx] <= 0.2)
               & (grain_distribution[:, silt_idx] <= 0.5) & ((grain_distribution[:, sand_idx] > 0.525) | (grain_distribution[:, clay_idx] <= 0.075))] = 9  # sandy loam
    soil_class[(grain_distribution[:, sand_idx]-0.5*grain_distribution[:, clay_idx] <= 0.85)
               & (grain_distribution[:, sand_idx]-grain_distribution[:, clay_idx] > 0.7)] = 10  # loamy sand
    soil_class[(grain_distribution[:, sand_idx]-0.5*grain_distribution[:, clay_idx] > 0.85)] = 11  # sand

    return soil_class
