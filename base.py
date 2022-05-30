from abc import ABC, abstractmethod
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from dataset import Dataset
from utils import confusion_matrix, accuracy


class Classifier(ABC):
    @abstractmethod
    def train():
        ...

    @abstractmethod
    def predict(self, x) -> np.array:
        ...

    def evaluate(self, dataset, axs=None, *args, **kwargs):
        x, y = dataset.x, dataset.y
        y_pred = self.predict(x, *args, **kwargs)
        y_sort = y_pred.argsort(-1)
        conf_matrix = confusion_matrix(y, y_sort[:, -1])
        top_1_acc = accuracy(conf_matrix)
        top_2_acc = top_1_acc + accuracy(confusion_matrix(y, y_sort[:, -2]))
        m = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)

        if axs is None:
            _, axs = plt.subplots(nrows=2, ncols=1, figsize=(20, 12))
        m.plot(cmap="Blues", colorbar=False, ax=axs[0])
        self.decision_boundary_plot(dataset, ax=axs[1])
        return {
            "name": self.__class__.__name__,
            "hyperparameters": ", ".join([f"{k}={v}" for k, v in self.hyperparameters.items()]),
            "dataset": dataset.name,
            "top1_acc": top_1_acc,
            "top2_acc": top_2_acc,
        }

    def decision_boundary_plot(self, dataset, ax=None):
        x, _ = dataset.x, dataset.y
        min1, max1 = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
        min2, max2 = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(min1, max1, 0.15), np.arange(min2, max2, 0.15))

        yhat = self.predict(np.c_[xx.ravel(), yy.ravel()]).argmax(-1)
        zz = yhat.reshape(yy.shape)

        if ax is None:
            _, ax = plt.subplots()
        ax.contourf(xx, yy, zz, alpha=0.6, cmap="Spectral")
        dataset.plot(ax=ax)


class ClassifierWithHp(Classifier):
    hyperparameters = {}

    def _get_hp_grid(**kwargs):
        keys = kwargs.keys()
        vals = kwargs.values()
        for instance in itertools.product(*vals):
            yield dict(zip(keys, instance))

    @classmethod
    def hyperparameter_search(cls, dataset, cv, *args, **kwargs):
        x, y = dataset.x, dataset.y
        acc_scores = []
        for hp in cls._get_hp_grid(**kwargs):
            accuracies = []

            for train, test in cv.split(x, y=y):
                X_train, X_test, y_train, y_test = x[train], x[test], y[train], y[test]
                model = cls(*args, **hp)
                model.train(Dataset(X_train, y_train))
                y_pred = model.predict(X_test).argmax(-1)
                conf_matrix = confusion_matrix(y_test, y_pred)
                accuracies.append(accuracy(conf_matrix))

            acc_scores.append({**hp, "accuracy": np.mean(accuracies)})
        df = pd.DataFrame(acc_scores)

        if len(kwargs) == 1:
            lin = list(kwargs.keys())[0]
            sns.lineplot(data=df, x=lin, y="accuracy")
        elif len(kwargs) == 2:
            keys = list(kwargs.keys())
            if isinstance(kwargs[keys[0]][0], str):
                lin = keys[1]
                hue = keys[0]
            else:
                lin = keys[0]
                hue = keys[1]
            sns.lineplot(data=df, x=lin, y="accuracy", hue=hue)
