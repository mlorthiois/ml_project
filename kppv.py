import numpy as np
from base import ClassifierWithHp
from utils import euclidian, matrix_dist


class KPPV(ClassifierWithHp):
    def __init__(self, k, distance=euclidian):
        self.k = k
        self.distance = distance

    @property
    def hyperparameters(self):
        return {"k": self.k}

    def train(self, dataset):
        self._x = dataset.x
        self._y = dataset.y

    def predict(self, x):
        labels = np.unique(self._y)
        distances = matrix_dist(x, self._x, distance=self.distance)
        labels_idx = distances.argsort(axis=1)[:, : self.k]
        y_preds = self._y[labels_idx]
        y_preds_prob = np.column_stack([(y_preds == i).sum(axis=-1) / self.k for i in labels])
        return y_preds_prob
