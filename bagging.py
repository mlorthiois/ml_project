import numpy as np
from base import ClassifierWithHp
import math


class Bagging(ClassifierWithHp):
    def __init__(self, base, n: int, max_percent: float = 0.3, *args, **kwargs):
        self.base = base
        self.n = n
        self.percent = max_percent
        self.args = args
        self.kwargs = kwargs
        self.models = []

    @property
    def hyperparameters(self):
        return {"model": self.models[0].__class__.__name__, "n": self.n} + self.models[
            0
        ].hyperparameters

    def train(self, dataset):
        for _ in range(self.n):
            num_true = math.ceil(len(dataset) * self.percent)
            num_false = len(dataset) - num_true
            mask = np.array([True] * num_true + [False] * num_false)
            np.random.shuffle(mask)
            filtered_d = dataset[mask]

            model = self.base(*self.args, **self.kwargs)
            model.train(filtered_d)
            self.models.append(model)

    def predict(self, x):
        predictions = []
        for model in self.models:
            prediction = model.predict(x)
            predictions.append(prediction)

        predictions = np.array(predictions)
        predictions = np.stack(predictions, axis=2)
        return predictions.mean(2)
