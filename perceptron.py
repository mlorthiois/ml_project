from base import ClassifierWithHp
import itertools
import numpy as np


class Perceptron(ClassifierWithHp):
    def __init__(self, vs="one", class_to_exclude=4, epochs=100, transform=None):
        self.vs = vs
        self.excl_class = class_to_exclude
        self.epochs = epochs
        self.trained = False
        self.transform = transform

    @property
    def hyperparameters(self):
        return {"versus": self.vs}

    def train(self, dataset):
        x, y = dataset.x, dataset.y

        if self.transform is not None:
            x = self.transform(x)

        if self.vs == "all":
            x = x[y != self.excl_class]
            y = y[y != self.excl_class]

        labels = np.unique(y)
        a_all = []
        couples = []
        if self.vs == "one":
            for fr, to in itertools.combinations(labels, 2):
                mask = (y == fr) | (y == to)
                a_all.append(self._find_hyperplan(x[mask], y[mask], fr))
                couples.append((fr, to))
        else:
            for label in labels:
                a_all.append(self._find_hyperplan(x, y, label))
                couples.append((label, np.nan))

        self.a_all = np.array(a_all)
        self.couples = couples
        self.trained = True

    def predict(self, x):
        if not self.trained:
            raise Exception("classifier should be trained")

        predictions = []
        classes = set([label for couple in self.couples for label in couple if label is not np.nan])

        if self.transform is not None:
            x = self.transform(x)

        for sample in x:
            prediction = []
            for a, couple in zip(self.a_all, self.couples):
                prediction.append(self._compute(sample, a, couple[0], couple[1]))
            prob = [prediction.count(i) / len(prediction) for i in classes]
            predictions.append(prob)
        return np.array(predictions)

    def _epoch(self, x, a):
        num_err = 0
        for sample in x:
            if a.T @ sample <= 0:
                a += sample
                num_err += 1
        return a, num_err

    def _transform_input(self, x, y, fr):
        x_transformed = []
        for _, (sample, label) in enumerate(zip(x, y)):
            if label == fr:
                x_transformed.append(np.append(sample, 1))
            else:
                x_transformed.append(np.append(-sample, -1))
        return np.array(x_transformed)

    def _find_hyperplan(self, x, y, fr):
        x_transformed = self._transform_input(x, y, fr)
        a = np.random.uniform(x.min(), x.max(), x_transformed.shape[1])
        err = len(x_transformed)
        best_a = a
        for _ in range(self.epochs):
            a, new_err = self._epoch(x_transformed, a)
            if new_err == 0:
                return a
            if new_err < err:
                err = new_err
                best_a = a
        return best_a

    def _compute(self, x, a, fr, to):
        if a[:-1].T @ x + a[-1] > 0:
            return fr
        return to
