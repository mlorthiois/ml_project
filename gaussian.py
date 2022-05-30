from base import ClassifierWithHp
from utils import softmin, euclidian, mahalanobis
import numpy as np


class GaussianEstimator(ClassifierWithHp):
    def __init__(self, distance="euclidian"):
        self.trained = False
        self.distance = distance

    @property
    def hyperparameters(self):
        return {"distance": self.distance}

    def train(self, dataset):
        labels = dataset.labels
        self.dataset_name = dataset.name
        x, y = dataset.x, dataset.y
        self.probs = np.array([(y == label).sum() / len(y) for label in labels])
        self.means = np.array([x[y == label].T.mean(axis=1) for label in labels])
        if self.distance == "mahalanobis":
            self.covs = np.array([np.cov(x[y == label].T) for label in labels])
            self.iv_covs = np.linalg.inv(self.covs)
        self.trained = True

    def predict(self, x):
        if not self.trained:
            raise Exception("Classifier should be trained")
        n_labels = len(self.means)
        predictions = []
        for sample in x:
            if self.distance == "euclidian":
                distances = np.array([euclidian(sample, self.means[i]) for i in range(n_labels)])
            elif self.distance == "mahalanobis":
                distances = np.array(
                    [
                        mahalanobis(sample, self.means[i], self.covs[i], self.iv_covs[i], self.probs[i])
                        for i in range(n_labels)
                    ]
                )
            else:
                raise Exception("distance not recognized")
            predictions.append(softmin(distances))
        return np.array(predictions)
