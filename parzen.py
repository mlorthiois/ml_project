from base import ClassifierWithHp
import numpy as np


class Parzen(ClassifierWithHp):
    def __init__(self, h, kernel) -> None:
        self.h = h
        if not kernel in ("uniform", "gaussian"):
            raise Exception(f"Kernel {kernel} not implemented")
        self.kernel = kernel
        self.trained = False

    @property
    def hyperparameters(self):
        return {"h": self.h, "kernel": self.kernel}

    def train(self, dataset) -> None:
        self._x = dataset.x
        self._y = dataset.y

    def predict(self, x) -> np.array:
        probs = []
        for sample in x:
            probs.append(self._compute_prob(sample))
        return np.array(probs)

    def __uniform(self, dist) -> np.array:
        return np.where(dist < self.h, 1, 0)

    def __gaussian(self, dist) -> np.array:
        # https://sebastianraschka.com/Articles/2014_kernel_density_est.html#5-replacing-the-hypercube-with-a-gaussian-kernel
        return (1 / (np.sqrt(2 * np.pi) * self.h)) * np.exp(-1 / 2 * (dist / self.h) ** 2)

    def _compute_prob(self, point):
        prob = []
        for i in np.unique(self._y):
            small_x = self._x[self._y == i]
            dist = np.sqrt(((point - small_x) ** 2).sum(-1))
            phy = self.__uniform(dist) if self.kernel == "uniform" else self.__gaussian(dist)
            prob.append(phy.sum() / len(small_x))
        return prob
