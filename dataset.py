import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, x, y, correction=None, name=""):
        self.x = x
        if min(y) == 1:
            y -= 1
        self.y = y
        self.name = name
        if correction is not None:
            self._correct_labels(correction)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return Dataset(self.x[item], self.y[item], name=self.name)

    @property
    def labels(self):
        return np.unique(self.y)

    @classmethod
    def _load_file(cls, filename, **kwargs):
        data = np.genfromtxt(filename, delimiter=" ")
        x = data[:, 1:]
        y = data[:, 0].astype("int")
        return cls(x, y, **kwargs)

    @classmethod
    def from_file(cls, filename, **kwargs):
        if isinstance(filename, list) or isinstance(filename, tuple):
            datasets = []
            for f in filename:
                name = f.split("_")[1]
                datasets.append(cls._load_file(f, name=name, **kwargs))
            return datasets
        name = filename.split("_")[1]
        return cls._load_file(filename, name=name, **kwargs)

    def plot(self, **kwargs):
        if not "ax" in kwargs:
            _, ax = plt.subplot()
        else:
            ax = kwargs["ax"]

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

        return sns.scatterplot(
            x=self.x[:, 0],
            y=self.x[:, 1],
            hue=self.y,
            palette="Spectral",
            s=100,
            linewidths=2,
            edgecolor="k",
            antialiased=True,
            **kwargs
        )

    def _correct_labels(self, correction):
        masks = [self.y == i for i in range(len(correction))]
        for mask, corr in zip(masks, correction):
            self.y[mask] = corr
        sort_mask = self.y.argsort()
        self.x = self.x[sort_mask]
        self.y = self.y[sort_mask]
