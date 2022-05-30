import numpy as np


def confusion_matrix(y, y_pred):
    n_labels = np.unique(y).shape[0]
    m = np.zeros(shape=(n_labels, n_labels), dtype="int")
    for y_true, y_hat in zip(y, y_pred):
        m[y_true, y_hat] += 1
    return m


def accuracy(conf_m):
    return np.diag(conf_m).sum() / conf_m.sum()


def mahalanobis(u, v, cov, iv_cov, prob):
    delta = u - v
    bracket = np.log(np.linalg.det(cov)) - 2 * np.log(prob)
    return delta.T @ iv_cov @ delta + bracket


def euclidian(u, v):
    delta = u - v
    return delta.T @ delta


def matrix_dist(u, v, distance):
    mU = u.shape[0]
    mV = v.shape[0]
    dm = np.empty((mU, mV), dtype=np.double)
    for i in range(mU):
        for j in range(mV):
            dm[i, j] = distance(u[i], v[j])
    return dm


def softmax(x: np.array) -> np.array:
    row_max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - row_max)
    row_sum = np.sum(e_x, axis=-1, keepdims=True)
    return e_x / row_sum


def softmin(x: np.array) -> np.array:
    exp_row = np.exp(-x)
    return exp_row / np.sum(exp_row, axis=-1, keepdims=True)


assert euclidian(np.array([2, -1]), np.array([-2, 2])) == 25.0
assert euclidian(np.array([0, 0]), np.array([2, 0])) == 4.0
