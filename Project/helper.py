import numpy as np


# ----------------------------------------------------------------------

# methods for ML
def estimate_mean_ml(points, n):
    points = np.array(points)
    points = points[:, :n]
    mean = np.sum(points, axis=1)
    mean = mean / n
    mean = np.array(mean)[np.newaxis]
    return mean.transpose()


def estimate_cov_ml(points, mean, n):
    points = np.array(points)
    mean = np.array(mean)
    cov = (points - mean) @ (points - mean).transpose()
    cov = cov / n
    return cov


# ----------------------------------------------------------------------

def estimate_mean_bl(points, mean0, cov_initial, cov_actual, n):
    points = np.array(points)
    points = points[:, :n]
    mean0 = np.array(mean0)
    cov_initial = np.array(cov_initial)
    cov_actual = np.array(cov_actual)
    points_sum = np.sum(points, axis=1) / n
    points_sum = np.array(points_sum)[np.newaxis]
    points_sum = points_sum.transpose()

    m = cov_actual / n @ np.linalg.inv(
        cov_actual / n + cov_initial) @ mean0 + cov_initial @ np.linalg.inv(
        cov_actual / n + cov_initial) @ points_sum
    return m
