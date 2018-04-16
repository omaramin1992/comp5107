import numpy as np
import math


def kernel_function(x, xi, cov):
    result = (1 / (math.sqrt(2 * math.pi) * cov)) * math.exp(-math.pow(x - xi, 2) / (2 * math.pow(cov, 2)))
    return result


# ----------------------------------------------------------------------

def parzen_expected_mean(x, f_x, delta_x):
    return x * f_x * delta_x


# ----------------------------------------------------------------------

def parzen_expected_covariance(x, f_x, delta_x, mean):
    return math.pow(x - mean, 2) * f_x * delta_x


# ----------------------------------------------------------------------

def estimated_mean_parzen(x1_training_points, x2_training_points, dimensions, kernel_cov, step_size):
    x1_parzen_est_mean = []
    x1_parzen_est_cov = []

    x2_parzen_est_mean = []
    x2_parzen_est_cov = []

    for i in range(0, dimensions, 1):
        # for class 1
        f_x1_points = []
        f_x1_values = []
        for j in np.arange(min(x1_training_points[i, :]) - 1, max(x1_training_points[i, :]), step_size):
            f_x1_points = np.append(f_x1_points, j)

        f_x1_points = np.sort(f_x1_points)

        for x in f_x1_points:
            f_x = 0.0
            for xi in x1_training_points[i, :]:
                f_x = f_x + kernel_function(x, xi, kernel_cov)
            f_x = f_x / x1_training_points[i, :].size
            f_x1_values = np.append(f_x1_values, f_x)

        est_mean = 0.0
        for x in range(0, f_x1_points.size):
            est_mean = est_mean + parzen_expected_mean(f_x1_points[x], f_x1_values[x], step_size)
        x1_parzen_est_mean = np.append(x1_parzen_est_mean, est_mean)

        est_cov = 0.0
        for x in range(0, f_x1_points.size):
            est_cov = est_cov + parzen_expected_covariance(f_x1_points[x], f_x1_values[x], step_size, est_mean)
        x1_parzen_est_cov = np.append(x1_parzen_est_cov, est_cov)

        # for class 2
        f_x2_points = []
        f_x2_values = []
        for j in np.arange(min(x2_training_points[i, :]) - 1, max(x2_training_points[i, :]), step_size):
            f_x2_points = np.append(f_x2_points, j)
        f_x2_points = np.sort(f_x2_points)

        for x in f_x2_points:
            f_x = 0.0
            for xi in x2_training_points[i, :]:
                f_x = f_x + kernel_function(x, xi, kernel_cov)
            f_x = f_x / x2_training_points[i, :].size
            f_x2_values = np.append(f_x2_values, f_x)

        est_mean = 0.0
        for x in range(0, f_x2_points.size):
            est_mean = est_mean + parzen_expected_mean(f_x2_points[x], f_x2_values[x], step_size)
        x2_parzen_est_mean = np.append(x2_parzen_est_mean, est_mean)

        est_cov = 0.0
        for x in range(0, f_x2_points.size):
            est_cov = est_cov + parzen_expected_covariance(f_x2_points[x], f_x2_values[x], step_size, est_mean)
        x2_parzen_est_cov = np.append(x2_parzen_est_cov, est_cov)

    x1_parzen_est_mean = np.array(x1_parzen_est_mean)[np.newaxis]
    x1_parzen_est_mean = x1_parzen_est_mean.transpose()

    x2_parzen_est_mean = np.array(x2_parzen_est_mean)[np.newaxis]
    x2_parzen_est_mean = x2_parzen_est_mean.transpose()

    x1_parzen_est_cov = np.diag(x1_parzen_est_cov)
    x2_parzen_est_cov = np.diag(x2_parzen_est_cov)

    return x1_parzen_est_mean, x1_parzen_est_cov, x2_parzen_est_mean, x2_parzen_est_cov

