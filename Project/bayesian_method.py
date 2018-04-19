import numpy as np
import max_likelihood as ml


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


def bl_expected_mean(x1_training_points, x2_training_points, sigma_x1, sigma_x2, m1, m2):
    w1_plot_points = [[], [], []]
    w1_plot_points_index = []
    w1_mean_values = [[], [], []]

    w1_mean_difference = []
    w1_ml_mean_difference = []
    w1_ml_cov_difference = []
    sigma0 = np.identity(len(x1_training_points[0]))
    m1_0 = np.array([[1], [1], [1]])

    w2_plot_points = [[], [], []]
    w2_plot_points_index = []
    w2_mean_values = [[], [], []]

    w2_mean_difference = []
    w2_ml_mean_difference = []
    w2_ml_cov_difference = []
    m2_0 = np.array([[1], [1], [1]])

    for n in range(1, len(x1_training_points[0]), 1):
        # for class 1
        w1_plot_points_index = np.append(w1_plot_points_index, n)

        x1_ml_estimated_mean = ml.estimate_mean_ml(x1_training_points, n)
        x1_ml_estimated_cov = ml.estimate_cov_ml(x1_training_points, x1_ml_estimated_mean, n)
        w1_ml_mean_difference = np.append(w1_ml_mean_difference, np.absolute(np.linalg.norm(x1_ml_estimated_mean - m1)))
        w1_ml_cov_difference = np.append(w1_ml_cov_difference,
                                         np.absolute(np.linalg.norm(x1_ml_estimated_cov - sigma_x1)))

        m1_0 = estimate_mean_bl(x1_training_points, m1_0, sigma0, sigma_x1, n)
        w1_mean_values = np.append(w1_mean_values, m1_0, axis=1)
        w1_plot_points = np.append(w1_plot_points, x1_training_points[:, [n]], axis=1)
        w1_mean_difference = np.append(w1_mean_difference, np.absolute(np.linalg.norm(m1_0 - m1)))

    for n in range(1, len(x2_training_points[0]), 1):
        # for class 2
        w2_plot_points_index = np.append(w2_plot_points_index, n)

        x2_ml_estimated_mean = ml.estimate_mean_ml(x2_training_points, n)
        x2_ml_estimated_cov = ml.estimate_cov_ml(x2_training_points, x2_ml_estimated_mean, n)
        w2_ml_mean_difference = np.append(w2_ml_mean_difference, np.absolute(np.linalg.norm(x2_ml_estimated_mean - m2)))
        w2_ml_cov_difference = np.append(w2_ml_cov_difference,
                                         np.absolute(np.linalg.norm(x2_ml_estimated_cov - sigma_x2)))

        m2_0 = estimate_mean_bl(x2_training_points, m2_0, sigma0, sigma_x2, n)
        w2_mean_values = np.append(w2_mean_values, m2_0, axis=1)
        w2_plot_points = np.append(w2_plot_points, x2_training_points[:, [n]], axis=1)
        w2_mean_difference = np.append(w2_mean_difference, np.absolute(np.linalg.norm(m2_0 - m2)))
    return m1_0, m2_0,


