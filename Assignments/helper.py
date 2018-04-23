import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D


########################################################################################

# create the covariance matrices
def covariance_matrix(a, b, c, alpha, beta):
    # covariance matrix sigma1
    cov_matrix_1 = np.array([[np.math.pow(a, 2), beta * a * b, alpha * a * c],
                             [beta * a * b, np.math.pow(b, 2), beta * b * c],
                             [alpha * a * c, beta * b * c, np.math.pow(c, 2)]])

    # covariance matrix sigma2
    cov_matrix_2 = np.array([[np.math.pow(c, 2), alpha * b * c, beta * a * c],
                             [alpha * b * c, np.math.pow(b, 2), alpha * a * b],
                             [beta * a * c, alpha * a * b, np.math.pow(a, 2)]])
    return cov_matrix_1, cov_matrix_2


# generating gaussian random vectors from Uniform random variables
def generate_point():
    dim = 3
    point = []
    for d in range(0, dim):
        z = 0
        for i in range(0, 12):
            rand = np.random.uniform(0, 1)
            z = z + rand
        z = z - 6
        point.append([z])
    point = np.array(point)
    return point


# generate points from gaussian distribution and transform it back to class distribution
def generate_point_matrix(v, lambda_x, m, points):
    # create initial point
    z_matrix = generate_point()

    # convert them back to the classes distributions
    x_matrix = v @ np.power(lambda_x, 0.5) @ z_matrix + m

    # generate number of points and append them in an array
    for j in range(1, points):
        z_point = generate_point()
        z_matrix = np.append(z_matrix, z_point, axis=1)

        x = v @ np.power(lambda_x, 0.5) @ z_point + m
        x_matrix = np.append(x_matrix, x, axis=1)

    return z_matrix, x_matrix


########################################################################################

# PLOTTING #
def plot_2d_graph(w1_matrix, w2_matrix, d1, d2, d1_label, d2_label, title):
    plt.plot(w1_matrix[d1 - 1, :], w1_matrix[d2 - 1, :], 'b.', label="Class 1")
    plt.plot(w2_matrix[d1 - 1, :], w2_matrix[d2 - 1, :], 'r.', label="Class 2")
    plt.xlabel(d1_label)
    plt.ylabel(d2_label)
    max_w = max(max(max(w1_matrix[d1 - 1, :]), max(w2_matrix[d1 - 1, :])),
                max(max(w1_matrix[d2 - 1, :]), max(w2_matrix[d2 - 1, :])))
    min_w = min(min(min(w1_matrix[d1 - 1, :]), min(w2_matrix[d1 - 1, :])),
                min(min(w1_matrix[d2 - 1, :]), min(w2_matrix[d2 - 1, :])))
    plt.axis([min_w - 1, max_w + 1, min_w - 1, max_w + 1])
    plt.title(title)
    plt.legend(loc=2)
    plt.show()


def plot_3d_graph(w1_matrix, w2_matrix, d1_label, d2_label, d3_label, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(w1_matrix[0, :], w1_matrix[1, :], w1_matrix[2, :], c='b', marker='.', label="Class 1")
    ax.scatter(w2_matrix[0, :], w2_matrix[1, :], w2_matrix[2, :], c='r', marker='.', label="Class 2")
    max_w = max(max(max(w1_matrix[0, :]), max(w2_matrix[0, :])),
                max(max(w1_matrix[1, :]), max(w2_matrix[1, :])),
                max(max(w1_matrix[2, :]), max(w2_matrix[2, :])))
    min_w = min(min(min(w1_matrix[0, :]), min(w2_matrix[0, :])),
                min(min(w1_matrix[1, :]), min(w2_matrix[1, :])),
                min(min(w1_matrix[2, :]), min(w2_matrix[2, :])))
    ax.set_xlim(min_w, max_w)
    ax.set_ylim(min_w, max_w)
    ax.set_zlim(min_w, max_w)
    ax.set_xlabel(d1_label)
    ax.set_ylabel(d2_label)
    ax.set_zlabel(d3_label)
    ax.set_title(title)
    plt.show()


########################################################################################

# calculate the discriminant value for a given point or set of points
def calculate_discriminant(x, sigma_1, sigma_2, mean_1, mean_2, p1, p2):
    x = np.array(x)
    sigma_1 = np.array(sigma_1)
    sigma_2 = np.array(sigma_2)
    mean_1 = np.array(mean_1)
    mean_2 = np.array(mean_2)

    # print("points:\n", x)
    # print('sigma 1:\n', sigma_1)
    # print('sigma 2:\n', sigma_2)
    # print('mean 1:\n', mean_1)
    # print('mean 2:\n', mean_2)

    a = ((np.linalg.inv(sigma_2) - np.linalg.inv(sigma_1)) / 2)
    b = np.array(mean_1.transpose() @ np.linalg.inv(sigma_1) - mean_2.transpose() @ np.linalg.inv(sigma_2))
    c = np.math.log(p1 / p2) + np.log(np.linalg.det(sigma_2) / np.linalg.det(sigma_1))

    # print('A:\n', a)
    # print('B:\n', b.transpose())
    # print('C:\n', c)

    discriminant_value = x.transpose() @ a @ x + b @ x + c

    return discriminant_value


def remove_dimension(x, d):
    x = np.array(x)
    x[d - 1] = 0
    return x


def diagonalize():
    print()


def diagonalize_simultaneously(x1_matrix, x2_matrix, sigma_x1, sigma_x2, m1, m2):
    # eigenvalues and eigenvectors respectively
    w_x1, v_x1 = np.linalg.eig(sigma_x1)
    w_x2, v_x2 = np.linalg.eig(sigma_x2)

    # transform points for two classes in Y world
    y1_matrix = v_x1.transpose() @ x1_matrix
    y2_matrix = v_x1.transpose() @ x2_matrix

    # transform points for the two classes in Z
    z1_matrix = np.diag(np.power(w_x1, -0.5)) @ v_x1.transpose() @ x1_matrix
    z2_matrix = np.diag(np.power(w_x1, -0.5)) @ v_x1.transpose() @ x2_matrix

    # covariance matrix of z1 and z2
    sigma_z1 = np.diag(np.power(w_x1, -0.5)) @ np.diag(w_x1) @ np.diag(np.power(w_x1, -0.5))
    sigma_z2 = np.diag(np.power(w_x1, -0.5)) @ v_x1.transpose() @ sigma_x2 @ v_x1 @ np.diag(np.power(w_x1, -0.5))

    # eigenvalues and eigenvectors of z2 covariance
    w_z1, v_z1 = np.linalg.eig(sigma_z1)
    w_z2, v_z2 = np.linalg.eig(sigma_z2)

    # covariance matrix of v1 and v2
    sigma_v1 = np.round(v_z2.transpose() @ sigma_z1 @ v_z2, 2)
    sigma_v2 = np.round(v_z2.transpose() @ sigma_z2 @ v_z2, 2)

    # P overall
    p_overall = v_z2.transpose() @ np.diag(np.power(w_x1, -0.5)) @ v_x1.transpose()

    # means for y1 and y2
    m_y1 = v_x1.transpose() @ m1
    m_y2 = v_x2.transpose() @ m2

    # means for z1 and z2
    m_z1 = v_x1.transpose() @ m_y1
    m_z2 = v_x2.transpose() @ m_y2

    # means for v1 and v2
    m_v1 = v_x1.transpose() @ m_z1
    m_v2 = v_x2.transpose() @ m_z2

    # means for v1 and v2
    m_v1 = p_overall @ m1
    m_v2 = p_overall @ m2

    # transform points for the two classes in V
    v1_matrix = p_overall @ x1_matrix
    v2_matrix = p_overall @ x2_matrix

    return v1_matrix, v2_matrix, sigma_v1, sigma_v2, m_v1, m_v2


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


def kernel_function(x, xi, cov):
    result = (1 / (math.sqrt(2 * math.pi) * cov)) * math.exp(-math.pow(x - xi, 2) / (2 * math.pow(cov, 2)))
    return result


def parzen_expected_mean(x, f_x, delta_x):
    return x * f_x * delta_x


def parzen_expected_covariance(x, f_x, delta_x, mean):
    return math.pow(x - mean, 2) * f_x * delta_x


def calculate_disc_function_with_plot(m1, m2, cov1, cov2, x1_points, x2_points, p1, p2, d1, d2, method):
    a = ((np.linalg.inv(cov2) - np.linalg.inv(cov1)) / 2)
    b = np.array(m1.transpose() @ np.linalg.inv(
        cov1) - m2.transpose() @ np.linalg.inv(cov2))
    c = np.math.log(p1 / p2) + np.log(np.linalg.det(cov2) / np.linalg.det(cov1))

    equation_points = []
    roots_1 = []
    roots_2 = []

    min_w = min(min(min(x1_points[d1 - 1, :]), min(x2_points[d1 - 1, :])),
                min(min(x1_points[d2 - 1, :]), min(x2_points[d2 - 1, :])))
    max_w = max(max(max(x1_points[d1 - 1, :]), max(x2_points[d1 - 1, :])),
                max(max(x1_points[d2 - 1, :]), max(x2_points[d2 - 1, :])))

    for x1 in np.arange(min_w - 1, max_w + 1, 1):
        equation_points.append(x1)
        x2_square_coefficient = a[d2 - 1][d2 - 1]
        x2_coefficient = (a[d1 - 1][d2 - 1] * x1) + (a[d2 - 1][d1 - 1] * x1) + b[d1 - 1][d2 - 1]
        constant = a[d1 - 1][d1 - 1] * np.math.pow(x1, 2) + b[d1 - 1][d1 - 1] * x1 + c

        poly_coefficients = [x2_square_coefficient, x2_coefficient, constant]
        roots = np.roots(poly_coefficients)
        roots_1.append(roots[0])
        roots_2.append(roots[1])

    plt.plot(x1_points[d1 - 1, :], x1_points[d2 - 1, :], 'b.', label="Class 1")
    plt.plot(x2_points[d1 - 1, :], x2_points[d2 - 1, :], 'r.', label="Class 2")
    plt.plot(equation_points, roots_2, 'g--', label="Dis.Fnc.")
    plt.plot(equation_points, roots_1, 'y--', label="Dis.Fnc.")
    plt.xlabel('x' + str(d1))
    plt.ylabel('x' + str(d2))

    plt.axis([min_w - 1, max_w + 1, min_w - 1, max_w + 1])
    plt.title('Dis. Fun. for ' + method + ' for x' + str(d1) + '-x' + str(d2))
    plt.legend(loc=2)
    plt.show()


def bl_expected_mean(x1_training_points, x2_training_points, sigma_x1, sigma_x2, m1, m2, number_of_points):
    w1_plot_points = [[],
                      [],
                      []]
    w1_plot_points_index = []
    w1_mean_values = [[],
                      [],
                      []]
    w1_ml_mean_values = [[],
                         [],
                         []]
    w1_ml_cov_values = [[],
                        [],
                        []]

    w1_mean_difference = []
    w1_ml_mean_difference = []
    w1_ml_cov_difference = []
    sigma0 = np.identity(3)
    m1_0 = np.array([[0],
                     [1],
                     [0]])

    w2_plot_points = [[],
                      [],
                      []]
    w2_plot_points_index = []
    w2_mean_values = [[],
                      [],
                      []]
    w2_ml_mean_values = [[],
                         [],
                         []]
    w2_ml_cov_values = [[], [], []]
    w2_mean_difference = []
    w2_ml_mean_difference = []
    w2_ml_cov_difference = []
    m2_0 = np.array([[0],
                     [1],
                     [0]])

    for n in range(1, number_of_points, 1):
        # for class 1
        w1_plot_points_index = np.append(w1_plot_points_index, n)

        x1_ml_estimated_mean = estimate_mean_ml(x1_training_points, n)
        x1_ml_estimated_cov = estimate_cov_ml(x1_training_points, x1_ml_estimated_mean, n)
        w1_ml_mean_difference = np.append(w1_ml_mean_difference, np.absolute(np.linalg.norm(x1_ml_estimated_mean - m1)))
        w1_ml_cov_difference = np.append(w1_ml_cov_difference,
                                         np.absolute(np.linalg.norm(x1_ml_estimated_cov - sigma_x1)))

        m1_0 = estimate_mean_bl(x1_training_points, m1_0, sigma0, sigma_x1, n)
        w1_mean_values = np.append(w1_mean_values, m1_0, axis=1)
        w1_plot_points = np.append(w1_plot_points, x1_training_points[:, [n]], axis=1)
        w1_mean_difference = np.append(w1_mean_difference, np.absolute(np.linalg.norm(m1_0 - m1)))

        # for class 2
        w2_plot_points_index = np.append(w2_plot_points_index, n)

        x2_ml_estimated_mean = estimate_mean_ml(x2_training_points, n)
        x2_ml_estimated_cov = estimate_cov_ml(x2_training_points, x2_ml_estimated_mean, n)
        w2_ml_mean_difference = np.append(w2_ml_mean_difference, np.absolute(np.linalg.norm(x2_ml_estimated_mean - m2)))
        w2_ml_cov_difference = np.append(w2_ml_cov_difference,
                                         np.absolute(np.linalg.norm(x2_ml_estimated_cov - sigma_x2)))

        m2_0 = estimate_mean_bl(x2_training_points, m2_0, sigma0, sigma_x2, n)
        w2_mean_values = np.append(w2_mean_values, m2_0, axis=1)
        w2_plot_points = np.append(w2_plot_points, x2_training_points[:, [n]], axis=1)
        w2_mean_difference = np.append(w2_mean_difference, np.absolute(np.linalg.norm(m2_0 - m2)))
    return m1_0, m2_0,


def estimated_mean_parzen(x1_training_points, x2_training_points, kernel_covariance, step_size):
    x1_parzen_estimated_mean = []
    x1_parzen_estimated_covariance = []

    x2_parzen_estimated_mean = []
    x2_parzen_estimated_covariance = []

    for i in range(0, 3, 1):
        # for class 1
        f_x1_points = []
        f_x1_values = []
        for j in np.arange(min(x1_training_points[i, :]) - 1, max(x1_training_points[i, :]), step_size):
            f_x1_points = np.append(f_x1_points, j)

        f_x1_points = np.sort(f_x1_points)

        for x in f_x1_points:
            f_x = 0.0
            for xi in x1_training_points[i, :]:
                f_x = f_x + kernel_function(x, xi, kernel_covariance)
            f_x = f_x / x1_training_points[i, :].size
            f_x1_values = np.append(f_x1_values, f_x)

        estimated_mean = 0.0
        for x in range(0, f_x1_points.size):
            estimated_mean = estimated_mean + parzen_expected_mean(f_x1_points[x], f_x1_values[x], step_size)
        x1_parzen_estimated_mean = np.append(x1_parzen_estimated_mean, estimated_mean)

        estimated_covariance = 0.0
        for x in range(0, f_x1_points.size):
            estimated_covariance = estimated_covariance + parzen_expected_covariance(f_x1_points[x], f_x1_values[x],
                                                                                     step_size, estimated_mean)
        x1_parzen_estimated_covariance = np.append(x1_parzen_estimated_covariance, estimated_covariance)

        # for class 2
        f_x2_points = []
        f_x2_values = []
        for j in np.arange(min(x2_training_points[i, :]) - 1, max(x2_training_points[i, :]), step_size):
            f_x2_points = np.append(f_x2_points, j)
        f_x2_points = np.sort(f_x2_points)

        for x in f_x2_points:
            f_x = 0.0
            for xi in x2_training_points[i, :]:
                f_x = f_x + kernel_function(x, xi, kernel_covariance)
            f_x = f_x / x2_training_points[i, :].size
            f_x2_values = np.append(f_x2_values, f_x)

        estimated_mean = 0.0
        for x in range(0, f_x2_points.size):
            estimated_mean = estimated_mean + parzen_expected_mean(f_x2_points[x], f_x2_values[x], step_size)
        x2_parzen_estimated_mean = np.append(x2_parzen_estimated_mean, estimated_mean)

        estimated_covariance = 0.0
        for x in range(0, f_x2_points.size):
            estimated_covariance = estimated_covariance + parzen_expected_covariance(f_x2_points[x], f_x2_values[x],
                                                                                     step_size, estimated_mean)
        x2_parzen_estimated_covariance = np.append(x2_parzen_estimated_covariance, estimated_covariance)

    x1_parzen_estimated_mean = np.array(x1_parzen_estimated_mean)[np.newaxis]
    x1_parzen_estimated_mean = x1_parzen_estimated_mean.transpose()

    x2_parzen_estimated_mean = np.array(x2_parzen_estimated_mean)[np.newaxis]
    x2_parzen_estimated_mean = x2_parzen_estimated_mean.transpose()

    x1_parzen_estimated_covariance = np.diag(x1_parzen_estimated_covariance)
    x2_parzen_estimated_covariance = np.diag(x2_parzen_estimated_covariance)

    return x1_parzen_estimated_mean, x1_parzen_estimated_covariance, x2_parzen_estimated_mean, x2_parzen_estimated_covariance


def test_classifier(class1_test_points, class2_test_points, x1_ml_estimated_cov, x2_ml_estimated_cov,
                    x1_ml_estimated_mean, x2_ml_estimated_mean, number_of_testing_points):
    # classification results
    class1_true = 0.0
    class1_false = 0.0

    class2_true = 0.0
    class2_false = 0.0
    # print(class1_test_points[:, 1])
    # classify each point
    for j in range(number_of_testing_points - 1):
        discriminant_value = calculate_discriminant(class1_test_points[:, j], x1_ml_estimated_cov,
                                                    x2_ml_estimated_cov, x1_ml_estimated_mean, x2_ml_estimated_mean,
                                                    0.5,
                                                    0.5)
        if discriminant_value > 0:
            class1_true += 1
        else:
            class1_false += 1

    for j in range(number_of_testing_points - 1):
        discriminant_value = calculate_discriminant(class2_test_points[:, j], x1_ml_estimated_cov,
                                                    x2_ml_estimated_cov, x1_ml_estimated_mean, x2_ml_estimated_mean,
                                                    0.5,
                                                    0.5)
        if discriminant_value < 0:
            class2_true += 1
        else:
            class2_false += 1

    class1_accuracy = (class1_true / number_of_testing_points) * 100
    class2_accuracy = (class2_true / number_of_testing_points) * 100

    return class1_accuracy, class2_accuracy
