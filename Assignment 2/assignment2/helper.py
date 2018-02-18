import matplotlib.pyplot as plt
import numpy as np
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
