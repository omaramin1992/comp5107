import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# ----------------------------------------------------------------------


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


def calculate_discriminant(x, sigma_1, sigma_2, mean_1, mean_2, p1, p2):
    x = np.array([x])
    x = x.transpose()
    sigma_1 = np.array(sigma_1, dtype=np.float64)
    sigma_2 = np.array(sigma_2, dtype=np.float64)
    mean_1 = np.array(mean_1, dtype=np.float64)
    mean_2 = np.array(mean_2, dtype=np.float64)

    # print("point:\n", x)
    # print('sigma 1:\n', sigma_1)
    # print('sigma 2:\n', sigma_2)
    # print('mean 1:\n', mean_1)
    # print('mean 2:\n', mean_2)

    a = ((np.linalg.inv(sigma_2) - np.linalg.inv(sigma_1)) / 2)
    b = np.array(mean_1.transpose() @ np.linalg.inv(sigma_1) - mean_2.transpose() @ np.linalg.inv(sigma_2),
                 dtype=np.float64)
    c = np.math.log(p1 / p2) + np.log(np.linalg.det(sigma_2) / np.linalg.det(sigma_1))
    # print(np.linalg.inv(sigma_1))
    # print(np.linalg.inv(sigma_2))
    # print(mean_1.transpose() @ np.linalg.inv(sigma_1))
    # print(mean_2.transpose() @ np.linalg.inv(sigma_2))
    # print('A:\n', a)
    # print('B:\n', b.transpose())
    # print('C:\n', c)

    discriminant_value = x.transpose() @ a @ x + b @ x + c
    # print("discriminant value: ", discriminant_value)
    return discriminant_value


# ----------------------------------------------------------------------


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


def normalize_data(data1, data2, low, high):
    data1 = np.array(data1)
    data2 = np.array(data2)
    full_data = np.append(data1, data2, axis=1)
    print(full_data.shape)
    mins = np.array([np.min(full_data, axis=1)]).transpose()
    print("mins:", mins)
    maxs = np.array([np.max(full_data, axis=1)]).transpose()
    print("maxs:", maxs)
    rng = maxs - mins
    data1_normalized = high - (((high - low) * (maxs - data1)) / rng)
    data2_normalized = high - (((high - low) * (maxs - data2)) / rng)
    # np_minmax = (dat1 - data.min()) / (data.max() - data.min())
    return data1_normalized, data2_normalized
