import numpy as np
import helper as h
from random import uniform
from prettytable import PrettyTable
import matplotlib.pyplot as plt

"""
Consider the two-class pattern recognition problem in which the class conditional distributions are both
normally distributed with arbitrary means M1 and M2, and covariance matrices Sigma1 and Sigma2 respectively.
Assume that you are working in a 3-D space (for example, as in Assignment II) and that the covariance
matrices are not equal.
"""

m1 = np.array([[3],
               [1],
               [4]])

m2 = np.array([[-3],
               [1],
               [-4]])

# parameters of covariance matrices
a1 = 2
b1 = 3
c1 = 4

alpha = 0.1
beta = 0.2

number_of_points = 200

p1 = 0.5
p2 = 0.5

########################################################

"""
(a): Generate 200 training points of each distribution before diagonalization and plot them in the (x1–x2) and (
x1– x3) domains. 
"""
# creating the covariance matrices with the parameters
sigma_x1, sigma_x2 = h.covariance_matrix(a1, b1, c1, alpha, beta)
print('Sigma x1:')
print(sigma_x1)
print('\nSigma x2:')
print(sigma_x2)

# eigenvalues and eigenvectors respectively
w_x1, v_x1 = np.linalg.eig(sigma_x1)
lambda_x1 = np.diag(w_x1)

w_x2, v_x2 = np.linalg.eig(sigma_x2)
lambda_x2 = np.diag(w_x2)
print('\neigenvalues of x1:')
print(lambda_x1)
print('\neigenvalues of x2:')
print(lambda_x2)

# create point matrices for the two classes X1 and X2
z1_training_points, x1_training_points = h.generate_point_matrix(v_x1, lambda_x1, m1, number_of_points)
z2_training_points, x2_training_points = h.generate_point_matrix(v_x2, lambda_x2, m2, number_of_points)

# PLOTTING #
# X WORLD
# plot the first class as blue for (d1 - d2) domain and second class as red
h.plot_2d_graph(x1_training_points, x2_training_points, 1, 2, 'x1', 'x2', 'x1-x2')

# plot the first class as blue for (d1 - d3) domain and second class as red
h.plot_2d_graph(x1_training_points, x2_training_points, 1, 3, 'x1', 'x3', 'x1-x3')

########################################################

"""
(b): Using these training points estimate the parameters of each distribution using a maximum
likelihood and a Bayesian methodology. In the latter, assume that you know the covariances. Plot
the convergence of the parameters with the number of samples in each case. 
"""

# get estimated mean and covariance using ML method
x1_ml_estimated_mean = h.estimate_mean_ml(x1_training_points, number_of_points)
x1_ml_estimated_cov = h.estimate_cov_ml(x1_training_points, x1_ml_estimated_mean, number_of_points)
print('\nEstimated Mean1 using ML:')
print(x1_ml_estimated_mean)
print('\nEstimated covariance1 using ML:')
print(x1_ml_estimated_cov)

x2_ml_estimated_mean = h.estimate_mean_ml(x2_training_points, number_of_points)
x2_ml_estimated_cov = h.estimate_cov_ml(x2_training_points, x2_ml_estimated_mean, number_of_points)
print('\nEstimated Mean2 using ML:')
print(x2_ml_estimated_mean)
print('\nEstimated covariance2 using ML:')
print(x2_ml_estimated_cov)

# Estimating the means using BL
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
w2_ml_cov_values = [[],
                    [],
                    []]
w2_mean_difference = []
w2_ml_mean_difference = []
w2_ml_cov_difference = []
m2_0 = np.array([[0],
                 [1],
                 [0]])

for n in range(1, number_of_points, 1):
    # for class 1
    w1_plot_points_index = np.append(w1_plot_points_index, n)

    x1_ml_estimated_mean = h.estimate_mean_ml(x1_training_points, n)
    x1_ml_estimated_cov = h.estimate_cov_ml(x1_training_points, x1_ml_estimated_mean, n)
    w1_ml_mean_difference = np.append(w1_ml_mean_difference, np.absolute(np.linalg.norm(x1_ml_estimated_mean - m1)))
    w1_ml_cov_difference = np.append(w1_ml_cov_difference, np.absolute(np.linalg.norm(x1_ml_estimated_cov - sigma_x1)))

    m1_0 = h.estimate_mean_bl(x1_training_points, m1_0, sigma0, sigma_x1, n)
    w1_mean_values = np.append(w1_mean_values, m1_0, axis=1)
    w1_plot_points = np.append(w1_plot_points, x1_training_points[:, [n]], axis=1)
    w1_mean_difference = np.append(w1_mean_difference, np.absolute(np.linalg.norm(m1_0 - m1)))

    # for class 2
    w2_plot_points_index = np.append(w2_plot_points_index, n)

    x2_ml_estimated_mean = h.estimate_mean_ml(x2_training_points, n)
    x2_ml_estimated_cov = h.estimate_cov_ml(x2_training_points, x2_ml_estimated_mean, n)
    w2_ml_mean_difference = np.append(w2_ml_mean_difference, np.absolute(np.linalg.norm(x2_ml_estimated_mean - m2)))
    w2_ml_cov_difference = np.append(w2_ml_cov_difference, np.absolute(np.linalg.norm(x2_ml_estimated_cov - sigma_x2)))

    m2_0 = h.estimate_mean_bl(x2_training_points, m2_0, sigma0, sigma_x2, n)
    w2_mean_values = np.append(w2_mean_values, m2_0, axis=1)
    w2_plot_points = np.append(w2_plot_points, x2_training_points[:, [n]], axis=1)
    w2_mean_difference = np.append(w2_mean_difference, np.absolute(np.linalg.norm(m2_0 - m2)))

print('\nEstimated mean 1 using BL:')
print(m1_0)
print('\nEstimated mean 2 using BL:')
print(m2_0)

# PLOTTING #

# mean difference plots for ML
plt.plot(w1_plot_points_index, w1_ml_mean_difference, 'r-', label="class 1")
plt.plot(w2_plot_points_index, w2_ml_mean_difference, 'b-', label="class 2")

plt.xlabel('n')
plt.ylabel('Mean Diff')

plt.axis([0, 200, 0, 5])
plt.title('Mean Convergence using ML')
plt.legend(loc=2)
plt.show()

# mean difference plots for ML
plt.plot(w1_plot_points_index, w1_ml_cov_difference, 'r-', label="class 1")
plt.plot(w2_plot_points_index, w2_ml_cov_difference, 'b-', label="class 2")

plt.xlabel('n')
plt.ylabel('Covariance Diff')

plt.axis([0, 200, 0, 200])
plt.title('Covariance Convergence using ML')
plt.legend(loc=2)
plt.show()

# plots for BL
plt.plot(w1_plot_points_index, w1_mean_difference, 'r-', label="class 1")
plt.plot(w2_plot_points_index, w2_mean_difference, 'b-', label="class 2")

plt.xlabel('n')
plt.ylabel('Mean Diff')

plt.axis([0, 200, 0, 5])
plt.title('Mean Convergence using BL')
plt.legend(loc=2)
plt.show()

########################################################

"""
(c): Using these same training points estimate each uni-variate distribution using a Parzen Window
approach. In this case, work with the features in each dimension separately, and with an
appropriate Gaussian kernel. For the output, you must plot the final learned distribution of the
features in each dimension, and print out their “sample” mean and variance in each dimension.
"""

kernel_covariance = 0.3

for i in range(0, 3, 1):
    # for class 1
    f_x1_points = []
    f_x1_values = []
    for j in range(0, 100):
        x = uniform(min(x1_training_points[i, :]), max(x1_training_points[i, :]))
        f_x1_points = np.append(f_x1_points, x)
    f_x1_points = np.sort(f_x1_points)

    for x in f_x1_points:
        f_x = 0.0
        for xi in x1_training_points[i, :]:
            f_x = f_x + h.kernel_function(x, xi, kernel_covariance)
        f_x1_values = np.append(f_x1_values, f_x)

    # for class 2
    f_x2_points = []
    f_x2_values = []
    for j in range(0, 100):
        x = uniform(min(x2_training_points[i, :]), max(x2_training_points[i, :]))
        f_x2_points = np.append(f_x2_points, x)
    f_x2_points = np.sort(f_x2_points)

    for x in f_x2_points:
        f_x = 0.0
        for xi in x2_training_points[i, :]:
            f_x = f_x + h.kernel_function(x, xi, kernel_covariance)
        f_x2_values = np.append(f_x2_values, f_x)

    min_x_axis = min(min(x1_training_points[i, :]), min(x2_training_points[i, :]))
    max_x_axis = max(max(x1_training_points[i, :]), max(x2_training_points[i, :]))

    # parzen window plots
    plt.plot(f_x1_points, f_x1_values, 'r--', label="class 1")
    plt.plot(f_x2_points, f_x2_values, 'b--', label="class 2")

    plt.xlabel('x values')
    plt.ylabel('f(x)')

    plt.xlim(min_x_axis, max_x_axis)
    plt.title('f(x)')
    plt.legend(loc=2)
    plt.show()

########################################################

"""
(d): Using the estimated distributions, compute the optimal Bayes discriminant function (for the ML,
Bayes and Parzen schemes) and plot it in the (x1– x2) and (x1– x3) domains. 
"""

########################################################

"""
(e): Generate 200 new points for each class for testing purposes, classify them and report the
classification accuracy. Do this (i.e., using all 400 points) using a ten-fold cross validation. 
"""

########################################################

"""
(f): Repeat (a)-(d) for the same data after you have diagonalized it.  
"""
