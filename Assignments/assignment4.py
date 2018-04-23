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
print('\nEstimated Mean1 using ML before Diagonalization:')
print(x1_ml_estimated_mean)
print('\nEstimated covariance1 using ML before Diagonalization:')
print(x1_ml_estimated_cov)

x2_ml_estimated_mean = h.estimate_mean_ml(x2_training_points, number_of_points)
x2_ml_estimated_cov = h.estimate_cov_ml(x2_training_points, x2_ml_estimated_mean, number_of_points)
print('\nEstimated Mean2 using ML before Diagonalization:')
print(x2_ml_estimated_mean)
print('\nEstimated covariance2 using ML before Diagonalization:')
print(x2_ml_estimated_cov)

# Estimating the means using BL

x1_bl_estimated_mean, x2_bl_estimated_mean = h.bl_expected_mean(x1_training_points, x2_training_points, sigma_x1,
                                                                sigma_x2, m1, m2, number_of_points)

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

print('\nEstimated mean 1 using BL before Diagonalization:')
print(x1_bl_estimated_mean)
print('\nEstimated mean 2 using BL before Diagonalization:')
print(x2_bl_estimated_mean)

# PLOTTING #

# mean difference plots for ML
plt.plot(w1_plot_points_index, w1_ml_mean_difference, 'r-', label="class 1")
plt.plot(w2_plot_points_index, w2_ml_mean_difference, 'b-', label="class 2")

plt.xlabel('n')
plt.ylabel('Mean Diff')

plt.axis([0, 200, 0, 5])
plt.title('Mean Convergence using ML before Diagonalization')
plt.legend(loc=2)
plt.show()

# mean difference plots for ML
plt.plot(w1_plot_points_index, w1_ml_cov_difference, 'r-', label="class 1")
plt.plot(w2_plot_points_index, w2_ml_cov_difference, 'b-', label="class 2")

plt.xlabel('n')
plt.ylabel('Covariance Diff')

plt.axis([0, 200, 0, 200])
plt.title('Covariance Convergence using ML before Diagonalization')
plt.legend(loc=2)
plt.show()

# plots for BL
plt.plot(w1_plot_points_index, w1_mean_difference, 'r-', label="class 1")
plt.plot(w2_plot_points_index, w2_mean_difference, 'b-', label="class 2")

plt.xlabel('n')
plt.ylabel('Mean Diff')

plt.axis([0, 200, 0, 5])
plt.title('Mean Convergence using BL before Diagonalization')
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
kernel_points = 100

step_size = 0.01

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
            f_x = f_x + h.kernel_function(x, xi, kernel_covariance)
        f_x = f_x / x1_training_points[i, :].size
        f_x1_values = np.append(f_x1_values, f_x)

    estimated_mean = 0.0
    for x in range(0, f_x1_points.size):
        estimated_mean = estimated_mean + h.parzen_expected_mean(f_x1_points[x], f_x1_values[x], step_size)
    x1_parzen_estimated_mean = np.append(x1_parzen_estimated_mean, estimated_mean)

    estimated_covariance = 0.0
    for x in range(0, f_x1_points.size):
        estimated_covariance = estimated_covariance + h.parzen_expected_covariance(f_x1_points[x], f_x1_values[x],
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
            f_x = f_x + h.kernel_function(x, xi, kernel_covariance)
        f_x = f_x / x2_training_points[i, :].size
        f_x2_values = np.append(f_x2_values, f_x)

    estimated_mean = 0.0
    for x in range(0, f_x2_points.size):
        estimated_mean = estimated_mean + h.parzen_expected_mean(f_x2_points[x], f_x2_values[x], step_size)
    x2_parzen_estimated_mean = np.append(x2_parzen_estimated_mean, estimated_mean)

    estimated_covariance = 0.0
    for x in range(0, f_x2_points.size):
        estimated_covariance = estimated_covariance + h.parzen_expected_covariance(f_x2_points[x], f_x2_values[x],
                                                                                   step_size, estimated_mean)
    x2_parzen_estimated_covariance = np.append(x2_parzen_estimated_covariance, estimated_covariance)

    min_x_axis = min(min(x1_training_points[i, :]), min(x2_training_points[i, :]))
    max_x_axis = max(max(x1_training_points[i, :]), max(x2_training_points[i, :]))

    title = 'f(x' + str(i + 1) + ')'

    # parzen window plots
    plt.plot(f_x1_points, f_x1_values, 'r--', label="class 1")
    plt.plot(f_x2_points, f_x2_values, 'b--', label="class 2")

    plt.xlabel('x values')
    plt.ylabel('f(x)')

    plt.xlim(min_x_axis, max_x_axis)
    plt.title(title)
    plt.legend(loc=2)
    plt.show()

print('\nEstimated mean 1 using Parzen before Diagonalization:')
x1_parzen_estimated_mean = np.array(x1_parzen_estimated_mean)[np.newaxis]
x1_parzen_estimated_mean = x1_parzen_estimated_mean.transpose()
print(x1_parzen_estimated_mean)
print('\nEstimated mean 2 using Parzen before Diagonalization:')
x2_parzen_estimated_mean = np.array(x2_parzen_estimated_mean)[np.newaxis]
x2_parzen_estimated_mean = x2_parzen_estimated_mean.transpose()
print(x2_parzen_estimated_mean)

print('\nEstimated covariance 1 using Parzen before Diagonalization:')
x1_parzen_estimated_covariance = np.diag(x1_parzen_estimated_covariance)
print(x1_parzen_estimated_covariance)
print('\nEstimated covariance 2 using Parzen before Diagonalization:')
x2_parzen_estimated_covariance = np.diag(x2_parzen_estimated_covariance)
print(x2_parzen_estimated_covariance)

########################################################

"""
(d): Using the estimated distributions, compute the optimal Bayes discriminant function (for the ML,
Bayes and Parzen schemes) and plot it in the (x1– x2) and (x1– x3) domains. 
"""
# discriminant function for ML
h.calculate_disc_function_with_plot(x1_ml_estimated_mean, x2_ml_estimated_mean, x1_ml_estimated_cov,
                                    x2_ml_estimated_cov, x1_training_points, x2_training_points, p1, p2, 1, 2, 'ML')
h.calculate_disc_function_with_plot(x1_ml_estimated_mean, x2_ml_estimated_mean, x1_ml_estimated_cov,
                                    x2_ml_estimated_cov, x1_training_points, x2_training_points, p1, p2, 1, 3, 'ML')

# discriminant function for Bayes
h.calculate_disc_function_with_plot(x1_bl_estimated_mean, x2_bl_estimated_mean, sigma_x1,
                                    sigma_x2, x1_training_points, x2_training_points, p1, p2, 1, 2, 'BL')
h.calculate_disc_function_with_plot(x1_bl_estimated_mean, x2_bl_estimated_mean, sigma_x1,
                                    sigma_x2, x1_training_points, x2_training_points, p1, p2, 1, 3, 'BL')

# discriminant function for Parzen
h.calculate_disc_function_with_plot(x1_parzen_estimated_mean, x2_parzen_estimated_mean, x1_parzen_estimated_covariance,
                                    x2_parzen_estimated_covariance, x1_training_points, x2_training_points, p1, p2, 1,
                                    2, 'Parzen')
h.calculate_disc_function_with_plot(x1_parzen_estimated_mean, x2_ml_estimated_mean, x1_parzen_estimated_covariance,
                                    x2_parzen_estimated_covariance, x1_training_points, x2_training_points, p1, p2, 1,
                                    3, 'Parzen')

########################################################

"""
(e): Generate 200 new points for each class for testing purposes, classify them and report the
classification accuracy. Do this (i.e., using all 400 points) using a ten-fold cross validation. 
"""

# number of testing points
test_points_count = 200

# create testing points
_, x1_test_points = h.generate_point_matrix(v_x1, lambda_x1, m1, test_points_count)
_, x2_test_points = h.generate_point_matrix(v_x2, lambda_x2, m2, test_points_count)

test_results_ml_class1 = []
test_results_ml_class2 = []

test_results_bl_class1 = []
test_results_bl_class2 = []

test_results_parzen_class1 = []
test_results_parzen_class2 = []

k = 10

class1_total_points = x1_training_points
class1_total_points = np.append(class1_total_points, x1_test_points, axis=1)

class2_total_points = x2_training_points
class2_total_points = np.append(class2_total_points, x2_test_points, axis=1)

print(class1_total_points[:, 399])
n = number_of_points + test_points_count
for i in range(0, k, 1):
    print('Cross:' + str(i + 1))
    number_of_testing_points = int(n / k)
    number_of_training_points = int(n - n / k)
    start = int(n * i / k)
    end = int((i + 1) * n / k - 1)

    class1_test_points = class1_total_points[:, start: end]
    class1_train_points = class1_total_points[:, 0:start]
    class1_train_points = np.append(class1_train_points, class1_total_points[:, end:], axis=1)

    class2_test_points = class2_total_points[:, start: end]
    class2_train_points = class2_total_points[:, 0:start]
    class2_train_points = np.append(class2_train_points, class2_total_points[:, end:], axis=1)

    # estimated mean using ML
    x1_ml_estimated_mean = h.estimate_mean_ml(class1_train_points, number_of_training_points)
    x1_ml_estimated_cov = h.estimate_cov_ml(class1_train_points, x1_ml_estimated_mean, number_of_training_points)

    x2_ml_estimated_mean = h.estimate_mean_ml(class2_train_points, number_of_points)
    x2_ml_estimated_cov = h.estimate_cov_ml(class2_train_points, x2_ml_estimated_mean, number_of_training_points)

    # Estimating the means using BL
    x1_bl_estimated_mean, x2_bl_estimated_mean = h.bl_expected_mean(class1_train_points, class2_train_points, sigma_x1,
                                                                    sigma_x2, m1, m2, number_of_training_points)

    # estimated mean and cov using parzen window
    x1_parzen_estimated_mean, x1_parzen_estimated_covariance, x2_parzen_estimated_mean, x2_parzen_estimated_covariance = h.estimated_mean_parzen(
        class1_train_points, class2_train_points, kernel_covariance, step_size)

    ml_class1_accuracy, ml_class2_accuracy = h.test_classifier(class1_test_points, class2_test_points,
                                                               x1_ml_estimated_cov, x2_ml_estimated_cov,
                                                               x1_ml_estimated_mean, x2_ml_estimated_mean,
                                                               number_of_testing_points)
    test_results_ml_class1 = np.append(test_results_ml_class1, ml_class1_accuracy)
    test_results_ml_class2 = np.append(test_results_ml_class2, ml_class2_accuracy)

    bl_class1_accuracy, bl_class2_accuracy = h.test_classifier(class1_test_points, class2_test_points,
                                                               sigma_x1, sigma_x2,
                                                               x1_bl_estimated_mean, x2_bl_estimated_mean,
                                                               number_of_testing_points)
    test_results_bl_class1 = np.append(test_results_bl_class1, bl_class1_accuracy)
    test_results_bl_class2 = np.append(test_results_bl_class2, bl_class2_accuracy)

    parzen_class1_accuracy, parzen_class2_accuracy = h.test_classifier(class1_test_points, class2_test_points,
                                                                       x1_parzen_estimated_covariance,
                                                                       x2_parzen_estimated_covariance,
                                                                       x1_parzen_estimated_mean,
                                                                       x2_parzen_estimated_mean,
                                                                       number_of_testing_points)
    test_results_parzen_class1 = np.append(test_results_parzen_class1, parzen_class1_accuracy)
    test_results_parzen_class2 = np.append(test_results_parzen_class2, parzen_class2_accuracy)

print(test_results_ml_class1)
print(test_results_bl_class1)
print(test_results_parzen_class1)
print(test_results_ml_class2)
print(test_results_bl_class2)
print(test_results_parzen_class2)
print('\nML Accuracy before Diagonalization:')
x = PrettyTable()
x.field_names = ["Prd\\Tr", "Accuracy"]
x.add_row(["class 1", np.mean(test_results_ml_class1)])
x.add_row(["class 2", np.mean(test_results_ml_class2)])
print(x)

print('\nBL Accuracy before Diagonalization:')
x = PrettyTable()
x.field_names = ["Prd\\Tr", "Accuracy"]
x.add_row(["class 1", np.mean(test_results_bl_class1)])
x.add_row(["class 2", np.mean(test_results_bl_class2)])

print(x)

print('\nParzen Accuracy before Diagonalization:')
x = PrettyTable()
x.field_names = ["Prd\\Tr", "Accuracy"]
x.add_row(["class 1", np.mean(test_results_parzen_class1)])
x.add_row(["class 2", np.mean(test_results_parzen_class2)])

print(x)

########################################################

"""
(f): Repeat (a)-(d) for the same data after you have diagonalized it.  
"""

v1_training_points, v2_training_points, sigma_v1, sigma_v2, v1_mean, v2_mean = h.diagonalize_simultaneously(
    x1_training_points,
    x2_training_points, sigma_x1,
    sigma_x2, m1, m2)

# diagonalize testing points
v1_test_points, v2_test_points, _, _, _, _ = h.diagonalize_simultaneously(x1_test_points, x2_test_points, sigma_x1,
                                                                          sigma_x2, m1, m2)
################################################
# part(a)
h.plot_2d_graph(v1_training_points, v2_training_points, 1, 2, 'v1', 'v2', 'v1-v2')
h.plot_2d_graph(v1_training_points, v2_training_points, 1, 3, 'v1', 'v3', 'v1-v3')

################################################
# part(b)
# get estimated mean and covariance using ML method
x1_ml_estimated_mean = h.estimate_mean_ml(v1_training_points, number_of_points)
x1_ml_estimated_cov = h.estimate_cov_ml(v1_training_points, x1_ml_estimated_mean, number_of_points)
print('\nEstimated Mean1 using ML after Diagonalization:')
print(x1_ml_estimated_mean)
print('\nEstimated covariance1 using ML after Diagonalization:')
print(x1_ml_estimated_cov)

x2_ml_estimated_mean = h.estimate_mean_ml(v2_training_points, number_of_points)
x2_ml_estimated_cov = h.estimate_cov_ml(v2_training_points, x2_ml_estimated_mean, number_of_points)
print('\nEstimated Mean2 using ML after Diagonalization:')
print(x2_ml_estimated_mean)
print('\nEstimated covariance2 using ML after Diagonalization:')
print(x2_ml_estimated_cov)

# Estimating the means using BL

x1_bl_estimated_mean, x2_bl_estimated_mean = h.bl_expected_mean(v1_training_points, v2_training_points, sigma_v1,
                                                                sigma_v2, v1_mean, v2_mean, number_of_points)

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

    x1_ml_estimated_mean = h.estimate_mean_ml(v1_training_points, n)
    x1_ml_estimated_cov = h.estimate_cov_ml(v1_training_points, x1_ml_estimated_mean, n)
    w1_ml_mean_difference = np.append(w1_ml_mean_difference,
                                      np.absolute(np.linalg.norm(x1_ml_estimated_mean - v1_mean)))
    w1_ml_cov_difference = np.append(w1_ml_cov_difference, np.absolute(np.linalg.norm(x1_ml_estimated_cov - sigma_v1)))

    m1_0 = h.estimate_mean_bl(v1_training_points, m1_0, sigma0, sigma_v1, n)
    w1_mean_values = np.append(w1_mean_values, m1_0, axis=1)
    w1_plot_points = np.append(w1_plot_points, v1_training_points[:, [n]], axis=1)
    w1_mean_difference = np.append(w1_mean_difference, np.absolute(np.linalg.norm(m1_0 - v1_mean)))

    # for class 2
    w2_plot_points_index = np.append(w2_plot_points_index, n)

    x2_ml_estimated_mean = h.estimate_mean_ml(v2_training_points, n)
    x2_ml_estimated_cov = h.estimate_cov_ml(v2_training_points, x2_ml_estimated_mean, n)
    w2_ml_mean_difference = np.append(w2_ml_mean_difference,
                                      np.absolute(np.linalg.norm(x2_ml_estimated_mean - v2_mean)))
    w2_ml_cov_difference = np.append(w2_ml_cov_difference, np.absolute(np.linalg.norm(x2_ml_estimated_cov - sigma_v2)))

    m2_0 = h.estimate_mean_bl(v2_training_points, m2_0, sigma0, sigma_v2, n)
    w2_mean_values = np.append(w2_mean_values, m2_0, axis=1)
    w2_plot_points = np.append(w2_plot_points, v2_training_points[:, [n]], axis=1)
    w2_mean_difference = np.append(w2_mean_difference, np.absolute(np.linalg.norm(m2_0 - v2_mean)))

print('\nEstimated mean 1 using BL after Diagonalization:')
print(x1_bl_estimated_mean)
print('\nEstimated mean 2 using BL after Diagonalization:')
print(x2_bl_estimated_mean)

# PLOTTING #

# mean difference plots for ML
plt.plot(w1_plot_points_index, w1_ml_mean_difference, 'r-', label="class 1")
plt.plot(w2_plot_points_index, w2_ml_mean_difference, 'b-', label="class 2")

plt.xlabel('n')
plt.ylabel('Mean Diff')

plt.axis([0, 200, 0, 5])
plt.title('Mean Convergence using ML after Diagonalization')
plt.legend(loc=2)
plt.show()

# mean difference plots for ML
plt.plot(w1_plot_points_index, w1_ml_cov_difference, 'r-', label="class 1")
plt.plot(w2_plot_points_index, w2_ml_cov_difference, 'b-', label="class 2")

plt.xlabel('n')
plt.ylabel('Covariance Diff')

plt.axis([0, 200, 0, 200])
plt.title('Covariance Convergence using ML after Diagonalization')
plt.legend(loc=2)
plt.show()

# plots for BL
plt.plot(w1_plot_points_index, w1_mean_difference, 'r-', label="class 1")
plt.plot(w2_plot_points_index, w2_mean_difference, 'b-', label="class 2")

plt.xlabel('n')
plt.ylabel('Mean Diff')

plt.axis([0, 200, 0, 5])
plt.title('Mean Convergence using BL after Diagonalization')
plt.legend(loc=2)
plt.show()

################################################
# part(c)
kernel_covariance = 0.3
kernel_points = 100

step_size = 0.01

x1_parzen_estimated_mean = []
x1_parzen_estimated_covariance = []

x2_parzen_estimated_mean = []
x2_parzen_estimated_covariance = []

for i in range(0, 3, 1):
    # for class 1
    f_x1_points = []
    f_x1_values = []
    for j in np.arange(min(v1_training_points[i, :]) - 1, max(v1_training_points[i, :]), step_size):
        f_x1_points = np.append(f_x1_points, j)

    f_x1_points = np.sort(f_x1_points)

    for x in f_x1_points:
        f_x = 0.0
        for xi in v1_training_points[i, :]:
            f_x = f_x + h.kernel_function(x, xi, kernel_covariance)
        f_x = f_x / v1_training_points[i, :].size
        f_x1_values = np.append(f_x1_values, f_x)

    estimated_mean = 0.0
    for x in range(0, f_x1_points.size):
        estimated_mean = estimated_mean + h.parzen_expected_mean(f_x1_points[x], f_x1_values[x], step_size)
    x1_parzen_estimated_mean = np.append(x1_parzen_estimated_mean, estimated_mean)

    estimated_covariance = 0.0
    for x in range(0, f_x1_points.size):
        estimated_covariance = estimated_covariance + h.parzen_expected_covariance(f_x1_points[x], f_x1_values[x],
                                                                                   step_size, estimated_mean)
    x1_parzen_estimated_covariance = np.append(x1_parzen_estimated_covariance, estimated_covariance)

    # for class 2
    f_x2_points = []
    f_x2_values = []
    for j in np.arange(min(v2_training_points[i, :]) - 1, max(v2_training_points[i, :]), step_size):
        f_x2_points = np.append(f_x2_points, j)
    f_x2_points = np.sort(f_x2_points)

    for x in f_x2_points:
        f_x = 0.0
        for xi in v2_training_points[i, :]:
            f_x = f_x + h.kernel_function(x, xi, kernel_covariance)
        f_x = f_x / v2_training_points[i, :].size
        f_x2_values = np.append(f_x2_values, f_x)

    estimated_mean = 0.0
    for x in range(0, f_x2_points.size):
        estimated_mean = estimated_mean + h.parzen_expected_mean(f_x2_points[x], f_x2_values[x], step_size)
    x2_parzen_estimated_mean = np.append(x2_parzen_estimated_mean, estimated_mean)

    estimated_covariance = 0.0
    for x in range(0, f_x2_points.size):
        estimated_covariance = estimated_covariance + h.parzen_expected_covariance(f_x2_points[x], f_x2_values[x],
                                                                                   step_size, estimated_mean)
    x2_parzen_estimated_covariance = np.append(x2_parzen_estimated_covariance, estimated_covariance)

    min_x_axis = min(min(v1_training_points[i, :]), min(v2_training_points[i, :]))
    max_x_axis = max(max(v1_training_points[i, :]), max(v2_training_points[i, :]))

    title = 'f(x' + str(i + 1) + ')'

    # parzen window plots
    plt.plot(f_x1_points, f_x1_values, 'r--', label="class 1")
    plt.plot(f_x2_points, f_x2_values, 'b--', label="class 2")

    plt.xlabel('x values')
    plt.ylabel('f(x)')

    plt.xlim(min_x_axis, max_x_axis)
    plt.title(title)
    plt.legend(loc=2)
    plt.show()

print('\nEstimated mean 1 using Parzen after Diagonalization:')
x1_parzen_estimated_mean = np.array(x1_parzen_estimated_mean)[np.newaxis]
x1_parzen_estimated_mean = x1_parzen_estimated_mean.transpose()
print(x1_parzen_estimated_mean)
print('\nEstimated mean 2 using Parzen after Diagonalization:')
x2_parzen_estimated_mean = np.array(x2_parzen_estimated_mean)[np.newaxis]
x2_parzen_estimated_mean = x2_parzen_estimated_mean.transpose()
print(x2_parzen_estimated_mean)

print('\nEstimated covariance 1 using Parzen after Diagonalization:')
x1_parzen_estimated_covariance = np.diag(x1_parzen_estimated_covariance)
print(x1_parzen_estimated_covariance)
print('\nEstimated covariance 2 using Parzen after Diagonalization:')
x2_parzen_estimated_covariance = np.diag(x2_parzen_estimated_covariance)
print(x2_parzen_estimated_covariance)

################################################
# part(d)
# discriminant function for ML
h.calculate_disc_function_with_plot(x1_ml_estimated_mean, x2_ml_estimated_mean, x1_ml_estimated_cov,
                                    x2_ml_estimated_cov, v1_training_points, v2_training_points, p1, p2, 1, 2, 'ML')
h.calculate_disc_function_with_plot(x1_ml_estimated_mean, x2_ml_estimated_mean, x1_ml_estimated_cov,
                                    x2_ml_estimated_cov, v1_training_points, v2_training_points, p1, p2, 1, 3, 'ML')

# discriminant function for Bayes
h.calculate_disc_function_with_plot(x1_bl_estimated_mean, x2_bl_estimated_mean, sigma_v1,
                                    sigma_v2, v1_training_points, v2_training_points, p1, p2, 1, 2, 'BL')
h.calculate_disc_function_with_plot(x1_bl_estimated_mean, x2_bl_estimated_mean, sigma_v1,
                                    sigma_v2, v1_training_points, v2_training_points, p1, p2, 1, 3, 'BL')

# discriminant function for Parzen
h.calculate_disc_function_with_plot(x1_parzen_estimated_mean, x2_parzen_estimated_mean, x1_parzen_estimated_covariance,
                                    x2_parzen_estimated_covariance, v1_training_points, v2_training_points, p1, p2, 1,
                                    2, 'Parzen')
h.calculate_disc_function_with_plot(x1_parzen_estimated_mean, x2_ml_estimated_mean, x1_parzen_estimated_covariance,
                                    x2_parzen_estimated_covariance, v1_training_points, v2_training_points, p1, p2, 1,
                                    3, 'Parzen')

################################################
# part(e)
test_results_ml_class1 = []
test_results_ml_class2 = []

test_results_bl_class1 = []
test_results_bl_class2 = []

test_results_parzen_class1 = []
test_results_parzen_class2 = []

k = 10

class1_total_points = v1_training_points
class1_total_points = np.append(class1_total_points, v1_test_points, axis=1)

class2_total_points = v2_training_points
class2_total_points = np.append(class2_total_points, v2_test_points, axis=1)

print(class1_total_points[:, 399])
n = number_of_points + test_points_count
for i in range(0, k, 1):
    print('Cross:' + str(i + 1))
    number_of_testing_points = int(n / k)
    number_of_training_points = int(n - n / k)
    start = int(n * i / k)
    end = int((i + 1) * n / k)

    class1_test_points = class1_total_points[:, start: end]
    class1_train_points = class1_total_points[:, 0:start]
    class1_train_points = np.append(class1_train_points, class1_total_points[:, end:], axis=1)

    class2_test_points = class2_total_points[:, start: end]
    class2_train_points = class2_total_points[:, 0:start]
    class2_train_points = np.append(class2_train_points, class2_total_points[:, end:], axis=1)

    # estimated mean using ML
    x1_ml_estimated_mean = h.estimate_mean_ml(class1_train_points, number_of_training_points)
    x1_ml_estimated_cov = h.estimate_cov_ml(class1_train_points, x1_ml_estimated_mean, number_of_training_points)

    x2_ml_estimated_mean = h.estimate_mean_ml(class2_train_points, number_of_training_points)
    x2_ml_estimated_cov = h.estimate_cov_ml(class2_train_points, x2_ml_estimated_mean, number_of_training_points)

    # Estimating the means using BL
    x1_bl_estimated_mean, x2_bl_estimated_mean = h.bl_expected_mean(class1_train_points, class2_train_points, sigma_v1,
                                                                    sigma_v2, v1_mean, v2_mean,
                                                                    number_of_training_points)

    # estimated mean and cov using parzen window
    x1_parzen_estimated_mean, x1_parzen_estimated_covariance, x2_parzen_estimated_mean, x2_parzen_estimated_covariance = h.estimated_mean_parzen(
        class1_train_points, class2_train_points, kernel_covariance, step_size)

    ml_class1_accuracy, ml_class2_accuracy = h.test_classifier(class1_test_points, class2_test_points,
                                                               x1_ml_estimated_cov, x2_ml_estimated_cov,
                                                               x1_ml_estimated_mean, x2_ml_estimated_mean,
                                                               number_of_testing_points)
    test_results_ml_class1 = np.append(test_results_ml_class1, ml_class1_accuracy)
    test_results_ml_class2 = np.append(test_results_ml_class2, ml_class2_accuracy)

    bl_class1_accuracy, bl_class2_accuracy = h.test_classifier(class1_test_points, class2_test_points,
                                                               sigma_v1, sigma_v2,
                                                               x1_bl_estimated_mean, x2_bl_estimated_mean,
                                                               number_of_testing_points)
    test_results_bl_class1 = np.append(test_results_bl_class1, bl_class1_accuracy)
    test_results_bl_class2 = np.append(test_results_bl_class2, bl_class2_accuracy)

    parzen_class1_accuracy, parzen_class2_accuracy = h.test_classifier(class1_test_points, class2_test_points,
                                                                       x1_parzen_estimated_covariance,
                                                                       x2_parzen_estimated_covariance,
                                                                       x1_parzen_estimated_mean,
                                                                       x2_parzen_estimated_mean,
                                                                       number_of_testing_points)
    test_results_parzen_class1 = np.append(test_results_parzen_class1, parzen_class1_accuracy)
    test_results_parzen_class2 = np.append(test_results_parzen_class2, parzen_class2_accuracy)

print(test_results_ml_class1)
print(test_results_bl_class1)
print(test_results_parzen_class1)
print(test_results_ml_class2)
print(test_results_bl_class2)
print(test_results_parzen_class2)
print('\nML Accuracy After Diagonalization:')
x = PrettyTable()
x.field_names = ["Prd\\Tr", "Accuracy"]
x.add_row(["class 1", np.mean(test_results_ml_class1)])
x.add_row(["class 2", np.mean(test_results_ml_class2)])
print(x)

print('\nBL Accuracy after Diagonalization:')
x = PrettyTable()
x.field_names = ["Prd\\Tr", "Accuracy"]
x.add_row(["class 1", np.mean(test_results_bl_class1)])
x.add_row(["class 2", np.mean(test_results_bl_class2)])

print(x)

print('\nParzen Accuracy after Diagonalization:')
x = PrettyTable()
x.field_names = ["Prd\\Tr", "Accuracy"]
x.add_row(["class 1", np.mean(test_results_parzen_class1)])
x.add_row(["class 2", np.mean(test_results_parzen_class2)])

print(x)
