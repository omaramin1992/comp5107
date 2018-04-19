import numpy as np
import helper as h
import max_likelihood as ml
import bayesian_method as bl
import parzen_window as pz
import k_nn as kn
import testing as ts
import plotting as plt
import os
from sklearn import preprocessing
from prettytable import PrettyTable
import matplotlib.pyplot as mat_plt

###############################################################################

# parameters
features_count = 6
classes_count = 2

kernel_cov = 0.005
step_size = 0.001

k = 5
k_nn = 5
###############################################################################

# m1 = np.array([[3],
#                [1],
#                [4]])
#
# m2 = np.array([[-3],
#                [1],
#                [-4]])
#
# # parameters of covariance matrices
# a1 = 2
# b1 = 3
# c1 = 4
#
# alpha = 0.1
# beta = 0.2
#
# number_of_points = 300
#
#
# # creating the covariance matrices with the parameters
# sigma_x1, sigma_x2 = h.covariance_matrix(a1, b1, c1, alpha, beta)
# print('Sigma x1:')
# print(sigma_x1)
# print('\nSigma x2:')
# print(sigma_x2)
#
# # eigenvalues and eigenvectors respectively
# w_x1, v_x1 = np.linalg.eig(sigma_x1)
# lambda_x1 = np.diag(w_x1)
#
# w_x2, v_x2 = np.linalg.eig(sigma_x2)
# lambda_x2 = np.diag(w_x2)
# print('\neigenvalues of x1:')
# print(lambda_x1)
# print('\neigenvalues of x2:')
# print(lambda_x2)
#
# # create point matrices for the two classes X1 and X2
# z1_training_points, class1_data = h.generate_point_matrix(v_x1, lambda_x1, m1, number_of_points)
# z2_training_points, class2_data = h.generate_point_matrix(v_x2, lambda_x2, m2, number_of_points)
# class1_instances_count = len(class1_data[0])
# class2_instances_count = len(class2_data[0])

###############################################################################

# reading the data from the CSV files
my_path = os.path.abspath(os.path.dirname(__file__))

# class1_csv_path = os.path.join(my_path, "data/class1.csv")
# class2_csv_path = os.path.join(my_path, "data/class2.csv")

class1_csv_path = os.path.join(my_path, "data/window-new.data")
class2_csv_path = os.path.join(my_path, "data/non-window-new.data")

class1_data = np.genfromtxt(class1_csv_path, delimiter=',')
class2_data = np.genfromtxt(class2_csv_path, delimiter=',')

p1 = len(class1_data) / (len(class1_data) + len(class2_data))
p2 = 1 - (len(class1_data) / (len(class1_data) + len(class2_data)))

# p1 = 0.5
# p2 = 0.5

class1_instances_count = len(class1_data)
class2_instances_count = len(class2_data)

# print(len(class1_data[0]), len(class2_data), (len(class1_data) + len(class2_data)))

# # normalize the data

class1_data = class1_data.transpose()
class2_data = class2_data.transpose()

mat_plt.plot(class1_data[0], class1_data[1], 'b.', label="Class 1", alpha=0.5)
mat_plt.plot(class2_data[0], class2_data[1], 'r.', label="Class 1", alpha=0.5)
mat_plt.xlabel('x1')
mat_plt.ylabel('x2')
mat_plt.title('Before normalizing')
mat_plt.legend(loc=2)
mat_plt.grid(linestyle='dotted')
mat_plt.show()

class1_data, class2_data = h.normalize_data(class1_data, class2_data, 0.0, 1.0)
# full_data = np.append(class1_data, class2_data, axis=1)
# std_scale = preprocessing.StandardScaler().fit(full_data)
# class1_data = std_scale.transform(class1_data)
# class2_data = std_scale.transform(class2_data)


# print(class1_data)
# print(class2_data)
mat_plt.plot(class1_data[0], class1_data[1], 'b.', label="Class 1", alpha=0.5)
mat_plt.plot(class2_data[0], class2_data[1], 'r.', label="Class 1", alpha=0.5)
mat_plt.xlabel('x1')
mat_plt.ylabel('x2')
mat_plt.title('After normalizing')
mat_plt.legend(loc=2)
mat_plt.grid(linestyle='dotted')
mat_plt.show()

np.set_printoptions(suppress=True)

instances_count = class1_instances_count + class2_instances_count

###############################################################################
###############################################################################
"""
Create the ML Classifier
"""

# # normalize the data
# class1_data = h.normalize_data(class1_data)
# class2_data = h.normalize_data(class2_data)

# print(class1_data)
# print(class2_data)
# creating quadratic classifier

# estimate parameter for both classes
class1_ml_est_mean = ml.estimate_mean_ml(class1_data, class1_instances_count)
class2_ml_est_mean = ml.estimate_mean_ml(class2_data, class2_instances_count)

class1_ml_est_cov = ml.estimate_cov_ml(class1_data, class1_ml_est_mean, class1_instances_count)
class2_ml_est_cov = ml.estimate_cov_ml(class2_data, class2_ml_est_mean, class2_instances_count)

print("\nBEFORE DIAGONALIZING:")
print('\nClass 1 mean using ML:')
print(class1_ml_est_mean)
print('\nClass 2 mean using ML:')
print(class2_ml_est_mean)
print('\nClass 1 cov using ML:')
print(class1_ml_est_cov)
print('\nClass 2 cov using ML:')
print(class2_ml_est_cov)

# discriminant function for ML
# for i in range(1, 7):
#     for j in range(i, 7):
#         if i != j:
#             plt.plot_disc_func(class1_ml_est_mean, class2_ml_est_mean, class1_ml_est_cov,
#                                                 class2_ml_est_cov, class1_data, class2_data, p1, p2, i, j, 'ML')


plt.plot_disc_func(class1_ml_est_mean, class2_ml_est_mean, class1_ml_est_cov,
                   class2_ml_est_cov, class1_data, class2_data, p1, p2, 1, 2, 'ML')
plt.plot_disc_func(class1_ml_est_mean, class2_ml_est_mean, class1_ml_est_cov,
                   class2_ml_est_cov, class1_data, class2_data, p1, p2, 1, 3, 'ML')

test_results_ml_class1, test_results_ml_class2 = ts.ml_k_cross_validation(class1_data, class2_data, p1, p2, k,
                                                                          class1_instances_count,
                                                                          class2_instances_count)

print('\nML Accuracy Before Diagonalization:')
x = PrettyTable()
x.field_names = ["Prd\\Tr", "Accuracy"]
x.add_row(["class 1", np.mean(test_results_ml_class1)])
x.add_row(["class 2", np.mean(test_results_ml_class2)])
print(x)

# after diagonalizing
class1_data_diag, class2_data_diag, class1_diag_cov, class2_diag_cov, class1_diag_mean, class2_diag_mean = h.diagonalize_simultaneously(
    class1_data,
    class2_data, class1_ml_est_cov,
    class2_ml_est_cov, class1_ml_est_mean, class2_ml_est_mean)

class1_ml_est_mean_diag = ml.estimate_mean_ml(class1_data_diag, class1_instances_count)
class2_ml_est_mean_diag = ml.estimate_mean_ml(class2_data_diag, class2_instances_count)

class1_ml_est_cov_diag = ml.estimate_cov_ml(class1_data_diag, class1_ml_est_mean_diag, class1_instances_count)
class2_ml_est_cov_diag = ml.estimate_cov_ml(class2_data_diag, class2_ml_est_mean_diag, class2_instances_count)

plt.plot_disc_func(class1_ml_est_mean_diag, class2_ml_est_mean_diag, class1_ml_est_cov_diag,
                   class2_ml_est_cov_diag, class1_data_diag, class2_data_diag, p1, p2, 1, 2, 'ML After')
plt.plot_disc_func(class1_ml_est_mean_diag, class2_ml_est_mean_diag, class1_ml_est_cov_diag,
                   class2_ml_est_cov_diag, class1_data_diag, class2_data_diag, p1, p2, 1, 3, 'ML After')
# plt.plot_disc_func(class1_ml_est_mean_diag, class2_ml_est_mean_diag, class1_ml_est_cov_diag,
#                    class2_ml_est_cov_diag, class1_data_diag, class2_data_diag, p1, p2, 2, 3, 'ML After')

test_results_ml_class1, test_results_ml_class2 = ts.ml_k_cross_validation(class1_data_diag, class2_data_diag, p1, p2, k,
                                                                          class1_instances_count,
                                                                          class2_instances_count)

# test_results_ml_class1 = []
# test_results_ml_class2 = []

# test_results_bl_class1 = []
# test_results_bl_class2 = []


# n1 = class1_instances_count
# n2 = class2_instances_count
# for i in range(0, k, 1):
#     print('Cross:' + str(i + 1))
#     class1_testing_points_count = int(n1 / k)
#     class1_training_points_count = int(n1 - n1 / k)
#     class1_start = int(n1 * i / k)
#     class1_end = int((i + 1) * n1 / k)
#
#     class2_testing_points_count = int(n2 / k)
#     class2_training_points_count = int(n2 - n2 / k)
#     class2_start = int(n2 * i / k)
#     class2_end = int((i + 1) * n2 / k)
#
#     print("start:", class1_start, "\tend:", class1_end)
#     print("start:", class2_start, "\tend:", class2_end)
#
#     class1_test_points = class1_data[:, class1_start: class1_end]
#     class1_train_points = class1_data[:, 0:class1_start]
#     class1_train_points = np.append(class1_train_points, class1_data[:, class1_end:], axis=1)
#
#     class2_test_points = class2_data[:, class2_start: class2_end]
#     class2_train_points = class2_data[:, 0:class2_start]
#     class2_train_points = np.append(class2_train_points, class2_data[:, class2_end:], axis=1)
#
#     # estimated mean and cov using ML
#     x1_ml_estimated_mean = ml.estimate_mean_ml(class1_train_points, len(class1_train_points[0]))
#     x1_ml_estimated_cov = ml.estimate_cov_ml(class1_train_points, x1_ml_estimated_mean, class1_training_points_count)
#
#     x2_ml_estimated_mean = ml.estimate_mean_ml(class2_train_points, len(class2_train_points[0]))
#     x2_ml_estimated_cov = ml.estimate_cov_ml(class2_train_points, x2_ml_estimated_mean, class2_training_points_count)
#
#     # # Estimating the means using BL
#
#     # x1_bl_estimated_mean, x2_bl_estimated_mean = bl.bl_expected_mean(class1_train_points, class2_train_points,
#     #                                                                  class1_ml_est_cov,
#     #                                                                  class2_ml_est_cov, class1_ml_est_mean,
#     #                                                                  class2_ml_est_mean, number_of_training_points)
#
#     # # estimated mean and cov using parzen window
#     # x1_parzen_estimated_mean, x1_parzen_estimated_covariance, x2_parzen_estimated_mean, x2_parzen_estimated_covariance = h.estimated_mean_parzen(
#     #     class1_train_points, class2_train_points, kernel_covariance, step_size)
#
#     ml_class1_accuracy, ml_class2_accuracy = ts.test_classifier(class1_test_points, class2_test_points,
#                                                                 x1_ml_estimated_cov, x2_ml_estimated_cov,
#                                                                 x1_ml_estimated_mean, x2_ml_estimated_mean,
#                                                                 class1_testing_points_count,
#                                                                 class2_testing_points_count, p1, p2)
#     print(ml_class1_accuracy, ml_class2_accuracy)
#     test_results_ml_class1 = np.append(test_results_ml_class1, ml_class1_accuracy)
#     test_results_ml_class2 = np.append(test_results_ml_class2, ml_class2_accuracy)
#
#     # bl_class1_accuracy, bl_class2_accuracy = ts.test_classifier(class1_test_points, class2_test_points,
#     #                                                             x1_ml_estimated_cov, x2_ml_estimated_cov,
#     #                                                             x1_bl_estimated_mean, x2_bl_estimated_mean,
#     #                                                             number_of_testing_points)
#     # test_results_bl_class1 = np.append(test_results_bl_class1, bl_class1_accuracy)
#     # test_results_bl_class2 = np.append(test_results_bl_class2, bl_class2_accuracy)
#
#     # parzen_class1_accuracy, parzen_class2_accuracy = h.test_classifier(class1_test_points, class2_test_points,
#     #                                                                    x1_parzen_estimated_covariance,
#     #                                                                    x2_parzen_estimated_covariance,
#     #                                                                    x1_parzen_estimated_mean,
#     #                                                                    x2_parzen_estimated_mean,
#     #                                                                    number_of_testing_points)
#     # test_results_parzen_class1 = np.append(test_results_parzen_class1, parzen_class1_accuracy)
#     # test_results_parzen_class2 = np.append(test_results_parzen_class2, parzen_class2_accuracy)

print('\nML Accuracy After Diagonalization:')
x = PrettyTable()
x.field_names = ["Prd\\Tr", "Accuracy"]
x.add_row(["class 1", np.mean(test_results_ml_class1)])
x.add_row(["class 2", np.mean(test_results_ml_class2)])
print(x)

print("\nBEFORE DIAGONALIZING:")
print('\nClass 1 mean using ML:')
print(class1_ml_est_mean)
print('\nClass 2 mean using ML:')
print(class2_ml_est_mean)
print('\nClass 1 cov using ML:')
print(class1_ml_est_cov)
print('\nClass 2 cov using ML:')
print(class2_ml_est_cov)

print("\nAFTER DIAGONALIZING:")
print('\nClass 1 mean using ML:')
print(class1_ml_est_mean_diag)
print('\nClass 2 mean using ML:')
print(class2_ml_est_mean_diag)
print('\nClass 1 cov using ML:')
print(class1_ml_est_cov_diag)
print('\nClass 2 cov using ML:')
print(class2_ml_est_cov_diag)

# ------------------------------------------------------------------------------------
"""
Create the BL Classifier
"""

# initial mean and cov for class 1
class1_bl_initial_mean = np.ones((len(class1_data), 1))
class1_bl_initial_cov = np.identity(len(class1_data))

# initial mean and cov for class 2
class2_bl_initial_mean = np.ones((len(class2_data), 1))
class2_bl_initial_cov = np.identity(len(class2_data))

# estimate mean for Class 1
class1_bl_est_mean = bl.estimate_mean_bl(class1_data, class1_bl_initial_mean, class1_bl_initial_cov, class1_ml_est_cov,
                                         class1_instances_count)
# estimate mean for Class 2
class2_bl_est_mean = bl.estimate_mean_bl(class2_data, class2_bl_initial_mean, class2_bl_initial_cov, class2_ml_est_cov,
                                         class2_instances_count)

# discriminant function for Bayes
plt.plot_disc_func(class1_bl_est_mean, class2_bl_est_mean, class1_ml_est_cov,
                   class2_ml_est_cov, class1_data, class2_data, p1, p2, 1, 2, 'BL')
plt.plot_disc_func(class1_bl_est_mean, class2_bl_est_mean, class1_ml_est_cov,
                   class2_ml_est_cov, class1_data, class2_data, p1, p2, 1, 3, 'BL')

# test using 5-cross validation
test_results_bl_class1, test_results_bl_class2 = ts.bl_k_cross_validation(class1_data_diag, class2_data_diag, p1, p2, k,
                                                                          class1_instances_count,
                                                                          class2_instances_count)

# # after diagonalizing
# class1_data_diag, class2_data_diag, class1_diag_cov, class2_diag_cov, class1_diag_mean, class2_diag_mean = h.diagonalize_simultaneously(
#     class1_data,
#     class2_data, class1_ml_est_cov,
#     class2_ml_est_cov, class1_ml_est_mean, class2_ml_est_mean)

class1_ml_est_mean_diag = ml.estimate_mean_ml(class1_data_diag, class1_instances_count)
class2_ml_est_mean_diag = ml.estimate_mean_ml(class2_data_diag, class2_instances_count)

class1_ml_est_cov_diag = ml.estimate_cov_ml(class1_data_diag, class1_ml_est_mean_diag, class1_instances_count)
class2_ml_est_cov_diag = ml.estimate_cov_ml(class2_data_diag, class2_ml_est_mean_diag, class2_instances_count)

plt.plot_disc_func(class1_ml_est_mean_diag, class2_ml_est_mean_diag, class1_ml_est_cov_diag,
                   class2_ml_est_cov_diag, class1_data_diag, class2_data_diag, p1, p2, 1, 2, 'ML After')
plt.plot_disc_func(class1_ml_est_mean_diag, class2_ml_est_mean_diag, class1_ml_est_cov_diag,
                   class2_ml_est_cov_diag, class1_data_diag, class2_data_diag, p1, p2, 1, 3, 'ML After')

test_results_ml_class1, test_results_ml_class2 = ts.ml_k_cross_validation(class1_data_diag, class2_data_diag, p1, p2, k,
                                                                          class1_instances_count,
                                                                          class2_instances_count)

# print test results
print('\nBL Accuracy Before Diagonalization:')
x = PrettyTable()
x.field_names = ["Prd\\Tr", "Accuracy"]
x.add_row(["class 1", np.mean(test_results_bl_class1)])
x.add_row(["class 2", np.mean(test_results_bl_class2)])
print(x)

print('\nClass 1 mean using BL:')
print(class1_bl_est_mean)
print('\nClass 2 mean using BL:')
print(class2_bl_est_mean)

# ------------------------------------------------------------------------------------

# using Parzen window

class1_parzen_est_mean, class1_parzen_est_cov, class2_parzen_est_mean, class2_parzen_est_cov = pz.estimated_mean_parzen(
    class1_data, class2_data, len(class1_data), kernel_cov, step_size)
# class1_parzen_est_mean, class1_parzen_est_cov, class2_parzen_est_mean, class2_parzen_est_cov = pz.estimated_mean_parzen(
#     class1_data, class2_data, 1, kernel_cov, step_size)

# discriminant function for Parzen
plt.plot_disc_func(class1_parzen_est_mean, class2_parzen_est_mean, class1_parzen_est_cov,
                   class2_parzen_est_cov, class1_data, class2_data, p1, p2, 1,
                   2, 'Parzen')
plt.plot_disc_func(class1_parzen_est_mean, class2_parzen_est_mean, class1_parzen_est_cov,
                   class2_parzen_est_cov, class1_data, class2_data, p1, p2, 1,
                   3, 'Parzen')

print('\nClass 1 mean using Parzen:')
print(class1_parzen_est_mean)
print('\nClass 2 mean using Parzen:')
print(class2_parzen_est_mean)
print('\nClass 1 cov using Parzen:')
print(class1_parzen_est_cov)
print('\nClass 2 cov using Parzen:')
print(class2_parzen_est_cov)

# test_results_parzen_class1, test_results_parzen_class2 = ts.ml_k_cross_validation(class1_data, class2_data, p1, p2, k,
#                                                                           class1_instances_count,
#                                                                           class2_instances_count)

############################################################################################################
############################################################################################################
"""
Create the k-NN Classifier
"""

test_results_knn_class1, test_results_knn_class2 = ts.knn_k_cross_validation(class1_data, class2_data, k,
                                                                             class1_instances_count,
                                                                             class2_instances_count, k_nn)

# print test results
print('\nK-NN Accuracy:')
x = PrettyTable()
x.field_names = ["Prd\\Tr", "Accuracy"]
x.add_row(["class 1", np.mean(test_results_knn_class1)])
x.add_row(["class 2", np.mean(test_results_knn_class2)])
print(x)


############################################################################################################
############################################################################################################
"""
Create the Fisher Disc Classifier
"""

# calculate W ( representation for the plane)
s_w = class1_ml_est_cov + class2_ml_est_cov
w = np.array(np.linalg.inv(s_w)@(class1_ml_est_mean - class2_ml_est_mean))

fd_mean1 = w.transpose() @ class1_ml_est_mean
fd_mean2 = w.transpose() @ class2_ml_est_mean

fd_cov1 = w.transpose() @ class1_ml_est_cov @ w
fd_cov2 = w.transpose() @ class2_ml_est_cov @ w


############################################################################################################
############################################################################################################
"""
Create the Ho-Kashyap Classifier
"""

