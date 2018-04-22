import numpy as np
import helper as h
import max_likelihood as ml
import bayesian_method as bl
import parzen_window as pz
import testing as ts
import plotting as plt
import fishers_disc as fd
import ho_kashyab as hk
import os
from prettytable import PrettyTable
import matplotlib.pyplot as mat_plt

###############################################################################

# parameters
features_count = 6
classes_count = 2

kernel_cov = 0.1
step_size = 0.01

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

# normalize the data
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

# estimate parameter for both classes
class1_ml_est_mean = ml.estimate_mean_ml(class1_data, class1_instances_count)
class2_ml_est_mean = ml.estimate_mean_ml(class2_data, class2_instances_count)

class1_ml_est_cov = ml.estimate_cov_ml(class1_data, class1_ml_est_mean, class1_instances_count)
class2_ml_est_cov = ml.estimate_cov_ml(class2_data, class2_ml_est_mean, class2_instances_count)

# print("\nBEFORE DIAGONALIZING:")
# print('\nClass 1 mean using ML:')
# print(class1_ml_est_mean)
# print('\nClass 2 mean using ML:')
# print(class2_ml_est_mean)
# print('\nClass 1 cov using ML:')
# print(class1_ml_est_cov)
# print('\nClass 2 cov using ML:')
# print(class2_ml_est_cov)

# discriminant function for ML
plt.plot_disc_func(class1_ml_est_mean, class2_ml_est_mean, class1_ml_est_cov,
                   class2_ml_est_cov, class1_data, class2_data, p1, p2, 1, 2, 'ML')

# test the classifier
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

# class1_data_diag, class2_data_diag = h.normalize_data(class1_data_diag, class2_data_diag, 0.0, 1.0)

class1_ml_est_mean_diag = ml.estimate_mean_ml(class1_data_diag, class1_instances_count)
class2_ml_est_mean_diag = ml.estimate_mean_ml(class2_data_diag, class2_instances_count)

class1_ml_est_cov_diag = ml.estimate_cov_ml(class1_data_diag, class1_ml_est_mean_diag, class1_instances_count)
class2_ml_est_cov_diag = ml.estimate_cov_ml(class2_data_diag, class2_ml_est_mean_diag, class2_instances_count)

plt.plot_disc_func(class1_ml_est_mean_diag, class2_ml_est_mean_diag, class1_ml_est_cov_diag,
                   class2_ml_est_cov_diag, class1_data_diag, class2_data_diag, p1, p2, 1, 2, 'ML After')

test_results_ml_class1_diag, test_results_ml_class2_diag = ts.ml_k_cross_validation(class1_data_diag, class2_data_diag,
                                                                                    p1, p2, k,
                                                                                    class1_instances_count,
                                                                                    class2_instances_count)

print('\nML Accuracy After Diagonalization:')
x = PrettyTable()
x.field_names = ["Prd\\Tr", "Accuracy"]
x.add_row(["class 1", np.mean(test_results_ml_class1_diag)])
x.add_row(["class 2", np.mean(test_results_ml_class2_diag)])
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

# test using 5-cross validation
test_results_bl_class1, test_results_bl_class2 = ts.bl_k_cross_validation(class1_data_diag, class2_data_diag, p1, p2, k,
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

# after diagonalizing
class1_data_diag_bl, class2_data_diag_bl, _, _, _, _ = h.diagonalize_simultaneously(
    class1_data,
    class2_data, class1_ml_est_cov,
    class2_ml_est_cov, class1_bl_est_mean, class2_bl_est_mean)

# class1_data_diag, class2_data_diag = h.normalize_data(class1_data_diag, class2_data_diag, 0.0, 1.0)

# class1_ml_est_mean_diag = ml.estimate_mean_ml(class1_data_diag, class1_instances_count)
# class2_ml_est_mean_diag = ml.estimate_mean_ml(class2_data_diag, class2_instances_count)

class1_ml_est_cov_diag = ml.estimate_cov_ml(class1_data_diag_bl, class1_ml_est_mean_diag, class1_instances_count)
class2_ml_est_cov_diag = ml.estimate_cov_ml(class2_data_diag_bl, class2_ml_est_mean_diag, class2_instances_count)

# estimate mean for Class 1
class1_bl_est_mean_diag = bl.estimate_mean_bl(class1_data_diag_bl, class1_bl_initial_mean, class1_bl_initial_cov,
                                              class1_ml_est_cov_diag,
                                              class1_instances_count)
# estimate mean for Class 2
class2_bl_est_mean_diag = bl.estimate_mean_bl(class2_data_diag, class2_bl_initial_mean, class2_bl_initial_cov,
                                              class2_ml_est_cov_diag,
                                              class2_instances_count)

# discriminant function for Bayes
plt.plot_disc_func(class1_bl_est_mean_diag, class2_bl_est_mean_diag, class1_ml_est_cov_diag,
                   class2_ml_est_cov_diag, class1_data_diag_bl, class2_data_diag, p1, p2, 1, 2, 'BL After')

# test using 5-cross validation
test_results_bl_class1_diag, test_results_bl_class2_diag = ts.bl_k_cross_validation(class1_data_diag_bl,
                                                                                    class2_data_diag,
                                                                                    p1, p2, k,
                                                                                    class1_instances_count,
                                                                                    class2_instances_count)

# print test results
print('BL AFTER DIAGONALIZING:')
print('\nBL Accuracy After Diagonalization:')
x = PrettyTable()
x.field_names = ["Prd\\Tr", "Accuracy"]
x.add_row(["class 1", np.mean(test_results_bl_class1_diag)])
x.add_row(["class 2", np.mean(test_results_bl_class2_diag)])
print(x)

print('\nClass 1 mean using BL:')
print(class1_bl_est_mean_diag)
print('\nClass 2 mean using BL:')
print(class2_bl_est_mean_diag)

# ------------------------------------------------------------------------------------

# using Parzen window
class1_parzen_est_mean, class1_parzen_est_cov, class2_parzen_est_mean, class2_parzen_est_cov = pz.estimated_mean_parzen(
    class1_data, class2_data, len(class1_data), kernel_cov, step_size)
# class1_parzen_est_mean, class1_parzen_est_cov, class2_parzen_est_mean, class2_parzen_est_cov = pz.estimated_mean_parzen(
#     class1_data, class2_data, 1, kernel_cov, step_size)

# discriminant function for Parzen
# plt.plot_disc_func(class1_parzen_est_mean, class2_parzen_est_mean, class1_parzen_est_cov,
#                    class2_parzen_est_cov, class1_data, class2_data, p1, p2, 1,
#                    2, 'Parzen')

print('\nClass 1 mean using Parzen:')
print(class1_parzen_est_mean)
print('\nClass 2 mean using Parzen:')
print(class2_parzen_est_mean)
print('\nClass 1 cov using Parzen:')
print(class1_parzen_est_cov)
print('\nClass 2 cov using Parzen:')
print(class2_parzen_est_cov)

# after diagonalizing


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

test_results_knn_class1_diag, test_results_knn_class2_diag = ts.knn_k_cross_validation(class1_data_diag,
                                                                                       class2_data_diag, k,
                                                                                       class1_instances_count,
                                                                                       class2_instances_count, k_nn)

# print test results
print('\nK-NN Accuracy Diagonalized:')
x = PrettyTable()
x.field_names = ["Prd\\Tr", "Accuracy"]
x.add_row(["class 1", np.mean(test_results_knn_class1_diag)])
x.add_row(["class 2", np.mean(test_results_knn_class2_diag)])
print(x)

############################################################################################################
############################################################################################################
"""
Create the Fisher Disc Classifier
"""

# before diagonalizing
# calculate W ( representation for the plane)
s_w = class1_ml_est_cov + class2_ml_est_cov
w = np.array(np.linalg.inv(s_w) @ (class1_ml_est_mean - class2_ml_est_mean))

fd_mean1 = w.transpose() @ class1_ml_est_mean
fd_mean2 = w.transpose() @ class2_ml_est_mean

fd_cov1 = w.transpose() @ class1_ml_est_cov @ w
fd_cov2 = w.transpose() @ class2_ml_est_cov @ w

a, b, c = fd.disc_root(fd_mean1, fd_mean2, fd_cov1, fd_cov2, p1, p2)

fd.plot_fd(class1_data, class2_data, w, a, b, c)

test_results_fd_class1, test_results_fd_class2 = ts.fd_k_cross_validation(class1_data, class2_data, k,
                                                                          class1_instances_count,
                                                                          class2_instances_count, w, p1, p2)

title = 'After diagonalizing'
mat_plt.plot(class1_data_diag[0], class1_data_diag[1], 'b.', label="Class 1", alpha=0.5)
mat_plt.plot(class2_data_diag[0], class2_data_diag[1], 'r.', label="Class 2", alpha=0.5)
mat_plt.xlabel('x1')
mat_plt.ylabel('x2')
mat_plt.title(title)
mat_plt.legend(loc=2)
mat_plt.grid(linestyle='dotted')
mat_plt.show()

# print test results
print('\nFisher\'s Disc. Accuracy:')
x = PrettyTable()
x.field_names = ["Prd\\Tr", "Accuracy"]
x.add_row(["class 1", np.mean(test_results_fd_class1)])
x.add_row(["class 2", np.mean(test_results_fd_class2)])
print(x)

# AFTER DIAGONALIZING
# calculate W ( representation for the plane)
s_w_diag = class1_ml_est_cov_diag + class2_ml_est_cov_diag
w_diag = np.array(np.linalg.inv(s_w_diag) @ (class1_ml_est_mean_diag - class2_ml_est_mean_diag))

fd_mean1_diag = w_diag.transpose() @ class1_ml_est_mean_diag
fd_mean2_diag = w_diag.transpose() @ class2_ml_est_mean_diag

fd_cov1_diag = w_diag.transpose() @ class1_ml_est_cov_diag @ w
fd_cov2_diag = w_diag.transpose() @ class2_ml_est_cov_diag @ w

a, b, c = fd.disc_root(fd_mean1_diag, fd_mean2_diag, fd_cov1_diag, fd_cov2_diag, p1, p2)

fd.plot_fd(class1_data_diag, class2_data_diag, w_diag, a, b, c)

test_results_fd_class1_diag, test_results_fd_class2_diag = ts.fd_k_cross_validation(class1_data_diag, class2_data_diag,
                                                                                    k,
                                                                                    class1_instances_count,
                                                                                    class2_instances_count, w_diag, p1,
                                                                                    p2)

# print test results
print('\nFisher\'s Disc. Accuracy after diagonalizing:')
x = PrettyTable()
x.field_names = ["Prd\\Tr", "Accuracy"]
x.add_row(["class 1", np.mean(test_results_fd_class1_diag)])
x.add_row(["class 2", np.mean(test_results_fd_class2_diag)])
print(x)

############################################################################################################
############################################################################################################
"""
Create the Ho-Kashyap Classifier
"""

test_results_hk_class1, test_results_hk_class2 = ts.hk_k_cross_validation(class1_data, class2_data,
                                                                          k,
                                                                          class1_instances_count,
                                                                          class2_instances_count)

# print test results
print('\nHo-Kashyap Accuracy before diagonalizing:')
x = PrettyTable()
x.field_names = ["Prd\\Tr", "Accuracy"]
x.add_row(["class 1", np.mean(test_results_hk_class1)])
x.add_row(["class 2", np.mean(test_results_hk_class2)])
print(x)

print('AFTER DIAGON.: ')
test_results_hk_class1_diag, test_results_hk_class2_diag = ts.hk_k_cross_validation(class1_data_diag, class2_data_diag,
                                                                                    k,
                                                                                    class1_instances_count,
                                                                                    class2_instances_count)

# print test results
print('\nHo-Kashyap Accuracy After diagonalizing:')
x = PrettyTable()
x.field_names = ["Prd\\Tr", "Accuracy"]
x.add_row(["class 1", np.mean(test_results_hk_class1_diag)])
x.add_row(["class 2", np.mean(test_results_hk_class2_diag)])
print(x)
