import numpy as np
import helper as h
import max_likelihood as ml
import bayesian_method as bl
import parzen_window as pz
import testing as ts
import plotting as plt
import os
from prettytable import PrettyTable
import csv

###############################################################################

# parameters
number_of_features = 6
number_of_classes = 2

kernel_cov = 0.0005
step_size = 0.001

p1 = 87 / 154
p2 = 1 - (87 / 154)

###############################################################################

# reading the data from the CSV files
my_path = os.path.abspath(os.path.dirname(__file__))

class1_csv_path = os.path.join(my_path, "data/class1.csv")
class2_csv_path = os.path.join(my_path, "data/class2.csv")

class1_data = np.genfromtxt(class1_csv_path, delimiter=',')
class2_data = np.genfromtxt(class2_csv_path, delimiter=',')

class1_data = class1_data.transpose()
class2_data = class2_data.transpose()

np.set_printoptions(suppress=True)

number_of_instances = len(class1_data[0]) + len(class2_data[0])

###############################################################################

# creating quadratic classifier

# estimate parameter for both classes

# using ML
class1_ml_est_mean = ml.estimate_mean_ml(class1_data, len(class1_data[0]))
class2_ml_est_mean = ml.estimate_mean_ml(class2_data, len(class2_data[0]))

class1_ml_est_cov = ml.estimate_cov_ml(class1_data, class1_ml_est_mean, len(class1_data[0]))
class2_ml_est_cov = ml.estimate_cov_ml(class2_data, class2_ml_est_mean, len(class2_data[0]))

# using BL
class1_bl_initial_mean = np.ones((len(class1_data), 1))
class1_bl_initial_cov = np.identity(len(class1_data))
class2_bl_initial_mean = np.ones((len(class2_data), 1))
class2_bl_initial_cov = np.identity(len(class2_data))

class1_bl_est_mean = bl.estimate_mean_bl(class1_data, class1_bl_initial_mean, class1_bl_initial_cov, class1_ml_est_cov,
                                         len(class1_data[0]))
class2_bl_est_mean = bl.estimate_mean_bl(class2_data, class2_bl_initial_mean, class2_bl_initial_cov, class2_ml_est_cov,
                                         len(class2_data[0]))

# using Parzen window

class1_parzen_est_mean, class1_parzen_est_cov, class2_parzen_est_mean, class2_parzen_est_cov = pz.estimated_mean_parzen(
    class1_data, class2_data, len(class1_data), kernel_cov, step_size)
# class1_parzen_est_mean, class1_parzen_est_cov, class2_parzen_est_mean, class2_parzen_est_cov = h.estimated_mean_parzen(
#     class1_data, class2_data, 1, kernel_cov, step_size)

# discriminant function for ML
# for i in range(1, 7):
#     for j in range(i, 7):
#         if i != j:
#             h.calculate_disc_function_with_plot(class1_ml_est_mean, class2_ml_est_mean, class1_ml_est_cov,
#                                                 class2_ml_est_cov, class1_data, class2_data, p1, p2, i, j, 'ML')


plt.plot_disc_func(class1_ml_est_mean, class2_ml_est_mean, class1_ml_est_cov,
                   class2_ml_est_cov, class1_data, class2_data, p1, p2, 1, 2, 'ML')
plt.plot_disc_func(class1_ml_est_mean, class2_ml_est_mean, class1_ml_est_cov,
                   class2_ml_est_cov, class1_data, class2_data, p1, p2, 1, 3, 'ML')

# discriminant function for Bayes
plt.plot_disc_func(class1_bl_est_mean, class2_bl_est_mean, class1_ml_est_cov,
                   class2_ml_est_cov, class1_data, class2_data, p1, p2, 1, 2, 'BL')
plt.plot_disc_func(class1_bl_est_mean, class2_bl_est_mean, class1_ml_est_cov,
                   class2_ml_est_cov, class1_data, class2_data, p1, p2, 1, 3, 'BL')

# discriminant function for Parzen
plt.plot_disc_func(class1_parzen_est_mean, class2_parzen_est_mean, class1_parzen_est_cov,
                   class2_parzen_est_cov, class1_data, class2_data, p1, p2, 1,
                   2, 'Parzen')
plt.plot_disc_func(class1_parzen_est_mean, class2_parzen_est_mean, class1_parzen_est_cov,
                   class2_parzen_est_cov, class1_data, class2_data, p1, p2, 1,
                   3, 'Parzen')

test_results_ml_class1 = []
test_results_ml_class2 = []

test_results_bl_class1 = []
test_results_bl_class2 = []

k = 5
n = 76
for i in range(0, k, 1):
    print('Cross:' + str(i + 1))
    number_of_testing_points = int(n / k)
    number_of_training_points = int(n - n / k)
    start = int(n * i / k)
    end = int((i + 1) * n / k)

    print("start:", start, "\tend:", end)
    class1_test_points = class1_data[:, start: end]
    class1_train_points = class1_data[:, 0:start]
    class1_train_points = np.append(class1_train_points, class1_data[:, end:], axis=1)

    class2_test_points = class2_data[:, start: end]
    class2_train_points = class2_data[:, 0:start]
    class2_train_points = np.append(class2_train_points, class2_data[:, end:], axis=1)

    # estimated mean using ML
    x1_ml_estimated_mean = ml.estimate_mean_ml(class1_train_points, len(class1_train_points[0]))
    x1_ml_estimated_cov = ml.estimate_cov_ml(class1_train_points, x1_ml_estimated_mean, number_of_training_points)

    x2_ml_estimated_mean = ml.estimate_mean_ml(class2_train_points, len(class2_train_points[0]))
    x2_ml_estimated_cov = ml.estimate_cov_ml(class2_train_points, x2_ml_estimated_mean, number_of_training_points)

    # # Estimating the means using BL

    # x1_bl_estimated_mean, x2_bl_estimated_mean = bl.bl_expected_mean(class1_train_points, class2_train_points,
    #                                                                  class1_ml_est_cov,
    #                                                                  class2_ml_est_cov, class1_ml_est_mean,
    #                                                                  class2_ml_est_mean, number_of_training_points)

    # # estimated mean and cov using parzen window
    # x1_parzen_estimated_mean, x1_parzen_estimated_covariance, x2_parzen_estimated_mean, x2_parzen_estimated_covariance = h.estimated_mean_parzen(
    #     class1_train_points, class2_train_points, kernel_covariance, step_size)

    ml_class1_accuracy, ml_class2_accuracy = ts.test_classifier(class1_test_points, class2_test_points,
                                                                x1_ml_estimated_cov, x2_ml_estimated_cov,
                                                                x1_ml_estimated_mean, x2_ml_estimated_mean,
                                                                number_of_testing_points, p1, p2)
    print(ml_class1_accuracy, ml_class2_accuracy)
    test_results_ml_class1 = np.append(test_results_ml_class1, ml_class1_accuracy)
    test_results_ml_class2 = np.append(test_results_ml_class2, ml_class2_accuracy)

    # bl_class1_accuracy, bl_class2_accuracy = ts.test_classifier(class1_test_points, class2_test_points,
    #                                                             x1_ml_estimated_cov, x2_ml_estimated_cov,
    #                                                             x1_bl_estimated_mean, x2_bl_estimated_mean,
    #                                                             number_of_testing_points)
    # test_results_bl_class1 = np.append(test_results_bl_class1, bl_class1_accuracy)
    # test_results_bl_class2 = np.append(test_results_bl_class2, bl_class2_accuracy)

    # parzen_class1_accuracy, parzen_class2_accuracy = h.test_classifier(class1_test_points, class2_test_points,
    #                                                                    x1_parzen_estimated_covariance,
    #                                                                    x2_parzen_estimated_covariance,
    #                                                                    x1_parzen_estimated_mean,
    #                                                                    x2_parzen_estimated_mean,
    #                                                                    number_of_testing_points)
    # test_results_parzen_class1 = np.append(test_results_parzen_class1, parzen_class1_accuracy)
    # test_results_parzen_class2 = np.append(test_results_parzen_class2, parzen_class2_accuracy)

print('\nML Accuracy Before Diagonalization:')
x = PrettyTable()
x.field_names = ["Prd\\Tr", "Accuracy"]
x.add_row(["class 1", np.mean(test_results_ml_class1)])
x.add_row(["class 2", np.mean(test_results_ml_class2)])
print(x)

# print('\nBL Accuracy Before Diagonalization:')
# x = PrettyTable()
# x.field_names = ["Prd\\Tr", "Accuracy"]
# x.add_row(["class 1", np.mean(test_results_bl_class1)])
# x.add_row(["class 2", np.mean(test_results_bl_class2)])
#
# print(x)

# ----------------------------------------------------------------------

print(len(class1_data))
print(len(class2_data))
print('Class 1 mean using ML:')
print(class1_ml_est_mean)
print('\nClass 1 mean using BL:')
print(class1_bl_est_mean)
print('\nClass 2 mean using ML:')
print(class2_ml_est_mean)
print('\nClass 2 mean using BL:')
print(class2_bl_est_mean)
print('\nClass 1 mean using Parzen:')
print(class1_parzen_est_mean)
print('\nClass 2 mean using Parzen:')
print(class2_parzen_est_mean)
print('\nClass 1 cov using ML:')
print(class1_ml_est_cov)
print('\nClass 2 cov using ML:')
print(class2_ml_est_cov)
print('\nClass 1 cov using Parzen:')
print(class1_parzen_est_cov)
print('\nClass 2 cov using Parzen:')
print(class2_parzen_est_cov)
