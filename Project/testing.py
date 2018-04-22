import helper as h
import numpy as np
import max_likelihood as ml
import bayesian_method as bl
import parzen_window as pz
import ho_kashyab as hk
import k_nn as kn
import fishers_disc as fd


def test_classifier(class1_test_points, class2_test_points, x1_ml_estimated_cov, x2_ml_estimated_cov,
                    x1_ml_estimated_mean, x2_ml_estimated_mean, class1_testing_points_count,
                    class2_testing_points_count, p1, p2):
    # classification results
    class1_true = 0.0
    class1_false = 0.0

    class2_true = 0.0
    class2_false = 0.0
    # print(class1_test_points[:, 1])
    # print(class1_testing_points_count)

    # classify each point
    for j in range(class1_testing_points_count):
        discriminant_value = h.calculate_discriminant(class1_test_points[:, j], x1_ml_estimated_cov,
                                                      x2_ml_estimated_cov, x1_ml_estimated_mean, x2_ml_estimated_mean,
                                                      p1,
                                                      p2)
        # print("class1 Disc Val: ", discriminant_value)
        if discriminant_value > 0:
            class1_true += 1
        else:
            class1_false += 1

    for j in range(class2_testing_points_count):
        discriminant_value = h.calculate_discriminant(class2_test_points[:, j], x1_ml_estimated_cov,
                                                      x2_ml_estimated_cov, x1_ml_estimated_mean, x2_ml_estimated_mean,
                                                      p1,
                                                      p2)
        # print("class2 Disc Val: ", discriminant_value)
        if discriminant_value < 0:
            class2_true += 1
        else:
            class2_false += 1

    class1_accuracy = (class1_true / len(class1_test_points[0])) * 100
    class2_accuracy = (class2_true / len(class2_test_points[0])) * 100

    total_accuracy = (class1_true + class2_true) * 100 / (len(class1_test_points[0]) + len(class2_test_points[0]))

    # print(class1_true, class1_false)
    # print(class2_true, class2_false)
    print(total_accuracy)
    return class1_accuracy, class2_accuracy, total_accuracy


def ml_k_cross_validation(class1_data, class2_data, p1, p2, k, n1, n2):
    test_results_ml_class1 = []
    test_results_ml_class2 = []

    accuracies = []

    for i in range(0, k, 1):
        print('Cross:' + str(i + 1))
        class1_testing_points_count = int(n1 / k)
        class1_training_points_count = int(n1 - n1 / k)
        class1_start = int(n1 * i / k)
        class1_end = int((i + 1) * n1 / k)

        class2_testing_points_count = int(n2 / k)
        class2_training_points_count = int(n2 - n2 / k)
        class2_start = int(n2 * i / k)
        class2_end = int((i + 1) * n2 / k)

        # print("start:", class1_start, "\tend:", class1_end)
        # print("start:", class2_start, "\tend:", class2_end)

        class1_test_points = class1_data[:, class1_start: class1_end]
        class1_train_points = class1_data[:, 0:class1_start]
        class1_train_points = np.append(class1_train_points, class1_data[:, class1_end:], axis=1)

        class2_test_points = class2_data[:, class2_start: class2_end]
        class2_train_points = class2_data[:, 0:class2_start]
        class2_train_points = np.append(class2_train_points, class2_data[:, class2_end:], axis=1)

        # estimated mean and cov using ML
        x1_ml_estimated_mean = ml.estimate_mean_ml(class1_train_points, len(class1_train_points[0]))
        x1_ml_estimated_cov = ml.estimate_cov_ml(class1_train_points, x1_ml_estimated_mean,
                                                 class1_training_points_count)

        x2_ml_estimated_mean = ml.estimate_mean_ml(class2_train_points, len(class2_train_points[0]))
        x2_ml_estimated_cov = ml.estimate_cov_ml(class2_train_points, x2_ml_estimated_mean,
                                                 class2_training_points_count)

        ml_class1_accuracy, ml_class2_accuracy, total_accuracy = test_classifier(class1_test_points, class2_test_points,
                                                                                 x1_ml_estimated_cov,
                                                                                 x2_ml_estimated_cov,
                                                                                 x1_ml_estimated_mean,
                                                                                 x2_ml_estimated_mean,
                                                                                 class1_testing_points_count,
                                                                                 class2_testing_points_count, p1, p2)
        # print(ml_class1_accuracy, ml_class2_accuracy)
        test_results_ml_class1 = np.append(test_results_ml_class1, ml_class1_accuracy)
        test_results_ml_class2 = np.append(test_results_ml_class2, ml_class2_accuracy)

        accuracies = np.append(accuracies, total_accuracy)

    print('\nML Average Accuracy:', np.mean(accuracies))
    return test_results_ml_class1, test_results_ml_class2


def bl_k_cross_validation(class1_data, class2_data, p1, p2, k, n1, n2):
    test_results_bl_class1 = []
    test_results_bl_class2 = []

    accuracies = []

    for i in range(0, k, 1):
        print('Cross:' + str(i + 1))
        class1_testing_points_count = int(n1 / k)
        class1_training_points_count = int(n1 - n1 / k)
        class1_start = int(n1 * i / k)
        class1_end = int((i + 1) * n1 / k)

        class2_testing_points_count = int(n2 / k)
        class2_training_points_count = int(n2 - n2 / k)
        class2_start = int(n2 * i / k)
        class2_end = int((i + 1) * n2 / k)

        # print("start:", class1_start, "\tend:", class1_end)
        # print("start:", class2_start, "\tend:", class2_end)

        class1_test_points = class1_data[:, class1_start: class1_end]
        class1_train_points = class1_data[:, 0:class1_start]
        class1_train_points = np.append(class1_train_points, class1_data[:, class1_end:], axis=1)

        class2_test_points = class2_data[:, class2_start: class2_end]
        class2_train_points = class2_data[:, 0:class2_start]
        class2_train_points = np.append(class2_train_points, class2_data[:, class2_end:], axis=1)

        class1_ml_est_mean = ml.estimate_mean_ml(class1_train_points, len(class1_train_points[0]))
        class1_ml_est_cov = ml.estimate_cov_ml(class1_train_points, class1_ml_est_mean,
                                               class1_training_points_count)

        class2_ml_est_mean = ml.estimate_mean_ml(class2_train_points, len(class2_train_points[0]))
        class2_ml_est_cov = ml.estimate_cov_ml(class2_train_points, class2_ml_est_mean,
                                               class2_training_points_count)

        # Estimating the means using BL
        class1_bl_initial_mean = np.ones((len(class1_data), 1))
        class1_bl_initial_cov = np.identity(len(class1_data))

        class2_bl_initial_mean = np.ones((len(class2_data), 1))
        class2_bl_initial_cov = np.identity(len(class2_data))

        class1_bl_est_mean = bl.estimate_mean_bl(class1_train_points, class1_bl_initial_mean, class1_bl_initial_cov,
                                                 class1_ml_est_cov, len(class1_train_points[0]))
        class2_bl_est_mean = bl.estimate_mean_bl(class2_train_points, class2_bl_initial_mean, class2_bl_initial_cov,
                                                 class2_ml_est_cov, len(class2_train_points[0]))

        bl_class1_accuracy, bl_class2_accuracy, total_accuracy = test_classifier(class1_test_points, class2_test_points,
                                                                 class1_ml_est_cov, class2_ml_est_cov,
                                                                 class1_bl_est_mean, class2_bl_est_mean,
                                                                 class1_testing_points_count,
                                                                 class2_testing_points_count, p1, p2)
        # print(bl_class1_accuracy, bl_class2_accuracy)
        test_results_bl_class1 = np.append(test_results_bl_class1, bl_class1_accuracy)
        test_results_bl_class2 = np.append(test_results_bl_class2, bl_class2_accuracy)

        accuracies = np.append(accuracies, total_accuracy)

    print('\nBL Average Accuracy:', np.mean(accuracies))
    return test_results_bl_class1, test_results_bl_class2


# def parzen_k_cross_validation(class1_data, class2_data, p1, p2, k, n1, n2, kernel_cov, step_size):
#     test_results_parzen_class1 = []
#     test_results_parzen_class2 = []
#
#     accuracies = []
#
#     for i in range(0, k, 1):
#         print('Cross:' + str(i + 1))
#         class1_testing_points_count = int(n1 / k)
#         class1_training_points_count = int(n1 - n1 / k)
#         class1_start = int(n1 * i / k)
#         class1_end = int((i + 1) * n1 / k)
#
#         class2_testing_points_count = int(n2 / k)
#         class2_training_points_count = int(n2 - n2 / k)
#         class2_start = int(n2 * i / k)
#         class2_end = int((i + 1) * n2 / k)
#
#         # print("start:", class1_start, "\tend:", class1_end)
#         # print("start:", class2_start, "\tend:", class2_end)
#
#         class1_test_points = class1_data[:, class1_start: class1_end]
#         class1_train_points = class1_data[:, 0:class1_start]
#         class1_train_points = np.append(class1_train_points, class1_data[:, class1_end:], axis=1)
#
#         class2_test_points = class2_data[:, class2_start: class2_end]
#         class2_train_points = class2_data[:, 0:class2_start]
#         class2_train_points = np.append(class2_train_points, class2_data[:, class2_end:], axis=1)
#
#         # estimated mean and cov using parzen window
#         x1_parzen_estimated_mean, x1_parzen_estimated_covariance, x2_parzen_estimated_mean, x2_parzen_estimated_covariance = h.estimated_mean_parzen(
#             class1_train_points, class2_train_points, kernel_cov, step_size)
#
#         class1_parzen_est_mean, class1_parzen_est_cov, class2_parzen_est_mean, class2_parzen_est_cov = pz.estimated_mean_parzen(
#             class1_data, class2_data, len(class1_data), kernel_cov, step_size)
#
#         parzen_class1_accuracy, parzen_class2_accuracy = test_classifier(class1_test_points, class2_test_points,
#                                                                          x1_parzen_estimated_covariance,
#                                                                          x2_parzen_estimated_covariance,
#                                                                          x1_parzen_estimated_mean,
#                                                                          x2_parzen_estimated_mean,
#                                                                          class1_testing_points_count,
#                                                                          class2_testing_points_count, p1, p2)
#         test_results_parzen_class1 = np.append(test_results_parzen_class1, parzen_class1_accuracy)
#         test_results_parzen_class2 = np.append(test_results_parzen_class2, parzen_class2_accuracy)
#
#     return test_results_parzen_class1, test_results_parzen_class2


def knn_k_cross_validation(class1_data, class2_data, k, n1, n2, k_nn):
    test_results_knn_class1 = []
    test_results_knn_class2 = []

    accuracies = []

    for i in range(0, k, 1):
        print('Cross:' + str(i + 1))
        class1_testing_points_count = int(n1 / k)
        class1_training_points_count = int(n1 - n1 / k)
        class1_start = int(n1 * i / k)
        class1_end = int((i + 1) * n1 / k)

        class2_testing_points_count = int(n2 / k)
        class2_training_points_count = int(n2 - n2 / k)
        class2_start = int(n2 * i / k)
        class2_end = int((i + 1) * n2 / k)

        class1_test_points = class1_data[:, class1_start: class1_end]
        class1_train_points = class1_data[:, 0:class1_start]
        class1_train_points = np.append(class1_train_points, class1_data[:, class1_end:], axis=1)

        class2_test_points = class2_data[:, class2_start: class2_end]
        class2_train_points = class2_data[:, 0:class2_start]
        class2_train_points = np.append(class2_train_points, class2_data[:, class2_end:], axis=1)

        class1_test_points = np.array(class1_test_points).transpose()
        class2_test_points = np.array(class2_test_points).transpose()

        class1_true = 0
        class1_false = 0

        class2_true = 0
        class2_false = 0

        for x in class1_test_points:
            classification = kn.get_neighbors(x, class1_train_points, class2_train_points, k_nn)
            if classification == 1:
                class1_true = class1_true + 1
            else:
                class1_false = class1_false + 1

        for x in class2_test_points:
            classification = kn.get_neighbors(x, class1_train_points, class2_train_points, k_nn)
            if classification == 2:
                class2_true = class2_true + 1
            else:
                class2_false = class2_false + 1

        class1_accuracy = (class1_true / len(class1_test_points)) * 100
        class2_accuracy = (class2_true / len(class2_test_points)) * 100

        test_results_knn_class1 = np.append(test_results_knn_class1, class1_accuracy)
        test_results_knn_class2 = np.append(test_results_knn_class2, class2_accuracy)

        accuracy = (class1_true + class2_true) * 100 / (len(class1_test_points) + len(class2_test_points))
        accuracies = np.append(accuracies, accuracy)
        # print(class1_testing_points_count, class2_testing_points_count)
        #
        # print(class1_true, class1_false)
        # print(class2_true, class2_false)
        print(accuracy)
    print('\nK-NN Average Accuracy:', np.mean(accuracies))
    return test_results_knn_class1, test_results_knn_class2


def fd_k_cross_validation(class1_data, class2_data, k, n1, n2, w, p1, p2):
    test_results_fd_class1 = []
    test_results_fd_class2 = []

    accuracies = []

    for i in range(0, k, 1):
        print('Cross:' + str(i + 1))
        class1_testing_points_count = int(n1 / k)
        class1_training_points_count = int(n1 - n1 / k)
        class1_start = int(n1 * i / k)
        class1_end = int((i + 1) * n1 / k)

        class2_testing_points_count = int(n2 / k)
        class2_training_points_count = int(n2 - n2 / k)
        class2_start = int(n2 * i / k)
        class2_end = int((i + 1) * n2 / k)

        class1_test_points = class1_data[:, class1_start: class1_end]
        class1_train_points = class1_data[:, 0:class1_start]
        class1_train_points = np.append(class1_train_points, class1_data[:, class1_end:], axis=1)

        class2_test_points = class2_data[:, class2_start: class2_end]
        class2_train_points = class2_data[:, 0:class2_start]
        class2_train_points = np.append(class2_train_points, class2_data[:, class2_end:], axis=1)

        class1_ml_est_mean = ml.estimate_mean_ml(class1_train_points, len(class1_train_points[0]))
        class1_ml_est_cov = ml.estimate_cov_ml(class1_train_points, class1_ml_est_mean,
                                               class1_training_points_count)

        class2_ml_est_mean = ml.estimate_mean_ml(class2_train_points, len(class2_train_points[0]))
        class2_ml_est_cov = ml.estimate_cov_ml(class2_train_points, class2_ml_est_mean,
                                               class2_training_points_count)

        fd_mean1 = w.transpose() @ class1_ml_est_mean
        fd_mean2 = w.transpose() @ class2_ml_est_mean

        fd_cov1 = w.transpose() @ class1_ml_est_cov @ w
        fd_cov2 = w.transpose() @ class2_ml_est_cov @ w

        class1_test_points = np.array(class1_test_points).transpose()
        class2_test_points = np.array(class2_test_points).transpose()

        class1_true = 0
        class1_false = 0

        class2_true = 0
        class2_false = 0

        for x in class1_test_points:
            x_test = w.transpose() @ x
            classification = fd.classify(x_test, fd_mean1, fd_mean2, fd_cov1, fd_cov2, p1, p2)
            if classification == 1:
                class1_true = class1_true + 1
            else:
                class1_false = class1_false + 1

        for x in class2_test_points:
            x_test = w.transpose() @ x
            classification = fd.classify(x_test, fd_mean1, fd_mean2, fd_cov1, fd_cov2, p1, p2)
            if classification == 2:
                class2_true = class2_true + 1
            else:
                class2_false = class2_false + 1

        class1_accuracy = (class1_true / len(class1_test_points)) * 100
        class2_accuracy = (class2_true / len(class2_test_points)) * 100

        test_results_fd_class1 = np.append(test_results_fd_class1, class1_accuracy)
        test_results_fd_class2 = np.append(test_results_fd_class2, class2_accuracy)

        accuracy = (class1_true + class2_true) * 100 / (len(class1_test_points) + len(class2_test_points))
        accuracies = np.append(accuracies, accuracy)
        print(accuracy)

    print('\nFisher\'s Disc. Average Accuracy:', np.mean(accuracies))
    return test_results_fd_class1, test_results_fd_class2


def hk_k_cross_validation(class1_data, class2_data, k, n1, n2):
    test_results_hk_class1 = []
    test_results_hk_class2 = []

    accuracies = []

    for i in range(0, k, 1):
        print('Cross:' + str(i + 1))
        class1_testing_points_count = int(n1 / k)
        class1_training_points_count = int(n1 - n1 / k)
        class1_start = int(n1 * i / k)
        class1_end = int((i + 1) * n1 / k)

        class2_testing_points_count = int(n2 / k)
        class2_training_points_count = int(n2 - n2 / k)
        class2_start = int(n2 * i / k)
        class2_end = int((i + 1) * n2 / k)

        class1_test_points = class1_data[:, class1_start: class1_end]
        class1_train_points = class1_data[:, 0:class1_start]
        class1_train_points = np.append(class1_train_points, class1_data[:, class1_end:], axis=1)

        class2_test_points = class2_data[:, class2_start: class2_end]
        class2_train_points = class2_data[:, 0:class2_start]
        class2_train_points = np.append(class2_train_points, class2_data[:, class2_end:], axis=1)

        # class1_ml_est_mean = ml.estimate_mean_ml(class1_train_points, len(class1_train_points[0]))
        # class1_ml_est_cov = ml.estimate_cov_ml(class1_train_points, class1_ml_est_mean,
        #                                        class1_training_points_count)
        #
        # class2_ml_est_mean = ml.estimate_mean_ml(class2_train_points, len(class2_train_points[0]))
        # class2_ml_est_cov = ml.estimate_cov_ml(class2_train_points, class2_ml_est_mean,
        #                                        class2_training_points_count)
        #
        # fd_mean1 = w.transpose() @ class1_ml_est_mean
        # fd_mean2 = w.transpose() @ class2_ml_est_mean
        #
        # fd_cov1 = w.transpose() @ class1_ml_est_cov @ w
        # fd_cov2 = w.transpose() @ class2_ml_est_cov @ w

        a, b = hk.ho_kashyap(class1_train_points, class2_train_points)

        class1_ones = np.ones(len(class1_test_points[0]))
        class2_ones = np.ones(len(class2_test_points[0]))
        print('Adding ones:')
        class1_test_points = np.insert(class1_test_points, 0, class1_ones, axis=0)
        class2_test_points = np.insert(class2_test_points, 0, class2_ones, axis=0)
        print('Done')
        class1_test_points = np.array(class1_test_points).transpose()
        class2_test_points = -1*np.array(class2_test_points).transpose()

        class1_true = 0
        class1_false = 0

        class2_true = 0
        class2_false = 0

        for x in class1_test_points:
            classification = a.transpose() @ x
            if classification > 0:
                class1_true = class1_true + 1
            else:
                class1_false = class1_false + 1

        for x in class2_test_points:
            classification = a.transpose() @ x
            if classification < 0:
                class2_true = class2_true + 1
            else:
                class2_false = class2_false + 1

        class1_accuracy = (class1_true / len(class1_test_points)) * 100
        class2_accuracy = (class2_true / len(class2_test_points)) * 100

        test_results_hk_class1 = np.append(test_results_hk_class1, class1_accuracy)
        test_results_hk_class2 = np.append(test_results_hk_class2, class2_accuracy)

        accuracy = (class1_true + class2_true) * 100 / (len(class1_test_points) + len(class2_test_points))
        accuracies = np.append(accuracies, accuracy)
        print(accuracy)

    print('\nHo-Kashyap Average Accuracy:', np.mean(accuracies))
    return test_results_hk_class1, test_results_hk_class2
