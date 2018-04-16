import helper as h


def test_classifier(class1_test_points, class2_test_points, x1_ml_estimated_cov, x2_ml_estimated_cov,
                    x1_ml_estimated_mean, x2_ml_estimated_mean, number_of_testing_points, p1, p2):
    # classification results
    class1_true = 0.0
    class1_false = 0.0

    class2_true = 0.0
    class2_false = 0.0
    print(class1_test_points[:, 1])
    print(number_of_testing_points)
    # classify each point
    for j in range(number_of_testing_points - 1):
        discriminant_value = h.calculate_discriminant(class1_test_points[:, j], x1_ml_estimated_cov,
                                                      x2_ml_estimated_cov, x1_ml_estimated_mean, x2_ml_estimated_mean,
                                                      p1,
                                                      p2)
        if discriminant_value > 0:
            class1_true += 1
        else:
            class1_false += 1

    for j in range(number_of_testing_points - 1):
        discriminant_value = h.calculate_discriminant(class2_test_points[:, j], x1_ml_estimated_cov,
                                                      x2_ml_estimated_cov, x1_ml_estimated_mean, x2_ml_estimated_mean,
                                                      p1,
                                                      p2)
        if discriminant_value < 0:
            class2_true += 1
        else:
            class2_false += 1

    class1_accuracy = (class1_true / number_of_testing_points) * 100
    class2_accuracy = (class2_true / number_of_testing_points) * 100

    print(class1_true, class1_false)
    print(class2_true, class2_false)
    return class1_accuracy, class2_accuracy
