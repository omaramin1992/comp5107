import numpy as np
import math
import matplotlib.pyplot as plt


def classify(x, m1, m2, cov1, cov2, p1, p2):
    cov1_sqrt = math.sqrt(cov1)
    cov2_sqrt = math.sqrt(cov2)

    k1 = np.math.log(p1 / p2) + np.math.log(cov2_sqrt / cov1_sqrt)

    a = math.pow(cov1, 2) - math.pow(cov2, 2)
    b = 2 * m1 * math.pow(cov2, 2) - 2 * m2 * math.pow(cov1, 2)
    c = math.pow(m2 * cov1, 2) - math.pow(m1 * cov2, 2) + (2 * cov1 * cov2) * k1

    value = a * math.pow(x, 2) + b * x + c

    if value > 0:
        return 1
    else:
        return 2


def disc_root(m1, m2, cov1, cov2, p1, p2):
    cov1_sqrt = math.sqrt(cov1)
    cov2_sqrt = math.sqrt(cov2)

    k1 = np.math.log(p1 / p2) + np.math.log(cov2_sqrt / cov1_sqrt)

    a = math.pow(cov1, 2) - math.pow(cov2, 2)
    b = 2 * m1 * math.pow(cov2, 2) - 2 * m2 * math.pow(cov1, 2)
    c = math.pow(m2 * cov1, 2) - math.pow(m1 * cov2, 2) + (2 * cov1 * cov2) * k1

    return a, b, c


def plot_fd(class1_data, class2_data, w, a, b, c):
    class1_data = np.array(class1_data).transpose()
    class2_data = np.array(class2_data).transpose()

    class1_points = []
    class2_points = []

    for x in class1_data:
        x_point = w.transpose() @ x.transpose()
        class1_points = np.append(class1_points, x_point)

    for x in class2_data:
        x_point = w.transpose() @ x.transpose()
        class2_points = np.append(class2_points, x_point)

    class1_ones = np.zeros(len(class1_points))
    class2_ones = np.zeros(len(class2_points))

    poly_coefficients = [a, b, c]
    roots = np.roots(poly_coefficients)
    title = 'Fishers Disc'
    plt.figure()
    plt.plot(class1_points, class1_ones, 'b.', label="Class 1", alpha=0.5)
    plt.plot(class2_points, class2_ones, 'r.', label="Class 2", alpha=0.5)
    # plt.plot([roots[0], roots[0]], [0, 0.02], 'g-', label="Disc", alpha=1)
    plt.plot([roots[1], roots[1]], [0, 0.02], 'g--', label="Disc", alpha=1)

    # plt.xlim()
    plt.title(title)
    plt.legend(loc=2)
    plt.grid(linestyle='dotted')
    plt.show()
    plt.close()

    return None
