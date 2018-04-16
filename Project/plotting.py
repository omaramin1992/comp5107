import numpy as np
import matplotlib.pyplot as plt


def plot_disc_func(m1, m2, cov1, cov2, x1_points, x2_points, p1, p2, d1, d2, method):
    a = ((np.linalg.inv(cov2) - np.linalg.inv(cov1)) / 2)
    b = np.array(m1.transpose() @ np.linalg.inv(
        cov1) - m2.transpose() @ np.linalg.inv(cov2))
    c = np.math.log(p1 / p2) + np.log(np.linalg.det(cov2) / np.linalg.det(cov1))

    equation_points = []
    roots_1 = []
    roots_2 = []

    min_w = min(min(min(x1_points[d1 - 1, :]), min(x2_points[d1 - 1, :])),
                min(min(x1_points[d2 - 1, :]), min(x2_points[d2 - 1, :])))
    max_w = max(max(max(x1_points[d1 - 1, :]), max(x2_points[d1 - 1, :])),
                max(max(x1_points[d2 - 1, :]), max(x2_points[d2 - 1, :])))

    for x1 in np.arange(min_w - 1, max_w + 1, 0.01):
        equation_points.append(x1)
        x2_square_coefficient = a[d2 - 1][d2 - 1]
        x2_coefficient = (a[d1 - 1][d2 - 1] * x1) + (a[d2 - 1][d1 - 1] * x1) + b[d1 - 1][d2 - 1]
        constant = a[d1 - 1][d1 - 1] * np.math.pow(x1, 2) + b[d1 - 1][d1 - 1] * x1 + c

        poly_coefficients = [x2_square_coefficient, x2_coefficient, constant]
        roots = np.roots(poly_coefficients)
        roots_1.append(roots[0])
        roots_2.append(roots[1])

    plt.plot(x1_points[d1 - 1, :], x1_points[d2 - 1, :], 'b.', label="Class 1")
    plt.plot(x2_points[d1 - 1, :], x2_points[d2 - 1, :], 'r.', label="Class 2")
    plt.plot(equation_points, roots_2, 'g--', label="Dis.Fnc.")
    plt.plot(equation_points, roots_1, 'y--', label="Dis.Fnc.")
    plt.xlabel('x' + str(d1))
    plt.ylabel('x' + str(d2))

    plt.axis([min_w - 1, max_w + 1, min_w - 1, max_w + 1])
    plt.title('Dis. Fun. for ' + method + ' for x' + str(d1) + '-x' + str(d2))
    plt.legend(loc=2)
    plt.show()
