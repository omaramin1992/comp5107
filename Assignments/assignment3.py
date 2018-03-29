import numpy as np
import helper as h
from prettytable import PrettyTable
import matplotlib.pyplot as plt


########################################################################################
########################################################################################
'''
Consider the two-class pattern recognition problem in which the class conditional distributions are both
normally distributed with arbitrary means M1 and M2, and covariance matrices Sigma1 and Sigma2 respectively.
Assume that you are working in a 3-D space (for example, as in Assignment II) and that the covariance
matrices are not equal. Here, you must assume that the means can be submitted as input parameters (and
are not constant vectors) so that the classes can be made closer or more distant.
'''
# mean values for the two classes
m1 = np.array([[11],
               [-10],
               [15]])

m2 = np.array([[-11],
               [10],
               [-15]])
# m1 = np.array([[10],
#                [2],
#                [7]])
#
# m2 = np.array([[3],
#                [-5],
#                [1]])
m1 = np.array([[3],
               [1],
               [4]])

m2 = np.array([[-3],
               [1],
               [-4]])
########################################################################################
########################################################################################

# parameters of covariance matrices
a1 = 2
b1 = 3
c1 = 4

alpha = 0.1
beta = 0.2

number_of_points = 2000

p1 = 0.5
p2 = 0.5

########################################################################################
########################################################################################


'''
Q(a):
Generate 200 points of each distribution before diagonalization and plot them in the (x1– x2) and (x1– x3) domains.
'''
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
z1_matrix, x1_matrix = h.generate_point_matrix(v_x1, lambda_x1, m1, number_of_points)
z2_matrix, x2_matrix = h.generate_point_matrix(v_x2, lambda_x2, m2, number_of_points)

# PLOTTING #
# X WORLD
# plot the first class as blue for (d1 - d2) domain and second class as red
h.plot_2d_graph(x1_matrix, x2_matrix, 1, 2, 'x1', 'x2', 'x1-x2')

# plot the first class as blue for (d1 - d3) domain and second class as red
h.plot_2d_graph(x1_matrix, x2_matrix, 1, 3, 'x1', 'x3', 'x1-x3')

# # plot the first class as blue for (d2 - d3) domain and second class as red
# h.plot_2d_graph(x1_matrix, x2_matrix, 2, 3, 'x2', 'x3', 'x2-x3')

# # 3D plot for the points before diagonalizing
# h.plot_3d_graph(x1_matrix, x2_matrix, 'x1', 'x2', 'x3', 'X-world')

########################################################################################
########################################################################################
'''
Q(b):
Assuming that you know the means and covariance matrices of the two classes, compute the optimal Bayes
discriminant function, and plot it for in the (x1– x2) and (x1– x3) domains.
'''

a = ((np.linalg.inv(sigma_x2) - np.linalg.inv(sigma_x1)) / 2)
b = np.array(m1.transpose() @ np.linalg.inv(sigma_x1) - m2.transpose() @ np.linalg.inv(sigma_x2))
c = np.math.log(p1 / p2) + np.log(np.linalg.det(sigma_x2) / np.linalg.det(sigma_x1))

equation_points = []
roots_1 = []
roots_2 = []

min_w = min(min(min(x1_matrix[0, :]), min(x2_matrix[0, :])),
            min(min(x1_matrix[1, :]), min(x2_matrix[1, :])))
max_w = max(max(max(x1_matrix[0, :]), max(x2_matrix[0, :])),
            max(max(x1_matrix[1, :]), max(x2_matrix[1, :])))

# get the roots for the discriminant function
for x1 in range(-15, 10, 1):
    equation_points.append(x1)
    x2_square_coefficient = a[1][1]
    x2_coefficient = (a[0][1] * x1) + (a[1][0] * x1) + b[0][1]
    constant = a[0][0] * np.math.pow(x1, 2) + b[0][0] * x1 + c

    poly_coefficients = [x2_square_coefficient, x2_coefficient, constant]
    roots = np.roots(poly_coefficients)
    roots_1.append(roots[0])
    roots_2.append(roots[1])

plt.plot(x1_matrix[0, :], x1_matrix[1, :], 'b.', label="Class 1")
plt.plot(x2_matrix[0, :], x2_matrix[1, :], 'r.', label="Class 2")
plt.plot(equation_points, roots_2, 'g-', label="Dis.Fnc.")
plt.plot(equation_points, roots_1, 'y-', label="Dis.Fnc.")
plt.xlabel('x1')
plt.ylabel('x2')

plt.axis([min_w - 1, max_w + 1, min_w - 1, max_w + 1])
plt.title('x1-x2')
plt.legend(loc=2)
plt.show()

'''
vghjkghjghjkgj
'''
equation_points = []
roots_1 = []
roots_2 = []

min_w = min(min(min(x1_matrix[0, :]), min(x2_matrix[0, :])),
            min(min(x1_matrix[2, :]), min(x2_matrix[2, :])))
max_w = max(max(max(x1_matrix[0, :]), max(x2_matrix[0, :])),
            max(max(x1_matrix[2, :]), max(x2_matrix[2, :])))

for x1 in range(-15, 15, 1):
    equation_points.append(x1)
    x2_square_coefficient = a[2][2]
    x2_coefficient = (a[0][2] * x1) + (a[2][0] * x1) + b[0][2]
    constant = a[0][0] * np.math.pow(x1, 2) + b[0][0] * x1 + c

    poly_coefficients = [x2_square_coefficient, x2_coefficient, constant]
    roots = np.roots(poly_coefficients)
    roots_1.append(roots[0])
    roots_2.append(roots[1])
#     print(roots[1])
#
# print('disc x1 points:')
# print(equation_points)
#
# print('roots of the coefficients:')
# print(roots_1)
# print(roots_2)

plt.plot(x1_matrix[0, :], x1_matrix[2, :], 'b.', label="Class 1")
plt.plot(x2_matrix[0, :], x2_matrix[2, :], 'r.', label="Class 2")
plt.plot(equation_points, roots_2, 'g-', label="Dis.Fnc.")
plt.plot(equation_points, roots_1, 'y-', label="Dis.Fnc.")
plt.xlabel('x1')
plt.ylabel('x3')

plt.axis([min_w - 1, max_w + 1, min_w - 1, max_w + 1])
plt.title('x1-x3')
plt.legend(loc=2)
plt.show()
########################################################################################
########################################################################################
'''
Q(c):
Generate 200 new points for each class for testing purposes, classify them and report the classification accuracy
'''
# number of testing points
test_points = 200

# create testing points
_, x1_test_points = h.generate_point_matrix(v_x1, lambda_x1, m1, test_points)
_, x2_test_points = h.generate_point_matrix(v_x2, lambda_x2, m2, test_points)

# classification results
class1_true = 0.0
class1_false = 0.0

class2_true = 0.0
class2_false = 0.0

discriminant_values = []
# classify each point
for i in range(test_points):
    discriminant_value = h.calculate_discriminant(x1_test_points[:, i], sigma_x1, sigma_x2, m1, m2, 0.5, 0.5)
    discriminant_values.append(discriminant_value)
    if discriminant_value > 0:
        class1_true += 1
    else:
        class1_false += 1

print(discriminant_values)


discriminant_values = []
for i in range(test_points):
    discriminant_value = h.calculate_discriminant(x2_test_points[:, i], sigma_x1, sigma_x2, m1, m2, 0.5, 0.5)
    discriminant_values.append(discriminant_value)
    if discriminant_value < 0:
        class2_true += 1
    else:
        class2_false += 1

# print(discriminant_values)

class1_accuracy = (class1_true / test_points) * 100
class2_accuracy = (class2_true / test_points) * 100

x = PrettyTable()
x.field_names = ["Prd\\Tr", "class 1", "class 2", "Accuracy"]
x.add_row(["class 1", class1_true, class1_false, class1_accuracy])
x.add_row(["class 2", class2_false, class2_true, class2_accuracy])

print(x)

########################################################################################
########################################################################################
'''
Q(d):
Use the original training points after diagonalization and plot them for the in the (x1– x2) and (x1–x3) domains.
'''
# diagonalizing the original points and the testing points
v1_matrix, v2_matrix, sigma_v1, sigma_v2, v1_mean, v2_mean = h.diagonalize_simultaneously(x1_matrix, x2_matrix, sigma_x1, sigma_x2, m1, m2)
v1_test_points, v2_test_points, _, _, _, _ = h.diagonalize_simultaneously(x1_test_points, x2_test_points, sigma_x1, sigma_x2, m1, m2)

h.plot_2d_graph(v1_matrix, v2_matrix, 1, 2, 'v1', 'v2', 'v1-v2')
h.plot_2d_graph(v1_matrix, v2_matrix, 1, 3, 'v1', 'v3', 'v1-v3')

########################################################################################
########################################################################################
'''
Q(e):
Assuming that you know the means and covariance matrices of the two “transformed” (diagonalized) classes,
compute the optimal Bayes discriminant function in the transformed domain,
and plot it in the (x1– x2) and (x1– x3) domains.
'''
print('V Sigmas')
print(sigma_v1)
print(sigma_v2)
print('V mean')
print(v1_mean)
print(v2_mean)

a = ((np.linalg.inv(sigma_v2) - np.linalg.inv(sigma_v1)) / 2)
b = np.array(v1_mean.transpose() @ np.linalg.inv(sigma_v1) - v2_mean.transpose() @ np.linalg.inv(sigma_v2))
c = np.math.log(p1 / p2) + np.log(np.linalg.det(sigma_v2) / np.linalg.det(sigma_v1))

equation_points = []
roots_1 = []
roots_2 = []

min_w = min(min(min(v1_matrix[0, :]), min(v2_matrix[0, :])),
            min(min(v1_matrix[1, :]), min(v2_matrix[1, :])))
max_w = max(max(max(v1_matrix[0, :]), max(v2_matrix[0, :])),
            max(max(v1_matrix[1, :]), max(v2_matrix[1, :])))

for x1 in range(-15, 15, 1):
    equation_points.append(x1)
    x2_square_coefficient = a[1][1]
    x2_coefficient = (a[0][1] * x1) + (a[1][0] * x1) + b[0][1]
    constant = a[0][0] * np.math.pow(x1, 2) + b[0][0] * x1 + c

    poly_coefficients = [x2_square_coefficient, x2_coefficient, constant]
    roots = np.roots(poly_coefficients)
    roots_1.append(roots[0])
    roots_2.append(roots[1])


plt.plot(v1_matrix[0, :], v1_matrix[1, :], 'b.', label="Class 1")
plt.plot(v2_matrix[0, :], v2_matrix[1, :], 'r.', label="Class 2")
plt.plot(equation_points, roots_2, 'g-', label="Dis.Fnc.")
plt.plot(equation_points, roots_1, 'y-', label="Dis.Fnc.")
plt.xlabel('v1')
plt.ylabel('v2')

plt.axis([min_w - 1, max_w + 1, min_w - 1, max_w + 1])
plt.title('v1-v2')
plt.legend(loc=2)
plt.show()


equation_points = []
roots_1 = []
roots_2 = []

min_w = min(min(min(v1_matrix[0, :]), min(v2_matrix[0, :])),
            min(min(v1_matrix[2, :]), min(v2_matrix[2, :])))
max_w = max(max(max(v1_matrix[0, :]), max(v2_matrix[0, :])),
            max(max(v1_matrix[2, :]), max(v2_matrix[2, :])))

for x1 in range(-10, 10, 1):
    equation_points.append(x1)
    x2_square_coefficient = a[2][2]
    x2_coefficient = (a[0][2] * x1) + (a[2][0] * x1) + b[0][2]
    constant = a[0][0] * np.math.pow(x1, 2) + b[0][0] * x1 + c

    poly_coefficients = [x2_square_coefficient, x2_coefficient, constant]
    roots = np.roots(poly_coefficients)
    roots_1.append(roots[0])
    roots_2.append(roots[1])


plt.plot(v1_matrix[0, :], v1_matrix[2, :], 'b.', label="Class 1")
plt.plot(v2_matrix[0, :], v2_matrix[2, :], 'r.', label="Class 2")
plt.plot(equation_points, roots_2, 'g-', label="Dis.Fnc.")
plt.plot(equation_points, roots_1, 'y-', label="Dis.Fnc.")
# plt.plot(v1_test_points[0, :], v1_test_points[2, :], 'b.', label="Class 2")
# plt.plot(v2_test_points[0, :], v2_test_points[2, :], 'r.', label="Class 2")

plt.xlabel('v1')
plt.ylabel('v3')

plt.axis([min_w - 1, max_w + 1, min_w - 1, max_w + 1])
plt.title('v1-v3')
plt.legend(loc=2)
plt.show()

########################################################################################
########################################################################################
'''
Q(f):
Using the same testing points of (c), classify them in the transformed domain, and report the classification accuracy.
'''
# classification results
class1_true = 0.0
class1_false = 0.0

class2_true = 0.0
class2_false = 0.0

discriminant_values = []

# classify each point
for i in range(test_points):
    discriminant_value = h.calculate_discriminant(v1_test_points[:, i], sigma_v1, sigma_v2, v1_mean, v2_mean, 0.5, 0.5)
    discriminant_values.append(discriminant_value)
    if discriminant_value > 0:
        class1_true += 1
    else:
        class1_false += 1

# print(discriminant_values)

discriminant_values = []

for i in range(test_points):
    discriminant_value = h.calculate_discriminant(v2_test_points[:, i], sigma_v1, sigma_v2, v1_mean, v2_mean, 0.5, 0.5)
    discriminant_values.append(discriminant_value)
    if discriminant_value < 0:
        class2_true += 1
    else:
        class2_false += 1

# print(discriminant_values)

class1_accuracy = (class1_true / test_points) * 100
class2_accuracy = (class2_true / test_points) * 100

# print(v1_mean)
# print(v2_mean)
# print(sigma_v1)
# print(sigma_v2)
print('\nAfter diagonalizing:')
x = PrettyTable()
x.field_names = ["Prd\\Tr", "class 1", "class 2", "Accuracy"]
x.add_row(["class 1", class1_true, class1_false, class1_accuracy])
x.add_row(["class 2", class2_false, class2_true, class2_accuracy])

print(x)
