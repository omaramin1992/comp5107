import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

########################################################################################

# mean values for the two classes
m1 = np.array([[3],
               [1],
               [4]])

m2 = np.array([[-3],
               [1],
               [-4]])

########################################################################################

# parameters of covariance matrices
a1 = 2
b1 = 3
c1 = 4

alpha1 = 0.1
beta1 = 0.2
number_of_points = 5000


########################################################################################

# create the covariance matrices
def covariance_matrix(a, b, c, alpha, beta):
    # covariance matrix sigma1
    cov_matrix_1 = np.array([[np.math.pow(a, 2), beta * a * b, alpha * a * c],
                             [beta * a * b, np.math.pow(b, 2), beta * b * c],
                             [alpha * a * c, beta * b * c, np.math.pow(c, 2)]])

    # covariance matrix sigma2
    cov_matrix_2 = np.array([[np.math.pow(c, 2), alpha * b * c, beta * a * c],
                             [alpha * b * c, np.math.pow(b, 2), alpha * a * b],
                             [beta * a * c, alpha * a * b, np.math.pow(a, 2)]])
    return cov_matrix_1, cov_matrix_2


# generating gaussian random vectors from Uniform random variables
def generate_point():
    dim = 3
    point = []
    for d in range(0, dim):
        z = 0
        for i in range(0, 12):
            rand = np.random.uniform(0, 1)
            z = z + rand
        z = z - 6
        point.append([z])
    point = np.array(point)
    return point


# generate points from gaussian distribution and transform it back to class distribution
def generate_point_matrix(v, lambda_x, m, points):
    # create initial point
    z_matrix = generate_point()

    # convert them back to the classes distributions
    x_matrix = v @ np.power(lambda_x, 0.5) @ z_matrix + m

    # generate number of points and append them in an array
    for j in range(1, points):
        z_point = generate_point()
        z_matrix = np.append(z_matrix, z_point, axis=1)

        x = v @ np.power(lambda_x, 0.5) @ z_point + m
        x_matrix = np.append(x_matrix, x, axis=1)

    return z_matrix, x_matrix


########################################################################################

# creating the covariance matrices with the parameters
sigma_x1, sigma_x2 = covariance_matrix(a1, b1, c1, alpha1, beta1)
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

# means for y1 and y2
m_y1 = v_x1.transpose() @ m1
m_y2 = v_x2.transpose() @ m2

# means for z1 and z2
m_z1 = v_x1.transpose() @ m_y1
m_z2 = v_x2.transpose() @ m_y2

# create point matrices for the two classes X1 and X2
z1_matrix, x1_matrix = generate_point_matrix(v_x1, lambda_x1, m1, number_of_points)
z2_matrix, x2_matrix = generate_point_matrix(v_x2, lambda_x2, m2, number_of_points)

# covariances of y1 and y2 (classes 1 and 2 in Y world)
sigma_y1 = v_x1.transpose() @ sigma_x1 @ v_x1.transpose()
sigma_y2 = v_x1.transpose() @ sigma_x2 @ v_x1.transpose()
print('\nCovariance of Y1:')
print(sigma_y1)
print('\nCovariance of Y2:')
print(sigma_y2)

# transform points for two classes in Y world
y1_matrix = v_x1.transpose() @ x1_matrix
y2_matrix = v_x1.transpose() @ x2_matrix

# transform points for the two classes in Z
z1 = np.diag(np.power(w_x1, -0.5)) @ v_x1.transpose() @ x1_matrix
z2 = np.diag(np.power(w_x1, -0.5)) @ v_x1.transpose() @ x2_matrix

# covariance matrix of z1 and z2
sigma_z1 = np.diag(np.power(w_x1, -0.5)) @ np.diag(w_x1) @ np.diag(np.power(w_x1, -0.5))
sigma_z2 = np.diag(np.power(w_x1, -0.5)) @ v_x1.transpose() @ sigma_x2 @ v_x1 @ np.diag(np.power(w_x1, -0.5))
print('\nCovariance of Z1:')
print(sigma_z1)
print('\nCovariance of Z2:')
print(sigma_z2)

# print(sigma_z1)
# eigenvalues and eigenvectors of z2 covariance
w_z1, v_z1 = np.linalg.eig(sigma_z1)
w_z2, v_z2 = np.linalg.eig(sigma_z2)
print('\nEigenvalues of z2:')
print(w_z2)

# P overall
p_overall = v_z2.transpose() @ np.diag(np.power(w_x1, -0.5)) @ v_x1.transpose()
print('\nP overall:')
print(p_overall)

# transform points for the two classes in V
# v1_matrix = v_z2.transpose() @ z1_matrix
v1_matrix = p_overall @ x1_matrix

# v2_matrix = v_z2.transpose() @ z2_matrix
v2_matrix = p_overall @ x2_matrix

# covariance matrix of v1 and v2
sigma_v1 = np.round(v_z2.transpose() @ sigma_z1 @ v_z2, 2)
sigma_v2 = np.round(v_z2.transpose() @ sigma_z2 @ v_z2, 2)
print('\nCovariance of V1:')
print(sigma_v1)
print('\nCovariance of V2:')
print(sigma_v2)

########################################################################################

# PLOTTING #

# X WORLD
# plot the first class as blue for (d1 - d2) domain and second class as red
plt.plot(x1_matrix[0, :], x1_matrix[1, :], 'b.', label="Class 1")
plt.plot(x2_matrix[0, :], x2_matrix[1, :], 'r.', label="Class 2")
plt.xlabel("x1")
plt.ylabel("x2")
max_x = max(max(max(x1_matrix[0, :]), max(x2_matrix[1, :])),
            max(max(x1_matrix[1, :]), max(x2_matrix[1, :])))
min_x = min(min(min(x1_matrix[0, :]), min(x2_matrix[0, :])),
            min(min(x1_matrix[1, :]), min(x2_matrix[1, :])))
plt.axis([min_x-1, max_x+1, min_x-1, max_x+1])
plt.title('x1-x2')
plt.legend(loc=2)
plt.show()

# plot the first class as blue for (d1 - d3) domain and second class as red
plt.plot(x1_matrix[0, :], x1_matrix[2, :], 'b.', label="Class 1")
plt.plot(x2_matrix[0, :], x2_matrix[2, :], 'r.', label="Class 2")
plt.xlabel("x1")
plt.ylabel("x3")
max_x = max(max(max(x1_matrix[0, :]), max(x2_matrix[0, :])),
            max(max(x1_matrix[2, :]), max(x2_matrix[2, :])))
min_x = min(min(min(x1_matrix[0, :]), min(x2_matrix[0, :])),
            min(min(x1_matrix[2, :]), min(x2_matrix[2, :])))
plt.axis([min_x-1, max_x+1, min_x-1, max_x+1])
plt.title('x1-x3')
plt.legend(loc=2)
plt.show()

# plot the first class as blue for (d2 - d3) domain and second class as red
plt.plot(x1_matrix[1, :], x1_matrix[2, :], 'b.', label="Class 1")
plt.plot(x2_matrix[1, :], x2_matrix[2, :], 'r.', label="Class 2")
plt.xlabel("x2")
plt.ylabel("x3")
max_x = max(max(max(x1_matrix[1, :]), max(x2_matrix[1, :])),
            max(max(x1_matrix[2, :]), max(x2_matrix[2, :])))
min_x = min(min(min(x1_matrix[1, :]), min(x2_matrix[1, :])),
            min(min(x1_matrix[2, :]), min(x2_matrix[2, :])))
plt.axis([min_x-1, max_x+1, min_x-1, max_x+1])
plt.title('x2-x3')
plt.legend(loc=2)
plt.show()

# 3D plot for the points before diagonalizing
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1_matrix[0, :], x1_matrix[1, :], x1_matrix[2, :], c='b', marker='.', label="Class 1")
ax.scatter(x2_matrix[0, :], x2_matrix[1, :], x2_matrix[2, :], c='r', marker='.', label="Class 2")
max_x = max(max(max(x1_matrix[0, :]), max(x2_matrix[0, :])),
            max(max(x1_matrix[1, :]), max(x2_matrix[1, :])),
            max(max(x1_matrix[2, :]), max(x2_matrix[2, :])))
min_x = min(min(min(x1_matrix[0, :]), min(x2_matrix[0, :])),
            min(min(x1_matrix[1, :]), min(x2_matrix[1, :])),
            min(min(x1_matrix[2, :]), min(x2_matrix[2, :])))
ax.set_xlim(min_x, max_x)
ax.set_ylim(min_x, max_x)
ax.set_zlim(min_x, max_x)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
plt.show()

# Y WORLD
# plot the first class as blue for (d1 - d2) domain and second class as red
plt.plot(y1_matrix[0, :], y1_matrix[1, :], 'b.', label="Class 1")
plt.plot(y2_matrix[0, :], y2_matrix[1, :], 'r.', label="Class 2")
plt.xlabel("y1")
plt.ylabel("y2")
max_y = max(max(max(y1_matrix[0, :]), max(y2_matrix[0, :])),
            max(max(y1_matrix[1, :]), max(y2_matrix[1, :])))
min_y = min(min(min(y1_matrix[0, :]), min(y2_matrix[0, :])),
            min(min(y1_matrix[1, :]), min(y2_matrix[1, :])))
plt.axis([min_y-1, max_y+1, min_y-1, max_y+1])
plt.title('y1-y2')
plt.legend(loc=2)
plt.show()

# plot the first class as blue for (d1 - d3) domain and second class as red
plt.plot(y1_matrix[0, :], y1_matrix[2, :], 'b.', label="Class 1")
plt.plot(y2_matrix[0, :], y2_matrix[2, :], 'r.', label="Class 2")
plt.xlabel("y1")
plt.ylabel("y3")
max_y = max(max(max(y1_matrix[0, :]), max(y2_matrix[0, :])),
            max(max(y1_matrix[2, :]), max(y2_matrix[2, :])))
min_y = min(min(min(y1_matrix[0, :]), min(y2_matrix[0, :])),
            min(min(y1_matrix[2, :]), min(y2_matrix[2, :])))
plt.axis([min_y-1, max_y+1, min_y-1, max_y+1])
plt.title('y1-y3')
plt.legend(loc=2)
plt.show()

# plot the first class as blue for (d2 - d3) domain and second class as red
plt.plot(y1_matrix[1, :], y1_matrix[2, :], 'b.', label="Class 1")
plt.plot(y2_matrix[1, :], y2_matrix[2, :], 'r.', label="Class 2")
plt.xlabel("y2")
plt.ylabel("y3")
max_y = max(max(max(y1_matrix[1, :]), max(y2_matrix[1, :])),
            max(max(y1_matrix[2, :]), max(y2_matrix[2, :])))
min_y = min(min(min(y1_matrix[1, :]), min(y2_matrix[1, :])),
            min(min(y1_matrix[2, :]), min(y2_matrix[2, :])))
plt.axis([min_y-1, max_y+1, min_y-1, max_y+1])
plt.title('y2-y3')
plt.legend(loc=2)
plt.show()

# 3D plot for the points before diagonalizing
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(y1_matrix[0, :], y1_matrix[1, :], y1_matrix[2, :], c='b', marker='.', label="Class 1")
ax.scatter(y2_matrix[0, :], y2_matrix[1, :], y2_matrix[2, :], c='r', marker='.', label="Class 2")
max_y = max(max(max(y1_matrix[0, :]), max(y2_matrix[0, :])),
            max(max(y1_matrix[1, :]), max(y2_matrix[1, :])),
            max(max(y1_matrix[2, :]), max(y2_matrix[2, :])))
min_y = min(min(min(y1_matrix[0, :]), min(y2_matrix[0, :])),
            min(min(y1_matrix[1, :]), min(y2_matrix[1, :])),
            min(min(y1_matrix[2, :]), min(y2_matrix[2, :])))
ax.set_xlim(min_y, max_y)
ax.set_ylim(min_y, max_y)
ax.set_zlim(min_y, max_y)
ax.set_xlabel('y1')
ax.set_ylabel('y2')
ax.set_zlabel('y3')
plt.show()


# Z world
# plot the first class as blue for (d1 - d2) domain and second class as red
plt.plot(z1[0, :], z1[1, :], 'b.', label="Class 1")
plt.plot(z2[0, :], z2[1, :], 'r.', label="Class 2")
plt.xlabel("z1")
plt.ylabel("z2")
max_z = max(max(max(z1[0, :]), max(z2[0, :])),
            max(max(z1[1, :]), max(z2[1, :])))
min_z = min(min(min(z1[0, :]), min(z2[0, :])),
            min(min(z1[1, :]), min(z2[1, :])))
plt.axis([min_z-1, max_z+1, min_z-1, max_z+1])
plt.title('z1-z2')
plt.legend(loc=2)
plt.show()

# plot the first class as blue for (d1 - d3) domain and second class as red
plt.plot(z1[0, :], z1[2, :], 'b.', label="Class 1")
plt.plot(z2[0, :], z2[2, :], 'r.', label="Class 2")
plt.xlabel("z1")
plt.ylabel("z3")
max_z = max(max(max(z1[0, :]), max(z2[0, :])),
            max(max(z1[2, :]), max(z2[2, :])))
min_z = min(min(min(z1[0, :]), min(z2[0, :])),
            min(min(z1[2, :]), min(z2[2, :])))
plt.axis([min_z-1, max_z+1, min_z-1, max_z+1])
plt.title('z1-z3')
plt.legend(loc=2)
plt.show()

# plot the first class as blue for (d2 - d3) domain and second class as red
plt.plot(z1[1, :], z1[2, :], 'b.', label="Class 1")
plt.plot(z2[1, :], z2[2, :], 'r.', label="Class 2")
plt.xlabel("z2")
plt.ylabel("z3")
max_z = max(max(max(z1[1, :]), max(z2[1, :])),
            max(max(z1[2, :]), max(z2[2, :])))
min_z = min(min(min(z1[1, :]), min(z2[1, :])),
            min(min(z1[2, :]), min(z2[2, :])))
plt.axis([min_z-1, max_z+1, min_z-1, max_z+1])
plt.title('z2-z3')
plt.legend(loc=2)
plt.show()

# 3D plot for the points before diagonalizing
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(z1[0, :], z1[1, :], z1_matrix[2, :], c='b', marker='.', label="Class 1")
ax.scatter(z2[0, :], z2[1, :], z2_matrix[2, :], c='r', marker='.', label="Class 2")
max_z = max(max(max(z1[0, :]), max(z2[0, :])),
            max(max(z1[1, :]), max(z2[1, :])),
            max(max(z1[2, :]), max(z2[2, :])))
min_z = min(min(min(z1[0, :]), min(z2[0, :])),
            min(min(z1[1, :]), min(z2[1, :])),
            min(min(z1[2, :]), min(z2[2, :])))
ax.set_xlim(min_z, max_z)
ax.set_ylim(min_z, max_z)
ax.set_zlim(min_z, max_z)
ax.set_xlabel('z1')
ax.set_ylabel('z2')
ax.set_zlabel('z3')

plt.show()

# AFTER DIAGONALIZING #
# plot the first class as blue for (d1 - d2) domain and second class as red
plt.plot(v1_matrix[0, :], v1_matrix[1, :], 'b.', label="Class 1")
plt.plot(v2_matrix[0, :], v2_matrix[1, :], 'r.', label="Class 2")
plt.xlabel("v1")
plt.ylabel("v2")
max_v = max(max(max(v1_matrix[0, :]), max(v2_matrix[0, :])),
            max(max(v1_matrix[1, :]), max(v2_matrix[1, :])))
min_v = min(min(min(v1_matrix[0, :]), min(v2_matrix[0, :])),
            min(min(v1_matrix[1, :]), min(v2_matrix[1, :])))
plt.axis([min_v-1, max_v+1, min_v-1, max_v+1])
plt.title('v1-v2')
plt.legend(loc=2)
plt.show()

# plot the first class as blue for (d1 - d3) domain and second class as red
plt.plot(v1_matrix[0, :], v1_matrix[2, :], 'b.', label="Class 1")
plt.plot(v2_matrix[0, :], v2_matrix[2, :], 'r.', label="Class 2")
plt.xlabel("v1")
plt.ylabel("v3")
max_v = max(max(max(v1_matrix[0, :]), max(v2_matrix[0, :])),
            max(max(v1_matrix[2, :]), max(v2_matrix[2, :])))
min_v = min(min(min(v1_matrix[0, :]), min(v2_matrix[0, :])),
            min(min(v1_matrix[2, :]), min(v2_matrix[2, :])))
plt.axis([min_v-1, max_v+1, min_v-1, max_v+1])
plt.title('v1-v3')
plt.legend(loc=2)
plt.show()

# plot the first class as blue for (d2 - d3) domain and second class as red
plt.plot(v1_matrix[1, :], v1_matrix[2, :], 'b.', label="Class 1")
plt.plot(v2_matrix[1, :], v2_matrix[2, :], 'r.', label="Class 2")
plt.xlabel("v2")
plt.ylabel("v3")
max_v = max(max(max(v1_matrix[1, :]), max(v2_matrix[1, :])),
            max(max(v1_matrix[2, :]), max(v2_matrix[2, :])))
min_v = min(min(min(v1_matrix[1, :]), min(v2_matrix[1, :])),
            min(min(v1_matrix[2, :]), min(v2_matrix[2, :])))
plt.axis([min_v-1, max_v+1, min_v-1, max_v+1])
plt.title('v2-v3')
plt.legend(loc=2)
plt.show()


# 3D plot for the points after diagonalizing
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(v1_matrix[0, :], v1_matrix[1, :], v1_matrix[2, :], c='b', marker='.', label="Class 1")
ax.scatter(v2_matrix[0, :], v2_matrix[1, :], v2_matrix[2, :], c='r', marker='.', label="Class 2")
max_v = max(max(max(v1_matrix[0, :]), max(v2_matrix[0, :])),
            max(max(v1_matrix[1, :]), max(v2_matrix[1, :])),
            max(max(v1_matrix[2, :]), max(v2_matrix[2, :])))
min_v = min(min(min(v1_matrix[0, :]), min(v2_matrix[0, :])),
            min(min(v1_matrix[1, :]), min(v2_matrix[1, :])),
            min(min(v1_matrix[2, :]), min(v2_matrix[2, :])))
ax.set_xlim(min_v, max_v)
ax.set_ylim(min_v, max_v)
ax.set_zlim(min_v, max_v)
ax.set_xlabel('V1')
ax.set_ylabel('V2')
ax.set_zlabel('V3')

plt.show()
