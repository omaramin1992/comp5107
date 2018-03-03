import numpy as np
import helper as h

########################################################################################
########################################################################################

# mean values for the two classes
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
number_of_points = 5000

########################################################################################
########################################################################################

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

# means for y1 and y2
m_y1 = v_x1.transpose() @ m1
m_y2 = v_x2.transpose() @ m2

# means for z1 and z2
m_z1 = v_x1.transpose() @ m_y1
m_z2 = v_x2.transpose() @ m_y2

# create point matrices for the two classes X1 and X2
z1_matrix, x1_matrix = h.generate_point_matrix(v_x1, lambda_x1, m1, number_of_points)
z2_matrix, x2_matrix = h.generate_point_matrix(v_x2, lambda_x2, m2, number_of_points)

# covariances of y1 and y2 (classes 1 and 2 in Y world)
sigma_y1 = v_x1.transpose() @ sigma_x1 @ v_x1
sigma_y2 = v_x1.transpose() @ sigma_x2 @ v_x1
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
########################################################################################

# PLOTTING #


# X WORLD
# plot the first class as blue for (d1 - d2) domain and second class as red
h.plot_2d_graph(x1_matrix, x2_matrix, 1, 2, 'x1', 'x2', 'x1-x2')

# plot the first class as blue for (d1 - d3) domain and second class as red
h.plot_2d_graph(x1_matrix, x2_matrix, 1, 3, 'x1', 'x3', 'x1-x3')

# plot the first class as blue for (d2 - d3) domain and second class as red
h.plot_2d_graph(x1_matrix, x2_matrix, 2, 3, 'x2', 'x3', 'x2-x3')

# 3D plot for the points before diagonalizing
h.plot_3d_graph(x1_matrix, x2_matrix, 'x1', 'x2', 'x3', 'X-world')
########################################################################################

# Y WORLD
# plot the first class as blue for (d1 - d2) domain and second class as red
h.plot_2d_graph(y1_matrix, y2_matrix, 1, 2, 'y1', 'y2', 'y1-y2')

# plot the first class as blue for (d1 - d3) domain and second class as red
h.plot_2d_graph(y1_matrix, y2_matrix, 1, 3, 'y1', 'y3', 'y1-y3')

# plot the first class as blue for (d2 - d3) domain and second class as red
h.plot_2d_graph(y1_matrix, y2_matrix, 2, 3, 'y2', 'y3', 'y2-y3')

# 3D plot for the points before diagonalizing
h.plot_3d_graph(y1_matrix, y2_matrix, 'y1', 'y2', 'y3', 'Y-world')
########################################################################################

# Z world
# plot the first class as blue for (d1 - d2) domain and second class as red
h.plot_2d_graph(z1, z2, 1, 2, 'z1', 'z2', 'z1-z2')

# plot the first class as blue for (d1 - d3) domain and second class as red
h.plot_2d_graph(z1, z2, 1, 3, 'z1', 'z3', 'z1-z3')

# plot the first class as blue for (d2 - d3) domain and second class as red
h.plot_2d_graph(z1, z2, 2, 3, 'z2', 'z3', 'z2-z3')

# 3D plot for the points before diagonalizing
h.plot_3d_graph(z1, z2, 'z1', 'z2', 'z3', 'Z-world')
########################################################################################

# AFTER DIAGONALIZING #
# plot the first class as blue for (d1 - d2) domain and second class as red
h.plot_2d_graph(v1_matrix, v2_matrix, 1, 2, 'v1', 'v2', 'v1-v2')

# plot the first class as blue for (d1 - d3) domain and second class as red
h.plot_2d_graph(v1_matrix, v2_matrix, 1, 3, 'v1', 'v3', 'v1-v3')

# plot the first class as blue for (d2 - d3) domain and second class as red
h.plot_2d_graph(v1_matrix, v2_matrix, 2, 3, 'v2', 'v3', 'v2-v3')

# 3D plot for the points after diagonalizing
h.plot_3d_graph(v1_matrix, v2_matrix, 'v1', 'v2', 'v3', 'V-world')
