import numpy as np
import helper as h

########################################################################################
########################################################################################

# mean values for the two classes
m1 = np.array([[1],
               [11],
               [14]])

m2 = np.array([[-13],
               [-11],
               [-14]])

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

# Q(a) #

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

# plot the first class as blue for (d2 - d3) domain and second class as red
h.plot_2d_graph(x1_matrix, x2_matrix, 2, 3, 'x2', 'x3', 'x1-x2')

# 3D plot for the points before diagonalizing
h.plot_3d_graph(x1_matrix, x2_matrix, 'x1', 'x2', 'x3', 'X-world')

########################################################################################
########################################################################################

# Q(b) #


########################################################################################
########################################################################################

# Q(c) #


########################################################################################
########################################################################################

# Q(d) #


########################################################################################
########################################################################################

# Q(e) #


########################################################################################
########################################################################################

# Q(f) #
