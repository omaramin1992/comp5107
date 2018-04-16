import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# ----------------------------------------------------------------------


# ----------------------------------------------------------------------

def calculate_discriminant(x, sigma_1, sigma_2, mean_1, mean_2, p1, p2):
    x = np.array(x)
    sigma_1 = np.array(sigma_1)
    sigma_2 = np.array(sigma_2)
    mean_1 = np.array(mean_1)
    mean_2 = np.array(mean_2)

    a = ((np.linalg.inv(sigma_2) - np.linalg.inv(sigma_1)) / 2)
    b = np.array(mean_1.transpose() @ np.linalg.inv(sigma_1) - mean_2.transpose() @ np.linalg.inv(sigma_2))
    c = np.math.log(p1 / p2) + np.log(np.linalg.det(sigma_2) / np.linalg.det(sigma_1))

    discriminant_value = x.transpose() @ a @ x + b @ x + c

    return discriminant_value


# ----------------------------------------------------------------------




