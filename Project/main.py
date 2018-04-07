import numpy as np
import os.path
import csv


# ----------------------------------------------------------------------

# parameters
number_of_features = 6
number_of_classes = 2


# ----------------------------------------------------------------------

# reading the data from the CSV files
my_path = os.path.abspath(os.path.dirname(__file__))

csv_path = os.path.join(my_path, "data/glass-modified.csv")
my_data = np.genfromtxt(csv_path, delimiter=',')

class1_csv_path = os.path.join(my_path, "data/class1.csv")
class2_csv_path = os.path.join(my_path, "data/class2.csv")

class1_data = np.genfromtxt(class1_csv_path, delimiter=',')
class2_data = np.genfromtxt(class2_csv_path, delimiter=',')

class1_data = class1_data.transpose()
class2_data = class2_data.transpose()

np.set_printoptions(suppress=True)
print("\nClass 1 Data:")
print(class1_data)

print("\nClass 2 Data:")
print(class2_data)


# ----------------------------------------------------------------------
