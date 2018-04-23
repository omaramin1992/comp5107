import numpy as np

x = np.array([[1, 2, 3, 4, 53, 56],
              [4, 5, 6, 1, 42, 34],
              [3, 8, 9, 1, 12, 36]])

# i = columns, j = rows
# sub_array = np.array([[x[i][j] for j in range(2, 3)] for i in range(3)])
sub_array = x[2, :]
print(sub_array.size)
# array_without_subarray = np.delete(x, x[:, 2:5], axis=1)
x1 = x[:, 0:2]
x1 = np.append(x1, x[:, 4:6], axis=1)

print(sub_array)
print(x1)
# print(x[:, [1]])
