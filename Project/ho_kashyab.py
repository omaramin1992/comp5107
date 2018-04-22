import numpy as np


def ho_kashyap(class1_data, class2_data):
    class1_ones = np.ones(len(class1_data[0]))
    class2_ones = np.ones(len(class2_data[0]))

    y1 = np.array(class1_data)
    y2 = np.array(class2_data)

    y1 = np.insert(y1, 0, class1_ones, axis=0)
    y2 = np.insert(y2, 0, class2_ones, axis=0)

    y1 = y1.transpose()
    y2 = -1 * y2.transpose()

    y = np.append(y1, y2, axis=0)

    a = np.ones((len(y[0]), 1))
    b = np.ones((len(y), 1))

    lr = 0.1  # learning rate
    i = 0
    e = y @ a - b
    e1 = np.linalg.norm(e)

    a_initial = np.array(a)
    # print(a)
    # while np.min(e1) < 0.1:
    # while e1 > 0.1:
    for i in range(400):
    # while not np.array_equal(e, e1):
        # print('before', e)
        # print(i)
        # # print(a_initial)
        # print(a)
        # print('e', y@a)
        e = y @ a - b
        b = b + lr * (e - np.abs(e))
        a = np.linalg.inv(y.transpose() @ y) @ y.transpose() @ b
        e1 = np.linalg.norm(e)
        # print(e1)

        i = i + 1
    # print('\nY1: ', y1, len(y1))
    # print('\nY2: ', y2, len(y2))
    #
    # print('\nY: ', y, len(y))

    print('\nA: ', a, len(a))
    # print('\nB: ', b, len(b))

    return a, b


ones = np.ones(5)
