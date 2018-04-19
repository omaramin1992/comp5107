import numpy as np
import math
import operator


def calculate_distance(point1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((point1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def get_neighbors(test_point, class1_training_set, class2_training_set, k):
    neighbors = np.array([[], []])
    distances = np.array([[], []])
    class1_training_set = np.array(class1_training_set).transpose()
    class2_training_set = np.array(class2_training_set).transpose()
    for x in class1_training_set:
        dist = np.absolute(np.linalg.norm(test_point - x))
        temp = np.array([[dist], [1]])
        distances = np.append(distances, temp, axis=1)
    for x in class2_training_set:
        dist = np.absolute(np.linalg.norm(test_point - x))
        temp = np.array([[dist], [2]])
        distances = np.append(distances, temp, axis=1)

    # print('\nbefore sorting:')
    # print(distances.transpose())
    class1_neighbor_count = 0
    class2_neighbor_count = 0

    distances = distances.transpose()
    distances = distances[distances[:, 0].argsort()]
    # print()
    for x in range(0, k):
        # print(distances[x], distances[x][1])
        if distances[x][1] == 1:
            class1_neighbor_count = class1_neighbor_count+1
        else:
            class2_neighbor_count = class2_neighbor_count+1

    point_class = 1 if class1_neighbor_count > class2_neighbor_count else 2
    # print('\nafter sorting:')
    # print(distances)
    # distances = []
    # # length = len(test_instance) - 1
    # for x in range(len(training_set)):
    #     dist = calculate_distance(test_instance, training_set[x], length)
    #     distances.append((training_set[x], dist))
    # distances.sort(key=operator.itemgetter(1))
    # neighbors = []
    # for x in range(k):
    #     neighbors.append(distances[x][0])
    return point_class
