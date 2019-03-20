import numpy as np


def load_data(max_range, single_class_side_number=24):
    label_c2 = np.ones(single_class_side_number * 2 + 2).reshape(single_class_side_number * 2 + 2, 1)
    label_c1 = np.zeros(single_class_side_number * 2 + 2).reshape(single_class_side_number * 2 + 2, 1)
    y = np.concatenate((label_c1, label_c2))

    originals = np.array([[0]])
    minor_c1_pos = np.random.uniform(-1, 1, size=(single_class_side_number, 1))
    minor_c1_pos = np.append(minor_c1_pos, originals)
    minor_c1_pos = minor_c1_pos.reshape((single_class_side_number + 1, 1))

    minor_c1_neg = np.random.uniform(-1, 1, size=(single_class_side_number, 1))
    minor_c1_neg = np.append(minor_c1_neg, originals)
    minor_c1_neg = minor_c1_neg.reshape((single_class_side_number + 1, 1))

    minor_c2_pos = np.random.uniform(-1, 1, size=(single_class_side_number, 1))
    minor_c2_pos = np.append(minor_c2_pos, originals)
    minor_c2_pos = minor_c2_pos.reshape((single_class_side_number + 1, 1))

    minor_c2_neg = np.random.uniform(-1, 1, size=(single_class_side_number, 1))
    minor_c2_neg = np.append(minor_c2_neg, originals)
    minor_c2_neg = minor_c2_neg.reshape((single_class_side_number + 1, 1))

    originals = np.array([[1]])
    major_c1_pos = np.random.uniform(1, max_range, size=(single_class_side_number, 1))
    major_c1_pos = np.append(major_c1_pos, originals)
    major_c1_pos = major_c1_pos.reshape((single_class_side_number + 1, 1))

    major_c2_pos = np.random.uniform(1, max_range, size=(single_class_side_number, 1))
    major_c2_pos = np.append(major_c2_pos, originals)
    major_c2_pos = major_c2_pos.reshape((single_class_side_number + 1, 1))

    originals = np.array([[-1]])
    major_c1_neg = np.random.uniform(-max_range, -1, size=(single_class_side_number, 1))
    major_c1_neg = np.append(major_c1_neg, originals)
    major_c1_neg = major_c1_neg.reshape((single_class_side_number + 1, 1))

    originals = np.array([[-1]])
    major_c2_neg = np.random.uniform(-max_range, -1, size=(single_class_side_number, 1))
    major_c2_neg = np.append(major_c2_neg, originals)
    major_c2_neg = major_c2_neg.reshape((single_class_side_number + 1, 1))

    c1_pos = np.append(major_c1_pos, minor_c1_pos, axis=1)
    c1_neg = np.append(major_c1_neg, minor_c1_neg, axis=1)
    c2_pos = np.append(minor_c2_pos, major_c2_pos, axis=1)
    c2_neg = np.append(minor_c2_neg, major_c2_neg, axis=1)

    c1 = np.append(c1_pos, c1_neg, axis=0)
    # c1 = c1.reshape((50,1), axis = 0)
    c2 = np.append(c2_pos, c2_neg, axis=0)
    # c2 = c2.reshape((50,1), axis = 0)

    X = np.append(c1, c2, axis=0)

    return X, y


def append_ones(x):
    ones_array = np.atleast_2d(np.ones(x.shape[0]))
    x = np.concatenate((x, ones_array.T), axis=1)
    return x


def shuffle_data(x, y):
    ind_list = [i for i in range(len(x))]
    np.random.shuffle(ind_list)
    x = x[ind_list, :]
    y = y[ind_list]

    return x, y
