from PIL import Image
import os
import pickle
import numpy as np
from path import get_path


# function that splits and merges all images into the respective train, validation or test tensors
# returns a list of tuple tensors
def data_gen(src_path, size):
    # start counters
    num_img = 0
    train_count = 0
    val_count = 0
    test_count = 0
    # save list of labels corresponding to each dir's name
    labels = os.listdir(path=src_path)
    labels.sort()
    # go through each label/dir and add number of images inside to num_img
    for i in labels:
        num_img += len(os.listdir(path=os.path.join(src_path, i)))

    # initialize arrays filled with zeros
    # corresponding to the output dimensions of the images and its labels for each split
    X_train = np.zeros(shape=[int(num_img * 0.6), size, size, 3])
    y_train = np.zeros(shape=[int(num_img * 0.6), len(labels)])
    X_val = np.zeros(shape=[int(num_img * 0.2), size, size, 3])
    y_val = np.zeros(shape=[int(num_img * 0.2), len(labels)])
    X_test = np.zeros(shape=[int(num_img * 0.2), size, size, 3])
    y_test = np.zeros(shape=[int(num_img * 0.2), len(labels)])

    # go through labels, save the path to its corresponding dir, and save the number of images inside
    for index, label in enumerate(labels):
        cur_path = os.path.join(src_path, label)
        n_img = len(os.listdir(path=cur_path))

        # open each image, convert into array, add data and labels to the corresponding split arrays
        for i in range(1, n_img + 1):
            # fill train split when n_img has not reached 61% of the label
            if i <= int(n_img * 0.6):
                # keep track of index of the train arrays
                train_count += 1
                img = Image.open(os.path.join(cur_path, labels[index] + '_{}.jpg'.format(i)))
                img_arr = np.array(img)
                label_arr = np.zeros(len(labels))
                np.put(label_arr, index, 1)
                X_train[train_count - 1][:size][:size][:size] = img_arr
                y_train[train_count - 1] = label_arr

                # fill val split when n_img has not reached 81% of the label
            elif i <= int(n_img * 0.8):
                # keep track of index of the val arrays
                val_count += 1
                img = Image.open(os.path.join(cur_path, labels[index] + '_{}.jpg'.format(i)))
                img_arr = np.array(img)
                label_arr = np.zeros(len(labels))
                np.put(label_arr, index, 1)
                X_val[val_count - 1][:size][:size][:size] = img_arr
                y_val[val_count - 1] = label_arr

                # fill test split with remaining images of the label
            else:
                # keep track of index of the test arrays
                test_count += 1
                img = Image.open(os.path.join(cur_path, labels[index] + '_{}.jpg'.format(i)))
                img_arr = np.array(img)
                label_arr = np.zeros(len(labels))
                np.put(label_arr, index, 1)
                X_test[test_count - 1][:size][:size][:size] = img_arr
                y_test[test_count - 1] = label_arr

                # return list of tensor tuples
    return [(X_train, y_train), (X_val, y_val), (X_test, y_test)]


# get path to images dir
dst_path = os.path.join(get_path(), 'data', 'raw')

# convert data into tensors and split it
data = data_gen(dst_path, 64)

# make interim dir
os.mkdir(os.path.join(get_path(), 'data', 'interim'))

# change working dir to interim
os.chdir(os.path.join(get_path(), 'data', 'interim'))

# save data into pickle format for later use in interim dir
for position, name in enumerate(['X_train.pickle', 'X_val.pickle', 'X_test.pickle']):
    with open(name, 'wb') as f:
        pickle.dump(data[position][0], f)

for position, name in enumerate(['y_train.pickle', 'y_val.pickle', 'y_test.pickle']):
    with open(name, 'wb') as f:
        pickle.dump(data[position][1], f)
