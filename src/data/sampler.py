import os
import random
import pickle
import numpy as np


# load image data stored in pickle format
def read_data(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    # change directory to processed images
    os.chdir(os.pardir)
    os.chdir(os.pardir)
    os.chdir(os.path.join('data', 'processed'))

    # load processed image train data
    X_train = read_data('X_train.pickle')
    y_train = read_data('y_train.pickle')

    # sample the train set to generate a representative smaller set
    num_img = 16200
    size = 64
    per_sample = 0.4
    X_sample = np.zeros(shape=[int(num_img * per_sample), size, size, 3])
    y_sample = np.zeros(shape=[int(num_img * per_sample), 10])

    labels = [(X_train[0:1800], y_train[0:1800]),
              (X_train[1800:3600], y_train[1800:3600]),
              (X_train[3600:5400], y_train[3600:5400]),
              (X_train[5400:6900], y_train[5400:6900]),
              (X_train[6900:8400], y_train[6900:8400]),
              (X_train[8400:9600], y_train[8400:9600]),
              (X_train[9600:11100], y_train[9600:11100]),
              (X_train[11100:12900], y_train[11100:12900]),
              (X_train[12900:14400], y_train[12900:14400]),
              (X_train[14400:], y_train[14400:])]

    count = 0
    for label in labels:
        sample = random.sample(range(label[0].shape[0]), int(label[0].shape[0] * per_sample))
        for index in sample:
            X_sample[count][:size][:size][:size] = label[0][index]
            y_sample[count] = label[1][index]
            count += 1

    # save data into pickle format for later use in processed data directory
    with open('X_train_small.pickle', 'wb') as f:
        pickle.dump(X_sample, f)

    with open('y_train_small.pickle', 'wb') as f:
        pickle.dump(y_sample, f)
