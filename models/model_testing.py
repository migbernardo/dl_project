import os
import pickle
from keras.models import load_model


# load image data stored in pickle format
def read_data(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    # change directory to processed images
    os.chdir(os.path.abspath(os.pardir))
    os.chdir(os.path.join(os.path.abspath(os.curdir), 'data', 'processed'))

    # load processed image test data
    X_test = read_data('X_test.pickle')
    y_test = read_data('y_test.pickle')

    # change directory to models
    os.chdir(os.path.abspath(os.pardir))
    os.chdir(os.path.abspath(os.pardir))
    os.chdir(os.path.join(os.path.abspath(os.curdir), 'models'))

    # load and evaluate model (input the desired model's name)
    model = load_model('1_shallow_cnn.h5')
    results = model.evaluate(X_test, y_test)
