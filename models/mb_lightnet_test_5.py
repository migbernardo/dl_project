import os
from keras.models import load_model
from mb_initial_model import read_data

if __name__ == '__main__':

    os.chdir(os.path.abspath(os.pardir))
    os.chdir(os.path.join(os.path.abspath(os.curdir), 'data', 'processed'))

    X_test = read_data('X_test.pickle')
    y_test = read_data('y_test.pickle')

    os.chdir(os.path.abspath(os.pardir))
    os.chdir(os.path.abspath(os.pardir))
    os.chdir(os.path.join(os.path.abspath(os.curdir), 'models'))

    model = load_model('5_shallow_cnn.h5')
    results = model.evaluate(X_test, y_test)