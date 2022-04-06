import os
import numpy as np
from keras.models import load_model
from mb_initial_model import read_data
from sklearn.metrics import multilabel_confusion_matrix

if __name__ == '__main__':

    os.chdir(os.path.abspath(os.pardir))
    os.chdir(os.path.join(os.path.abspath(os.curdir), 'data', 'processed'))

    X_test = read_data('X_test.pickle')
    y_test = read_data('y_test.pickle')

    os.chdir(os.path.abspath(os.pardir))
    os.chdir(os.path.abspath(os.pardir))
    os.chdir(os.path.join(os.path.abspath(os.curdir), 'models'))

    model = load_model('1_shallow_cnn.h5')
    pred = model.predict(X_test)

    preds = []
    for img in pred:
        encode = np.zeros(10)
        encode[int(np.argmax(img))] = 1
        preds.append(encode.tolist())

    conf = multilabel_confusion_matrix(y_test, preds)