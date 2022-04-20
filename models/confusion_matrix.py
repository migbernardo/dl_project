import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from model_testing import read_data
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':

    labels = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop',
              'Residential', 'River', 'SeaLake']

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

    # load and predict test set (input the desired model's name)
    model = load_model('3_shallow_cnn_smallds_3.h5')
    pred = model.predict(X_test)

    # plot confusion matrix
    ax = plt.subplot()
    cm = confusion_matrix(np.asarray(y_test).argmax(axis=1), np.asarray(pred).argmax(axis=1), normalize='true')
    cm = np.round(cm, decimals=2)
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    plt.subplots_adjust(bottom=0.4)
    plt.subplots_adjust(left=0.3)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.show()
