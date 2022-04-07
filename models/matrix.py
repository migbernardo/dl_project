import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from mb_initial_model import read_data
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':

    labels = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop',
              'Residential', 'River', 'SeaLake']

    os.chdir(os.path.abspath(os.pardir))
    os.chdir(os.path.join(os.path.abspath(os.curdir), 'data', 'processed'))

    X_test = read_data('X_test.pickle')
    y_test = read_data('y_test.pickle')

    os.chdir(os.path.abspath(os.pardir))
    os.chdir(os.path.abspath(os.pardir))
    os.chdir(os.path.join(os.path.abspath(os.curdir), 'models'))

    model = load_model('baseline_resnet50.h5')
    pred = model.predict(X_test)

    ax = plt.subplot()
    cm = confusion_matrix(np.asarray(y_test).argmax(axis=1), np.asarray(pred).argmax(axis=1), normalize='true')
    cm = np.round(cm, decimals=2)
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=30)
    plt.show()
