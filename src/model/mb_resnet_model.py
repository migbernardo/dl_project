import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from tensorflow import keras
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.applications.resnet import ResNet50
from mb_initial_model import read_data
from timer import TimingCallback

# enable tensorflow to use gpu
physical_devices = tensorflow.config.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

if __name__ == '__main__':
    # change directory to processed images
    os.chdir(os.path.abspath(os.pardir))
    os.chdir(os.path.abspath(os.pardir))
    os.chdir(os.path.join(os.path.abspath(os.curdir), 'data', 'processed'))

    # load processed image data
    X_train = read_data('X_train.pickle')
    y_train = read_data('y_train.pickle')
    X_val = read_data('X_val.pickle')
    y_val = read_data('y_val.pickle')

    # image augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
                                                           zoom_range=0.2,
                                                           shear_range=0.2
                                                           )

    aug_train = datagen.flow(x=X_train, y=y_train, batch_size=16, shuffle=True, seed=1)
    aug_val = datagen.flow(x=X_val, y=y_val, batch_size=16, shuffle=False, seed=1)

    # set exponential decay for the learning rate
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0001,
        decay_steps=10000000,
        decay_rate=1e-6
    )

    # save training time
    timer = TimingCallback()

    # load base model without top layer
    base_model = ResNet50(input_shape=(64, 64, 3), include_top=False, weights='imagenet')
    # make layers trainable
    for layer in base_model.layers:
        layer.trainable = True

    # add fully connected layer to model
    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(base_model.input, x)

    model.compile(optimizer=keras.optimizers.SGD(
        learning_rate=lr_schedule,
        momentum=0.9,
        nesterov=True,
        name='SGD'
    ), loss='categorical_crossentropy', metrics=['accuracy'])

    # save model's history
    hist = model.fit(x=aug_train,
                     epochs=50,
                     verbose=1,
                     validation_data=aug_val,
                     shuffle=False,
                     steps_per_epoch=500,
                     use_multiprocessing=False
                     )

    # change directory to models and save trained model
    os.chdir(os.path.abspath(os.pardir))
    os.chdir(os.path.abspath(os.pardir))
    os.chdir(os.path.join(os.path.abspath(os.curdir), 'models'))
    model.save('baseline_resnet50.h5', save_format='h5')

    # print training time
    print('Training time: {} seconds'.format(np.round(sum(timer.logs), decimals=2)))

    # plot training and validation loss
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # plot training and validation accuracy
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
