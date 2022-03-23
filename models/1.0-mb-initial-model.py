import os
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Flatten, Dense
from reader import read_data

# change working dir to processed
os.chdir(os.path.abspath(os.path.join(os.path.abspath(os.curdir), os.pardir, 'data', 'processed')))

X_train = read_data('X_train.pickle')
y_train = read_data('y_train.pickle')
X_val = read_data('X_val.pickle')
y_val = read_data('y_val.pickle')

datagen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
                                                       zoom_range=0.2,
                                                       shear_range=0.2
                                                       )
aug_train = datagen.flow(x=X_train, y=y_train, batch_size=16, shuffle=True, seed=1)
aug_val = datagen.flow(x=X_val, y=y_val, batch_size=16, shuffle=False, seed=1)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((4, 4), strides=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((4, 4), strides=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((4, 4), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=100000,
    decay_rate=1e-6
)
model.compile(optimizer=keras.optimizers.SGD(
    learning_rate=lr_schedule,
    momentum=0.9,
    nesterov=True,
    name='SGD'
), loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(x=aug_train,
                 epochs=5,
                 verbose=1,
                 validation_data=aug_val,
                 shuffle=False,
                 steps_per_epoch=20,
                 use_multiprocessing=True
                 )
