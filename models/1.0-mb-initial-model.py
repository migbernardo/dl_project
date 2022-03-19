import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Flatten, Dense
from reader import read_data

# change working dir to processed
os.chdir(os.path.abspath(os.path.join(os.path.abspath(os.curdir), os.pardir, 'data', 'processed')))

X_train = read_data('X_train.pickle')
y_train = read_data('y_train.pickle')
X_val = read_data('X_val.pickle')
y_val = read_data('y_val.pickle')

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((4, 4), strides=1))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((4, 4), strides=1))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((4, 4), strides=1))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(x=X_train,
                 y=y_train,
                 batch_size=32,
                 epochs=10,
                 verbose=1,
                 validation_data=(X_val, y_val),
                 shuffle=True,
                 use_multiprocessing=True
                 )
