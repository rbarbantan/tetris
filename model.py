from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import SGD

import tetris


def build_model():
    # Neural Net for Deep-Q learning Model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(4,4), input_shape=(24,10,1), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(tetris.ACTION_SIZE, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.0001), metrics=['accuracy'])

    return model
