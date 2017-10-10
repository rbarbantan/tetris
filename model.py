from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Deconv2D
from keras.optimizers import Adam, SGD
from keras import backend as K

import tetris

def build_model():
    # Neural Net for Deep-Q learning Model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(4,4), input_shape=(24,10,1), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Dropout(0.25))
    #model.add(Flatten())
    #model.add(Deconv2D((4,4), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    #model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(tetris.ACTION_SIZE, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.0001), metrics=['accuracy'])

    return model
