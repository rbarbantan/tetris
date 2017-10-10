import player
import tetris
import random
import numpy as np
import model
from joblib import Parallel, delayed
import multiprocessing
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import time


NO_GAMES = 5000
GAME_LENGTH = 1000


def play_game(g):
    x = []
    y = []
    input = ''.join([random.choice('IJLOSTZ') for _ in range(GAME_LENGTH)])
    game = tetris.Tetris(input)
    score = 0
    for i in input:
        if game.won or game.lost:
            break
        state = np.zeros(tetris.STATE_SIZE)
        header = np.zeros((4,10))
        shape = tetris.SHAPES[i]
        header[:len(shape),:len(shape[0])] = shape
        state[:4] = header
        state[4:] = game.grid
        state = (state > 0).astype(np.int8)
        action, r = player.propose_move(game)
        move = ((action % (tetris.GRID_W + 2)) - 1, int(action / (tetris.GRID_W + 2)))
        game.make_move(move[0], move[1])

        reward = np.zeros(tetris.ACTION_SIZE)
        reward[action] = r

        x.append(state)
        y.append(reward)
    print('playing game {} of {} scored {}'.format(g, NO_GAMES, game.score))
    return x,y


def create_data():
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(play_game)(g) for g in range(NO_GAMES))
    x = np.concatenate([a for a,_ in results])
    y = np.concatenate([a for _,a in results])

    np.save('x.npy', x)
    np.save('y.npy', y)


def train():
    x = np.load('x.npy')
    y = np.load('y.npy')

    #y = (y+1000)/2000
    print(len(x), len(y))

    x = x.reshape(x.shape[0], 24, 10, 1)
    checkpoint = ModelCheckpoint('model_0001.h5', monitor='val_loss', save_best_only=True)
    tensorboard = TensorBoard('./logs/{}'.format(time.time()))
    plateau = ReduceLROnPlateau(patience=3)
    brain = model.build_model()
    brain.fit(x,y,batch_size=32, epochs=20, validation_split=0.1, shuffle=True, callbacks=[checkpoint, tensorboard, plateau])
    brain.save('model.h5')


if __name__ == '__main__':
    #create_data()
    train()