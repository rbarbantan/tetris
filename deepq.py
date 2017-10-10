from collections import deque
import os
import numpy as np
import random
import subprocess
import json
import tetris
import player
import matplotlib.pyplot as plt
import keras
import model

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

class Game:
    def __init__(self):
        self.game_length = 100
        self.action_size = tetris.ACTION_SIZE
        self.reset()

    def reset(self):
        self.state = np.zeros(tetris.STATE_SIZE)
        self.inputs = ''.join([random.choice('IJLOSTZ') for _ in range(self.game_length)])
        self.action_history = []
        self.frame = 0
        self.score = 0
        self.tetris = tetris.Tetris(self.inputs)
        return self.state

    def convert_action(self):
        return ';'.join(['{}:{}'.format((action % 12)-1,int(action / 12)) for action in self.action_history])

    def step(self, action):
        self.action_history.append(action)
        self.tetris.make_move((action % 12)-1, int(action/12))
        self.state = np.zeros(tetris.STATE_SIZE)
        header = np.zeros((4, 10))
        shape = tetris.SHAPES[self.inputs[self.frame]]
        header[:len(shape), :len(shape[0])] = shape
        self.state[:4] = header
        self.state[4:] = self.tetris.grid
        self.state = (self.state > 0).astype(np.int8)

        old_score = self.score
        self.score = self.tetris.score#self.get_score(self.tetris.score)
        #print(self.score - old_score)
        self.frame += 1

        return self.state, self.score - old_score, self.tetris.won or self.tetris.lost



class Env:
    def __init__(self):
        self.sample_batch_size = 64
        self.episodes = 1000000
        self.env = Game()
        self.state_size = len(self.env.state)
        self.action_size = self.env.action_size
        self.agent = Agent(self.state_size, self.action_size)

    def __engineered_reward(self):
        heights = np.zeros(10)
        holes = 0
        for col in range(10):
            column = self.env.tetris.grid[:, col]
            nonzeros = np.flatnonzero(column)
            heights[col] = (20 - nonzeros[0]) if len(nonzeros) > 0 else 0
            ceiling = False
            for i in column:
                if i == 0 and ceiling:
                    # found hole
                    holes += 1
                    ceiling = False
                if i > 0:
                    ceiling = True

        agg_height = np.sum(heights)
        # lines = int((game.score - old_Score)/ 100)
        lines = 0
        for row in self.env.tetris.grid:
            if len(np.flatnonzero(row)) == len(row):
                lines += 1

        bumpiness = 0
        for col in range(9):
            bumpiness += abs(heights[col] - heights[col + 1])

        reward = -0.51066 * agg_height + 0.760666 * lines - 0.35663 * holes - 0.184483 * bumpiness
        return reward

    def __evaluate_model(self, eval_memory):
        x = [state for state, _, _, _, _ in eval_memory]
        y = [reward for _, _, reward, _, _ in eval_memory]
        actions = [action for _, action, _, _, _ in eval_memory]
        x = np.array(x)
        x = np.reshape(x, [x.shape[0], x.shape[1], x.shape[2], 1])
        predictions = self.agent.brain.predict(x)
        delta = 0
        for i in range(len(predictions)):
            delta += abs(predictions[i][actions[i]] - y[i])

        return delta



    def run(self):
        try:
            average_score = 0
            score_chart = {}
            eval_memory = np.load('eval_data.npy')

            for index_episode in range(self.episodes):
                print('runing episode %d' % index_episode)
                state = self.env.reset()
                #state = np.reshape(state, [1, self.state_size])
                self.agent.game = self.env.tetris
                done = False

                while not done:
                    #                    self.env.render()
                    action = self.agent.act(state)
                    reward = player.score_move(self.agent.game, action)
                    next_state, _, done = self.env.step(action)
                    #reward = (reward + 1000)/2000
                    #reward = self.__engineered_reward()

                    #next_state = np.reshape(next_state, [1, self.state_size])
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state

                average_score += self.env.tetris.score
                if self.env.tetris.won:
                    print("won with %d" % self.env.tetris.score)

                #if index_episode == 200:
                #    np.save('eval_data.npy', self.agent.memory)
                #    print('saved evaluation data')
                if index_episode % 5 == 0:
                    delta = self.__evaluate_model(eval_memory)
                    print('off by %f' % delta)
                    print(self.env.inputs)
                    print(self.env.convert_action())
                    print(self.env.tetris.score)
                    print(self.agent.exploration_rate)
                    print("average %d" % (average_score/50))
                    score_chart[index_episode] = delta #average_score/50
                    average_score = 0

                    tochart = sorted(score_chart.items())
                    x,y = zip(*tochart)
                    plt.plot(x,y)
                    #plt.bar(range(len(score_chart)), score_chart.values(), align='center')
                    #plt.xticks(range(len(score_chart)), list(score_chart.keys()))
                    plt.savefig('score_char.png')
                    plt.gcf().clear()
                    #plt.show()
                self.agent.replay(self.sample_batch_size)
                self.agent.target_train()
        finally:
            self.agent.save_model()


class Agent():
    def __init__(self, state_size, action_size):
        self.weight_backup      = "deepq_weight.h5"
        self.state_size         = state_size
        self.action_size        = action_size
        self.memory             = deque(maxlen=100000)
        self.gamma              = 0.95
        self.exploration_rate   = 1.0
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.99995
        self.brain              = model.build_model()
        #self.brain.load_weights(self.weight_backup)#keras.models.load_model('model_0001.h5')
        self.target             = model.build_model()#keras.models.load_model('model_0001.h5')#model.build_model()
        #self.target.load_weights(self.weight_backup)
        self.game = None


    def target_train(self):
        weights = self.brain.get_weights()
        target_weights = self.target.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target.set_weights(target_weights)
        #print(self.target.metrics[0])

    def save_model(self):
        self.brain.save(self.weight_backup)

    def act(self, state):
        some_rand = np.random.rand()
        if some_rand <= self.exploration_rate:
            if some_rand <= self.exploration_min:
                return random.randrange(self.action_size)
            else:
                return player.propose_move(self.game)[0]#
        act_values = self.brain.predict(np.reshape(state, [1,24, 10, 1]))
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        x = np.zeros((sample_batch_size, tetris.STATE_SIZE[0], tetris.STATE_SIZE[1], 1))
        y = np.zeros((sample_batch_size, tetris.ACTION_SIZE))
        sample_batch = random.sample(self.memory, sample_batch_size)
        for i, (state, action, reward, next_state, done) in enumerate(sample_batch):
            target = self.target.predict(np.reshape(state, [1,24, 10, 1]))
            if done:
                target[0][action] = reward
            else:
                q__future = max(self.target.predict(np.reshape(next_state, [1, 24, 10, 1]))[0])
                target[0][action] = reward + q__future * self.gamma
            x[i:i+1] = np.reshape(state, [24,10,1])
            y[i] = target[0]
        self.brain.train_on_batch(x, y)

        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay
            #print(self.exploration_rate)

if __name__ == '__main__':
    env = Env()
    env.run()
