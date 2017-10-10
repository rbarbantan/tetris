import copy
import numpy as np
import tetris


def propose_move(tetris):
    max_score = -1000
    best_move = 0
    for action in range(48):
        reward = score_move(tetris, action)
        if reward >= max_score:
            max_score = reward
            best_move = action

    return best_move, max_score


def score_move(tetris, action):
    game = copy.deepcopy(tetris)
    game.make_move((action % 12) - 1, int(action / 12), True)

    heights = np.zeros(10)
    holes = 0
    for col in range(10):
        column = game.grid[:, col]
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
    for row in game.grid:
        if len(np.flatnonzero(row)) == len(row):
            lines += 1

    bumpiness = 0
    for col in range(9):
        bumpiness += abs(heights[col] - heights[col + 1])

    reward = -0.51066 * agg_height + 0.760666 * lines - 0.35663 * holes - 0.184483 * bumpiness

    return reward

def predict():
    total_score = 0
    results = []
    count = 1
    with open('games.txt') as games_file:
        for line in games_file:
            print('playing {}'.format(count))
            count += 1
            line = line.rstrip()
            game = tetris.Tetris(line)

            action_history = []
            for i in range(len(line)):
                if not (game.won or game.lost):
                    action,_ = propose_move(game)
                    move = ((action % 12) - 1, int(action / 12))
                    game.make_move(move[0], move[1])
                    action_history.append(move)
            results.append(';'.join(['{}:{}'.format(a,b) for a,b in action_history]))
            if game.lost:
                game.score += 1000
            print(game.won, game.score)

            total_score += game.score
            print(total_score)

    with open('submissions/player.txt', 'w') as submission:
        for result in results:
            submission.write('{}\n'.format(result))



if __name__ == '__main__':
    '''action_history = []
    input = 'OOTLIIOLJIOSSSSJIJSJIJSLTJTISZSTSLZIIIOLSLISOJLOILTLIZOTTSTTIJZTOSOLJOZIJOLJZZZTSZJJJOJOOOTZLZLOIZJJ'
    game = tetris.Tetris(input)
    player = Player(game)
    for i in range(len(input)):
        if not (game.won or game.lost):
            action = player.propose_move()
            move = ((action % 12)-1, int(action/12))
            game.make_move(move[0], move[1])
            action_history.append(move)
    print(game.won, game.lost, game.score)
    print(';'.join(['{}:{}'.format(a,b) for a,b in action_history]))'''
    predict()
