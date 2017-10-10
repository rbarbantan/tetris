import subprocess
import random
import re
import json


positions = range(10)
rotations = range(4)

def random_solution(game_size):
  solution = ''
  for move in range(game_size):
    solution += '{}:{};'.format(random.choice(positions),random.choice(rotations))
  #print(solution)
  return solution

def play(game):
  #print(game)
  game_size = len(game)
  solution =  random_solution(game_size) #'0:0;'
  result = subprocess.check_output(['node', '../dan/tetris/tetris-eval.js', '--pieces', game, '--moves', solution])
  game_result = json.loads(result.decode('utf-8'))
  score = game_result['score']
  
  return score, solution

if __name__ == '__main__':
  solutions = []
  game_id = 0
  with open('games.txt') as games_file:
    for line in games_file:
      line = line.rstrip()
      best_score = 0
      best_solution = ''

      for i in range(200):
        score, solution = play(line)
        if score > best_score:
          best_score = score
          best_solution = solution
      
      print('for game %d scored %d points for solution %s' % (game_id, best_score, best_solution))
      solutions.append(best_solution)
      game_id += 1
      #break  #test first game for now

  with open('submissions\submission_random_200.csv', 'w+') as out:
    out.write('game,moves\n')
    for i,solution in enumerate(solutions):
      out.write('{},"{}"\n'.format(i,solution))
