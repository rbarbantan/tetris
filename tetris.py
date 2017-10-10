import numpy as np
import copy
import cProfile

'''
    converted from https://github.com/danmana/tetris 
'''
# Block shapes
SHAPES = {
    'I': [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
    'J': [[2, 0, 0], [2, 2, 2], [0, 0, 0]],
    'L': [[0, 0, 3], [3, 3, 3], [0, 0, 0]],
    'O': [[4, 4], [4, 4]],
    'S': [[0, 5, 5], [5, 5, 0], [0, 0, 0]],
    'T': [[0, 6, 0], [6, 6, 6], [0, 0, 0]],
    'Z': [[7, 7, 0], [0, 7, 7], [0, 0, 0]]
}


#Define 10x20 grid as the board
GRID_W = 10
GRID_H = 20
STATE_SIZE = (24,10)
ACTION_SIZE = 48


def get_bounded_x(shape, x):
    bounds = get_shape_position_bounds(shape)
    if x < bounds['min']:
        return bounds['min']

    if x > bounds['max']:
        return bounds['max']

    return x


def get_shape_position_bounds(shape):
    minX = GRID_W
    maxX = 0
    for i in range(len(shape)):
      for j in range(len(shape[i])):
        if shape[i][j]:
          minX = min(minX, j)
          maxX = max(maxX, j)

    return {'min': -minX, 'max': GRID_W - 1 - maxX}


def is_loss(shape, y):

    for i in range(len(shape)):
        for j in range(len(shape)):
            if shape[i][j] & (y + i < 0):
                return True

    return False


class Tetris:
    def __init__(self, nextShapes):
        self.grid = np.zeros((GRID_H, GRID_W))

        self.nextShapes = nextShapes
        self.shapeIndex = 0

        self.score = 0
        self.lost = False
        self.won = False

    '''
       * Make a move. This will alter the current state of the game.
       * @param x the column where to place the next shape
       * @param rot the numebr of rotations to apply to the next shape
    '''
    def make_move(self, x, rot, is_test=False):
        shape = copy.deepcopy(SHAPES[self.nextShapes[self.shapeIndex]])

        shape = self.rotate(shape, rot)

        # bound the X so the shape is in the grid
        x = get_bounded_x(shape, x)

        # find where the shape drops
        y = self.get_drop_location(shape, x)

        if is_loss(shape, y):
          self.lost = True
          #self.score -= 1000
        else:
          self.score += 1
          self.apply_shape(shape, x, y)
          if not is_test:
              self.score += self.clear_rows()

          self.shapeIndex += 1
          if self.shapeIndex >= len(self.nextShapes):
            self.won = True

    '''
       * Clear all filled rows
       * @param grid
       * @returns {number} the score from clearing rows
    '''
    def clear_rows(self):
        cleared = 0
        score = 0
        for i in range(GRID_H):
          is_full = True
          for j in range(GRID_W):
            if self.grid[i][j] == 0:
              is_full = False
              break

          if is_full:
            cleared += 1
            self.grid = np.delete(self.grid, i, 0)
            self.grid = np.insert(self.grid, 0, np.zeros(GRID_W), axis=0)

        if cleared == 1:
          score = 100
        elif cleared == 2:
          score = 200
        elif cleared == 3:
          score = 400
        elif cleared >= 4:
          score = 800

        return score

    '''
    * Paste the given shape in the grid at x,y starting with the top,left corner of the shape.
    * @param grid
    * @param shape
    * @param x
    * @param y
    '''
    def apply_shape(self, shape, x, y):
        for i in range(len(shape)):
            for j in range(len(shape[i])):
                if shape[i][j]:
                    if ((x + j) < GRID_W) and ((y + i) >= 0) and ((y + i) < GRID_H):
                        self.grid[y + i][x + j] = shape[i][j]

    '''
    * Get the row where this shape will drop
    * @param grid
    * @param shape
    * @param x
    * @returns {number}
    '''
    def get_drop_location(self, shape, x):
        for i in range(-len(shape), GRID_H):
            if self.collides(shape, x, i):
                return i - 1

        return GRID_H - 1


    '''
    * Check if placing the given shape at position x,y will collide with the grid or has filled cells outside the grid.
    * x,y represent where to place the top, left cell of the shape
    *
    * @param grid the grid to test on
    * @param shape the shape to try and place
    * @param x the column position where to place the shape
    * @param y the row where to place it (0 = on top, can be negative, GRID_H-1 on bottom)
    * @returns {boolean} if the shape collides with anything on the grid
    '''
    def collides(self, shape, x, y):

        for i in range(len(shape)):
            for j in range(len(shape[i])):
                if shape[i][j] > 0:
                    if ((x + j) < 0) or ((x + j) >= GRID_W):
                        return True

                    if (y + i) >= GRID_H:
                        return True

                    if (y + i) >= 0:
                        #print(x, y, i, j)
                        if self.grid[y + i][x + j] != 0:
                            return True

        return False

    '''
    * Rotate a shape or grid.
    * @param matrix the 2 dimensional array to rotate
    * @param times the number of rotations
    * @returns {*}
    '''
    def rotate(self, matrix, times):
        for t in range(times):
          #flip the shape matrix
          matrix = np.transpose(matrix)
          #and for the length of the matrix, reverse each column
          for i in range(len(matrix)):
                matrix[i] = np.flip(matrix[i],0)

        return matrix


def encode_piece(piece):
    encoded = np.zeros(7)
    encoded['IJLOSTZ'.index(piece)] = 1
    return encoded


def test():
    g = Tetris('ILILILILILILILILILILILILILILILILILILILILILILILILILIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII')
    for x in range(50):
        g.make_move(0,1)


if __name__ == '__main__':
    #game = Tetris('IIL')
    #game.make_move(0, 1)
    #game.make_move(4, 0)
    #print(game.grid)
    #print(game.grid.shape)
    cProfile.run('test()')

