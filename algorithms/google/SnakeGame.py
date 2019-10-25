

# Snakegame: Google
# https://leetcode.com/problems/design-snake-game
# Use deque to store snake, pop food from queue
class SnakeGame:

    def __init__(self, width, height, food):
        """
        Initialize your data structure here.
        @param width - screen width
        @param height - screen height
        @param food - A list of food positions
        E.g food = [[1,1], [1,0]] means the first food is positioned at [1,1], the second is at [1,0].
        :type width: int
        :type height: int
        :type food: List[List[int]]
        """
        self.width = width
        self.height = height
        self.food = deque(food)
        self.snake = deque([[0, 0]])
        # For checking if snake eats its body
        self.dirs = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

    def isValidMove(self, move):
        return (0 <= move[0] < self.height and 0 <= move[1] < self.width and
                (move not in self.snake or move == self.snake[-1]))

    def move(self, direction):
        """
        Moves the snake.
        @param direction - 'U' = Up, 'L' = Left, 'R' = Right, 'D' = Down
        @return The game's score after the move. Return -1 if game over.
        Game over when snake crosses the screen boundary or bites its body.
        :type direction: str
        :rtype: int
        """
        head = self.snake[0]
        delta = self.dirs[direction]
        nextMove = [head[0] + delta[0], head[1] + delta[1]]
        if not self.isValidMove(nextMove):
            return -1

        if self.food and nextMove == self.food[0]:
            self.food.popleft()
        else:
            self.snake.pop()

        self.snake.appendleft(nextMove)

        return len(self.snake) - 1
