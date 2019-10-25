

# Design tic toc toe game
# https://leetcode.com/problems/design-tic-tac-toe
# Assign value for each player move, win score = n*playerScore
# Store total value for row, column and 2 diagonals (x+y = n-1 and y-x = 0)
# Runtime O(1), space O(n)
class TicTacToe:

    def __init__(self, n):
        """
        Initialize your data structure here.
        :type n: int
        """
        self.rows = [0 for _ in range(n)]
        self.columns = [0 for _ in range(n)]
        # First diagonal x+y, second y-x
        self.diagonal = [0, 0]
        self.score = {1: 1, 2: n+1}
        self.win = {1: n, 2: (n+1)*n}
        self.size = n

    def move(self, row, col, player):
        """
        Player {player} makes a move at ({row}, {col}).
        @param row The row of the board.
        @param col The column of the board.
        @param player The player, can be either 1 or 2.
        @return The current winning condition, can be either:
                0: No one wins.
                1: Player 1 wins.
                2: Player 2 wins.
        :type row: int
        :type col: int
        :type player: int
        :rtype: int
        """
        score = self.score[player]
        win_score = self.win[player]
        self.rows[row] += score
        if self.rows[row] == win_score:
            return player

        self.columns[col] += score
        if self.columns[col] == win_score:
            return player
        if col - row == 0:
            self.diagonal[1] += score
            if self.diagonal[1] == win_score:
                return player
        if col + row == self.size - 1:
            self.diagonal[0] += score
            if self.diagonal[0] == win_score:
                return player
        return 0
