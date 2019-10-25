

# Surrounded region
# Look for O in edge, and do BFS to flip to H
# Turn remain O to X, H to O
# https://leetcode.com/problems/surrounded-regions/description/
class SurroundSolution(object):
    def bfs(self, board, x, y):
        board[x][y] = 'H'
        q = deque([(x, y)])
        dirs = [(0, -1), (0, 1), (1, 0), (-1, 0)]

        while q:
            i, j = q.popleft()
            for d in dirs:
                nx, ny = i + d[0], j + d[1]
                if (0 < nx < len(board) and 0 < ny < len(board[0]) and
                    board[nx][ny] == 'O'):
                    board[nx][ny] = 'H'
                    q.append((nx, ny))

    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        if not board:
            return
        m = len(board)
        n = len(board[0])

        for i in range(m):
            if board[i][0] == 'O':
                self.bfs(board, i, 0)
            if board[i][n-1] == 'O':
                self.bfs(board, i, n-1)

        for j in range(n):
            if board[0][j] == 'O':
                self.bfs(board, 0, j)

            if board[m-1][j] == 'O':
                self.bfs(board, m-1, j)

        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                elif board[i][j] == 'H':
                    board[i][j] = 'O'
