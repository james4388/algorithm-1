from .UnionFind import UnionFind


# https://leetcode.com/problems/bricks-falling-when-hit/
# Brick in grids = 1, it's not falling if connected to top grids, use series of
# hits, return number of falling bricks for each hit
# Solution: add all hit to current grid, use union find with size of each cluster
# Reverse hits, add brick back, find the new size
# Runtime: O(N*Q*a(N*Q)) where N = row  * col, Q = num hits, a = Inverse-Ackermann
class BrickFallingWhenHit(object):
    def hitBrick(self, grid, hits):

        row, col = len(grid), len(grid[0])
        dsu = UnionFind(row * col + 1)

        def neighbors(x, y):
            for (nx, ny) in ((x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)):
                if 0 <= nx < row and 0 <= ny < col:
                    yield (nx, ny)

        def index(x, y):
            return x * col + y

        for (i, j) in hits:
            grid[i][j] -= 1

        for i in range(row):
            for j in range(col):
                if grid[i][j]:
                    idx = index(i, j)
                    if i == 0:
                        dsu.union(idx, row * col)

                    if i and grid[i - 1][j]:
                        dsu.union(idx, index(i-1, j))

                    if j and grid[i][j-1]:
                        dsu.union(idx, index(i, j-1))
        ans = []
        for r, c in reversed(hits):
            grid[r][c] += 1
            if not grid[r][c]:
                ans.append(0)
            else:
                idx = index(r, c)
                current_size = dsu.size[row * col]
                for x, y in neighbors(r, c):
                    dsu.union(idx, index(x, y))

                if r == 0:
                    dsu.union(idx, row * col)
                ans.append(max(0, current_size - dsu.size[row * col] - 1))
        return ans[::-1]
