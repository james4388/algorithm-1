

# Number of islands
# https://leetcode.com/problems/number-of-islands/description/
def numIslands(self, grid):
    """
    :type grid: List[List[str]]
    :rtype: int
    """
    def _bfs(self, grid, i, j):
        if (i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or
            grid[i][j] == '0'):
            return
        grid[i][j] = '0'
        _bfs(grid, i+1, j)
        _bfs(grid, i-1, j)
        _bfs(grid, i, j+1)
        _bfs(grid, i, j-1)

    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                count += 1
                _bfs(grid, i, j)

    return count
