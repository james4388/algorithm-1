'''
499)
- The maze 3
- HARD: https://leetcode.com/problems/the-maze-iii/
- Ball can only change direction if it hits the wall
- Use BFS: check for possible directions of balls, add into queue
 (x, y, direction, step), pop an item from queue, then move it in direction
until next wall and increase step, select next direction so that ball not going
back, while moving ball if it drop into hole remove it and compare number of
step, if less than current or equal and smaller lexicographically
'''
from collections import deque


class Solution:
    def shortestDistance(maze, x, y, dest_x, dest_y):
        queue = deque()
        h, w = len(maze), len(maze[0])

        distance = [[float('inf')] * w for _ in range(h)]
        distance[x][y] = 0
        queue.add((x, y))
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while queue:
            i, j = queue.popleft()
            for dx, dy in dirs:
                x, y = i, j
                count = 0

                while 0 <= x < h and 0 <= y < w and maze[x][y] == 0:
                    x += dx
                    y += dy
                    count += 1
                x -= dx
                y -= dy
                count -= 1
                if distance[i][j] + count < distance[x][y]:
                    distance[x][y] = distance[i][j] + count
                    queue.append((x, y))

        return distance[dest_x][dest_y] if distance[dest_x][dest_y] < float('inf') else -1
