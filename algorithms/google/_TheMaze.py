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
