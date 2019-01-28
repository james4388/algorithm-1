'''
- Solve assignment problem: N salemans travel to N destination, assign each
saleman to each location to minimize travel cost
- Cost matrix NxN

Method:
- step 1: subtract element on each row to smallest element on each row
- step 2: subtract element on each column to smallest element on each column
- step 3: cover each row and column with minimal line
- step 4: test for optimality: if number of lines is N, then optimal assignment
is possible, finished. If less than N, moved to step 5
- step 5: subtract smallest entry for uncovered rows, then add to covered
columns, repeat step 3.
'''
