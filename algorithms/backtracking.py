
def generate_candidate(text, k, n):
    return text[k+1:n]


class Sudoku():
    DIM = 9

    def find_unassigned_box(self, grid):
        for x in range(self.DIM):
            for y in range(self.DIM):
                if grid[x][y] == 0:
                    return True, (x, y)
        return False, None

    def is_in_row(self, grid, row, num):
        for y in range(self.DIM):
            if grid[row][y] == num:
                return True
        return False

    def is_in_col(self, grid, col, num):
        for x in range(self.DIM):
            if grid[x][col] == num:
                return True
        return False

    def is_in_subgrid(self, grid, row, col, num):
        sg_row = row - row % 3
        sg_col = col - col % 3
        for i in range(sg_row, sg_row+3):
            for j in range(sg_col, sg_col+3):
                if grid[i][j] == num:
                    return True
        return False

    def is_safe(self, grid, row, col, num):
        return (not self.is_in_row(grid, row, num) and
                not self.is_in_col(grid, col, num) and
                not self.is_in_subgrid(grid, row, col, num))

    def solve_sudoku(self, grid):
        found, point = self.find_unassigned_box(grid)
        if not found:
            return True

        x, y = point

        for num in range(1, self.DIM+1):
            if self.is_safe(grid, x, y, num):
                # assign to cell
                grid[x][y] = num
                if self.solve_sudoku(grid):
                    return True
                # unassign cell
                grid[x][y] = 0
        return False

    def print_sudoku(self, grid):
        for x in range(self.DIM):
            print grid[x]


# sdk = Sudoku()
# grid=[[3,0,6,5,0,8,4,0,0],
#       [5,2,0,0,0,0,0,0,0],
#       [0,8,7,0,0,0,0,3,1],
#       [0,0,3,0,1,0,0,8,0],
#       [9,0,0,8,6,3,0,0,5],
#       [0,5,0,0,9,0,6,0,0],
#       [1,3,0,0,0,0,2,5,0],
#       [0,0,0,0,0,0,0,7,4],
#       [0,0,5,2,0,6,3,0,0]]
#
# if sdk.solve_sudoku(grid):
#     sdk.print_sudoku(grid)
# else:
#     print "Cannot solve"


# Match text by pattern
# try every substring from current position to end
# use map to store current substr for pattern
# support * and . wildcard
def match(pattern, text, i, k, j, n, store):
    if (i == k) and (j == n):
        return True

    if (i == k) or (j == n):
        return False

    p = pattern[i]
    if p == '.':
        return match(pattern, text, i+1, k, j+1, n, store)

    if p == '*':
        for x in range(j, n):
            if match(pattern, text, i+1, k, x, n, store):
                return True

    val = store.get(p, '')
    l = len(val)
    if p in pattern[:i]:
        if text[j: j+l] != val:
            return False
        return match(pattern, text, i+1, k, j+l, n, store)

    for x in range(1, n - l):
        val = text[j: j+x]
        store[p] = val
        if match(pattern, text, i+1, k, j+x, n, store):
            return True
        store.pop(p)

    return False


def pattern_matching(pattern, text):
    m = {}
    if match(pattern, text, 0, len(pattern), 0, len(text), m):
        print "Pattern found...", m
    else:
        print "No pattern found..."

# pattern_matching('a*a..', 'GraphTreeGraphCo')


# Generate all subsets of set
# iteractive: there're 2^n subset, check bit in mark and print each subset
# backtracking: for each element in set, it can either include
# or exclude to set
def _generate(arr, sub, idx, res):
    for i in range(idx, len(arr)):
        res.append(sub + [arr[i]])
        _generate(arr, sub + [arr[i]], i+1, res)
    return


def generateSubset(arr):
    res = []
    _generate(arr, [], 0, res)
    return res

print "generate subset...", generateSubset([1, 2, 3])


# Permutation of a string or array
# there're n! permutation: for position 0 there n value
# swap value at position 0 and do permutation for the rest n - 1 value
def _permutate(s, l, r, res):
    if l == r:
        res.append(''.join(s))
        return
    for i in range(l, r+1):
        s[i], s[l] = s[l], s[i]
        _permutate(s, l+1, r, res)
        s[i], s[l] = s[l], s[i]


def permutate(s):
    res = []
    _permutate(list(s), 0, len(s) - 1, res)
    return res

print "permutate...", permutate('abcd')
