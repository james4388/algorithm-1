# Longest common substring of 2 words in arrays
# Minimum edit distance
# Enumerate NxN matrix composed of words
# chess program architect
# Work break 2
# Stone picking game: 2 piles of stones A and B, each players pick 1 to K stone
# who pick last failed, check if first player can win.
# Give a board and start position check if can jump right/left to value 0
# e.g: [1, 3, 2, 0, 5, 2, 8, 2, 1, 0]
# Give a list of words and key, implement autocomplete
from collections import deque


# https://leetcode.com/problems/happy-number/description/
# 13 -> 1 + 3*3 = 10 -> 1 -> True
# 12 -> 1 + 2*2 = 5 -> 25 -> 4 + 25 = 29 -> 4 + 81 = 85 -> 64 + 25 = 89 ->
# 64 + 81 = 145 -> 1 + 16 + 25 = 42 -> 16 + 4 = 20 -> 4 -> 16 -> 37 -> 58 -> 89
def isHappy(n):
    visited = {n: True}

    while n != 1:
        val = 0
        while n > 0:
            i = n%10
            val += i*i
            n /= 10
        if val in visited:
            return False
        n = val
        visited[n] = True
    return True

# print "happy number...", isHappy(121349)


# Print matrix diagonal
# 1, 2, 3, 4
# 5, 6, 7, 8
# 9,10,11,12
# 1 -> 2 -> 5 -> 9 -> 6 -> 3 -> 4 -> 7 -> 10 -> 11 -> 8 -> 12
# 00 -> 01 -> 10 -> 20 -> 11 -> 02 -> 03 -> 12 -> 21 -> 22 -> 13 -> 23
def findDiagonalOrder(matrix):
    if not matrix:
        return []
    res = []
    m, n = len(matrix), len(matrix[0])
    r, c = 0, 0
    for i in range(m*n):
        res.append(matrix[r][c])
        # moving up
        if (r + c) % 2 == 0:
            if c == n - 1:
                r += 1
            elif r == 0:
                c += 1
            else:
                c += 1
                r -= 1
        # moving down
        else:
            if r == m - 1:
                c += 1
            elif c == 0:
                r += 1
            else:
                r += 1
                c -= 1
    return res

print "find Diagonal Order...", findDiagonalOrder([[1, 2, 3, 4], [5, 6, 7, 8],
                                                   [9, 10, 11, 12]])


# Word break 3, break strings into minimum list of words from dictionary
def wordBreak3(s, wordDict):
    if not s:
        return []

    n = len(s)
    q = deque([0])
    parent = {}
    found = False

    while q and not found:
        idx = q.popleft()
        for word in wordDict:
            if s[idx: idx+len(word)] == word:
                q.append(idx + len(word))
                if idx + len(word) not in parent:
                    parent[idx+len(word)] = idx
                if idx + len(word) == n:
                    found = True
    if found:
        print "parent...", parent
        res = []
        idx = n
        while idx != 0:
            res.append(s[parent[idx]: idx])
            idx = parent[idx]
        return res[::-1]
    return []

print "word break3...", wordBreak3('aaaismyname', ['a', 'aa', 'aaa', 'ais',
                                                   'my', 'name'])
