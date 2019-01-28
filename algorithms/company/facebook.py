from collections import defaultdict, Counter
from doubly_linkedlist import Node, DoublyLinkedlist
from heapq import heappop, heappush
from tree import (build_tree, TreeNode, inorder_tree,
                  postorder_tree, preorder_tree)
import sys
import random
import math


MAX = sys.maxint
MIN = -sys.maxint - 1


# Longest arithmetic sequence (every number has same difference)
# in same order as in array (if not same order, sort array first)
# Use 2 hashmap, mapping: num -> different -> length
def longestArithmeticSequence(nums):
    n = len(nums)
    if n <= 2:
        return n
    m = defaultdict(dict)
    res = 2

    # nums.sort()
    for i in range(1, n):
        num = nums[i]
        for j in range(i):
            cur = nums[j]
            d = num - cur
            # Find the length of current number with same difference and update
            length = max(2, m.get(cur, {}).get(d, 0) + 1)
            res = max(length, res)
            m[num][d] = length
    return res

nums = [1, 6, 3, 5, 9, 7, 8]
# print "longest arithmetic sequence....", longestArithmeticSequence(nums)


# Longest consecutive sequence (difference = 1)
# need O(n) runtime and space
# use hashmap for all numbers, for every number find upper and lower numbers
# and increase the count
def longestConsecutiveSequence(nums):
    n = len(nums)
    if n <= 2:
        return n

    res = 2
    m = {x: False for x in nums}
    for num in nums:
        # already process
        if m[num]:
            continue
        m[num] = True
        count = 1

        upper = num + 1
        while upper in m:
            m[upper] = True
            count += 1
            upper += 1

        lower = num - 1
        while lower in m:
            m[lower] = True
            count += 1
            lower -= 1
        res = max(res, count)
    return res

# print "longest consecutive sequence....", longestConsecutiveSequence(nums)


# Convert binary tree to double linkedlist inorder
def _bst(root):
    if not root:
        return None, None

    lstart, lend = _bst(root.left)
    node = Node(root.val)
    if lend:
        lend.next = node
        node.prev = lend

    rstart, rend = _bst(root.right)
    if rstart:
        node.next = rstart
        rstart.prev = node

    return (lstart or node), (rend or node)


def bstToDoubleLinkedList(root):
    if not root:
        return None

    s, e = _bst(root)
    return s

tree = build_tree(range(0, 40, 2))
head = bstToDoubleLinkedList(tree)
dll = DoublyLinkedlist()
dll.head = head
# print "bst to DLL...", dll


# Number ways of decoding integer to alphabet 'A' -> 1,
# 'B' -> 2, ..., 'Z' -> 26
# Dynamic programing: x(n) = x(n-1) + x(n-2)
# https://leetcode.com/problems/decode-ways/description/
def numDecodings(s):
    """
    :type s: str
    :rtype: int
    """
    if not s or s.startswith('0'):
        return 0

    n = len(s)
    w = [0 for x in range(n)]
    w[0] = 1
    for i in range(1, n):
        p = q = 0
        if int(s[i]) > 0:
            p = w[i-1]
        if 10 <= int(s[i-1: i+1]) <= 26:
            q = w[i-2] if i >= 2 else 1
        w[i] = p + q

    return w[n-1]


# Phone letter combination list
# https://leetcode.com/problems/letter-combinations-of-a-phone-number/description/
def dfs_combine(m, idx, string, out):
    if idx == len(m):
        out.append(string)
        return out

    for char in m[idx]:
        dfs_combine(m, idx + 1, string + char, out)


def letterCombination(digits):
    if not digits:
        return []

    string = ['', '', 'abc', 'def', 'ghi', 'jkl', 'mno', 'pqrs', 'tuv', 'wxyz']
    m = [string[int(x)] for x in digits]
    out = []

    dfs_combine(m, 0, '', out)
    return out


# Spiral 2D array, given integer return 2D array
# 123
# 894
# 765
def fill_layer(matrix, num, layer, n):
    start = layer
    _end = n - 1 - layer
    i = j = start
    if start == _end:
        matrix[i][j] = num
        return num
    while j < _end:
        matrix[i][j] = num
        num += 1
        j += 1
    while i < _end:
        matrix[i][j] = num
        num += 1
        i += 1
    while j > start:
        matrix[i][j] = num
        num += 1
        j -= 1
    while i > start:
        matrix[i][j] = num
        num += 1
        i -= 1
    return num


def spiral_array(n):
    if not n:
        return [[]]

    num = 1
    nlayer = n/2 if n % 2 == 0 else n/2 + 1
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for layer in range(nlayer):
        num = fill_layer(matrix, num, layer, n)
    return matrix

# print "spiral array...", spiral_array(1)


# Number encode 11 -> 21, 211 -> 1221
def encode(numstr):
    if not numstr:
        return ''

    prev_num = numstr[0]
    count = 1
    output = ''
    for i in range(1, len(numstr) + 1):
        if i == len(numstr):
            output += str(count) + prev_num
        elif numstr[i] != prev_num:
            output += str(count) + prev_num
            prev_num = numstr[i]
            count = 1
        else:
            count += 1
    return output


def lookSaySequence():
    seq = '1'
    while True:
        yield seq
        seq = encode(seq)

seq = lookSaySequence()
# print "look and say sequence...", [next(seq) for _ in xrange(10)]


# Longest palidrom substring
def _expand(s, i):
    p = q = i - 1
    l = 1

    while p >= 0 and q < len(s) and s[p] == s[q]:
        l += 2
        p -= 1
        q += 1
    d = 0
    p, q = i - 1, i
    while p >= 0 and q < len(s) and s[p] == s[q]:
        d += 2
        p -= 1
        q += 1
    return max(l, d)


def longestPalindrom(s):
    if not s:
        return 0
    n = len(s)
    if n <= 1:
        return n
    res = 1
    for i in range(1, n):
        l = _expand(s, i)
        res = max(res, l)
    return res


# print "longest palindrom...", longestPalindrom("abbadabeebadt")


# Remove invalid parentheses
# Brute force, BFS, remove one parenthese at each position and check is valid
# If valid put in result items from queue, if not repeat to next level
# runtime 0(n*2^n)
# DFS: count open and close, if number of close more than open, we remove
# random, but not consecutive bracket. Do same for open parenthese but from
# right to left, trick: reverse string and reuse same code
# https://leetcode.com/problems/remove-invalid-parentheses/description/
def _remove(s, p):
    out = [s]
    count = 0
    res = []
    for i in range(len(s)):
        if s[i] == p[0]:
            count += 1
        if s[i] == p[1]:
            count -= 1

        if count < 0:
            j = i
            n = []
            for item in out:
                for j in range(0, i+1):
                    if item[j] == p[1] and (j == 0 or item[j-1] != p[1]):
                        n.append(item[:j] + ' ' + item[j+1:])
            out = n
            count = 0
    if count > 0 and p[0] == '(':
        re = [_remove(x[::-1], [')', '(']) for x in out]
        res = [x[::-1] for r in re for x in r]
    else:
        res = out

    res = [x.replace(' ', '') for x in res]
    return list(set(res))


def removeInvalidParentheses(s):
    return _remove(s, ['(', ')'])


# Second solution
def dfs_parentheses(s, res, p, istart, jstart):
    count = 0
    for i in range(istart, len(s)):
        if s[i] == p[0]:
            count += 1
        if s[i] == p[1]:
            count -= 1

        if count < 0:
            for j in range(jstart, i+1):
                if s[j] == p[1] and (j == jstart or s[j-1] != p[1]):
                    ns = s[:j] + s[j+1:]
                    dfs_parentheses(ns, res, p, i, j)
            return
    reverse = s[::-1]
    if p[0] == '(':
        dfs_parentheses(reverse, res, [')', '('], 0, 0)
    else:
        res.append(reverse)


def removeInvalidParentheses2(s):
    res = []
    dfs_parentheses(s, res, ['(', ')'], 0, 0)
    return res


# print "balance bracket...", removeInvalidParentheses2("(r(()()(")


# Regex pattern matching:
# https://leetcode.com/problems/regular-expression-matching/description/
# DP solutions:
# s[i] = p[j] or p[j] = '.': dp[i][j] = dp[i-1][j-1]
# p[j] = '*': 2 cases
# p[j-1] != s[i]: dp[i][j] = dp[i][j-2]
# p[j-1] == s[i] or p[j-1] = '.':
# dp[i][j] = dp[i-1][j] (match multiple)
# or dp[i][j-1] (single match)
# or dp[i][j-2] (empty)

def isMatch(s, p):
    dp = [[False for _ in range(len(p) + 1)] for _ in range(len(s) + 1)]

    dp[0][0] = True
    for j in range(2, len(p) + 1):
        dp[0][j] = dp[0][j-2] and (p[j-1] == '*')

    for i in range(1, len(s)+1):
        for j in range(1, len(p)+1):
            if s[i-1] == p[j-1] or p[j-1] == '.':
                dp[i][j] = dp[i-1][j-1]

            if p[j-1] == '*':
                dp[i][j] = dp[i][j-1] or dp[i][j-2]

                if p[j-2] == s[i-1] or p[j-2] == '.':
                    dp[i][j] |= dp[i-1][j]
    return dp[-1][-1]


# Follow up: reduce space
def isMatch2(s, p):
    dp = [False for _ in range(len(p) + 1)]
    dp[0] = True

    for j in range(2, len(p) + 1):
        if p[j - 1] == '*':
            dp[j] = dp[j-2]

    for i in range(1, len(s) + 1):
        pre = dp[0]
        for j in range(1, len(p) + 1):
            tmp = dp[j]
            if s[i-1] == p[j-1] or p[j-1] == '.':
                dp[j] = pre and dp[j-1]
            if p[j-1] == '*':
                dp[j] = dp[j-1] or dp[j-2]

                if p[j-2] == s[i-1] or p[j-2] == '.':
                    dp[j] |= pre
            pre = tmp
    return dp[-1]


print "is Match...", isMatch2('abc', 'a.*')


# min window string contains all string
# https://leetcode.com/problems/minimum-window-substring/description/
# Running low and high pointer, count number of character cover
# if found window, increase low pointer to shorten window size
def minWindow(s, t):
    c = Counter(t)
    need = len(t)
    n = len(s)
    lo, hi = 0, 0
    mstart, mend = -1, n
    mwidth = MAX

    while hi < n:
        char = s[hi]
        if char in c:
            if c[char] > 0:
                need -= 1
            c[char] -= 1
        # found window
        while need == 0:
            width = hi - lo + 1
            if mwidth > width:
                mwidth = width
                mstart, mend = lo, hi

            mchar = s[lo]

            if mchar in c:
                c[mchar] += 1
                if c[mchar] > 0:
                    need += 1
            lo += 1

        hi += 1

    if mstart != -1:
        return s[mstart: mend+1]
    return ""

# print "minimum window....", minWindow("ADOBECODEBANC", "ABC")


# Wordbreak 2
# memorize the breaking location then using dfs to return list of string
# https://leetcode.com/problems/word-break-ii/description/
def wordBreak1(s, wordDict):
    n = len(s)
    res = [False for x in range(n)]
    for i in range(n):
        for word in wordDict:
            l = len(word)
            if i >= l-1:
                if s[i-l + 1: i+1] == word and ((i-l < 0) or res[i-l]):
                    res[i] = True
    return res[-1]

# print "word break...", wordBreak('bbb', ['b', 'bb', 'bbb'])


def dfs_wb(s, arr, res, idx, text):
    if idx == -1:
        res.append(text[:-1])
        return

    for prev in arr[idx]:
        dfs_wb(s, arr, res, prev, s[prev+1: idx+1] + ' ' + text)


def wordBreak2(s, wordDict):
    if not s:
        return []
    n = len(s)
    arr = [[] for _ in range(n)]

    for i in xrange(n):
        for word in wordDict:
            l = len(word)
            if s[i-l+1: i+1] == word and (i - l < 0 or arr[i-l]):
                arr[i].append(i-l)
    if not arr[n-1]:
        return []
    res = []
    dfs_wb(s, arr, res, n-1, '')
    return res

print "wordBreak2....", wordBreak2('catsanddog',
                                   ['cat', 'cats', 'and', 'sand', 'dog'])


# find consecutive number equals to k
# use hash to store previous prefix sum, everytime calculate prefix sum,
# look into hash table to find prefix - k
# https://leetcode.com/problems/subarray-sum-equals-k/description/
def subarraySum(self, nums, k):
    if not nums:
        return 0

    m = defaultdict(int)
    m[0] = 1
    s = 0
    res = 0

    for num in nums:
        s += num
        res += m.get(s-k, 0)
        m[s] += 1
    return res


# Move zeroes, 2 index running, assign low index to non-zero values
# assign the rest numbers to zeroes
# https://leetcode.com/submissions/detail/160603738/
def moveZeroes(nums):
    """
    :type nums: List[int]
    :rtype: void Do not return anything, modify nums in-place instead.
    """
    n = len(nums)
    idx = 0
    for i in range(n):
        if nums[i] != 0:
            nums[idx] = nums[i]
            idx += 1
    while idx < n:
        nums[idx] = 0
        idx += 1


# Monotonic numbers sequence, non-decreasing sequence
# Find number that smaller than its later number, decrease 1
# assign the rest to 9
# https://leetcode.com/problems/monotone-increasing-digits/description/
def monotoneIncreasingDigits(N):
    if not N or N < 10:
        return N

    k = map(int, str(N))
    idx = len(k)
    for i in range(len(k)-1, 0, -1):
        if k[i] < k[i-1]:
            k[i-1] -= 1
            idx = i
    for j in range(idx, len(k)):
        k[j] = 9

    return int(''.join(map(str, k)))

print "monotonic...", monotoneIncreasingDigits(12343)


# Find the greatest number less than current number with same digits
# find number that larger than number after it, swap it with closest number
# from the rest, and sort the rest in decreasing order
# e.g 1876|7569 -> 1876|6975
def nextNumber(num):
    if not num or num < 10:
        return None

    k = map(int, str(num))
    for i in range(len(k)-1, 0, -1):
        if k[i] < k[i-1]:
            break
    # number in increasing order
    if i == 0:
        return None
    h, l = i - 1, i
    # Find next number closest large to swap number
    for j in range(i+1, len(k)):
        if k[l] < k[j] < k[h]:
            l = k[j]

    k[h], k[l] = k[l], k[h]
    k = k[:i] + sorted(k[i:], reverse=True)
    return int(''.join(map(str, k)))

print "nextNumber...", nextNumber(18767569)


# Find if word exists in 2D array in horizontal or vertical
# Use backtracking technique, for every char if it's matched with first char
# then search from there
# mark cell to avoid search again -> move and unmake move
class SearchSolution():
    def _exist(self, board, word, idx, i, j):
        if board[i][j] != word[idx]:
            return False

        if idx == len(word) - 1:
            return True

        # Generate candidates
        candidates = []
        if i > 0:
            candidates.append((i-1, j))
        if j > 0:
            candidates.append((i, j-1))
        if i < len(board)-1:
            candidates.append((i+1, j))
        if j < len(board[0])-1:
            candidates.append((i, j+1))

        tmp = board[i][j]
        # move
        board[i][j] = ''

        for item in candidates:
            if self._exist(board, word, idx+1, item[0], item[1]):
                return True
        # unmove
        board[i][j] = tmp
        return False

    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        for i in range(len(board)):
            for j in range(len(board[0])):
                if self._exist(board, word, 0, i, j):
                    return True
        return False


# https://leetcode.com/problems/kth-largest-element-in-an-array/description/
class KLargestSolution(object):
    def partition(self, nums):
        pivot = nums[-1]
        i = 0
        for j in range(len(nums)):
            if nums[j] > pivot:
                nums[j], nums[i] = nums[i], nums[j]
                i += 1
        nums[i], nums[-1] = nums[-1], nums[i]
        return i

    def quickSelect(self, nums, k):
        idx = self.partition(nums)
        if k == idx + 1:
            return nums[idx]
        elif k > idx + 1:
            k = k - idx - 1
            return self.quickSelect(nums[idx+1:], k)
        else:
            return self.quickSelect(nums[:idx], k)

    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        return self.quickSelect(nums, k)


# Build binary search tree from pre-order and post order arrays
# First number of pre-order is root node, search root in  in-order array
# Recursively build left: pre-order (1: idx+1), in-order(0:idx)
# Build right: pre-order (idx+1: end), in-order (idx+1: end)
# https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/
def buildTree(self, preorder, inorder):
    """
    :type preorder: List[int]
    :type inorder: List[int]
    :rtype: TreeNode
    """
    if not preorder:
        return None
    if len(preorder) == 1:
        return TreeNode(preorder[0])

    node = TreeNode(preorder[0])
    idx = inorder.index(preorder[0])
    node.left = buildTree(preorder[1: idx+1], inorder[:idx])
    node.right = buildTree(preorder[idx+1:], inorder[idx+1:])


# First bad version
# Use binary search, middle m, check if m is bad and m-1 not bad, then return m
# otherwise, m is bad, r = m - 1, m is not bad, l = m + 1
# https://leetcode.com/problems/first-bad-version/description/
def isBadVersion(n):
    i = random.randint(n)
    return n % i == 0


def firstBadVersion(n):
    if n <= 0:
        return -1
    l = 1
    r = n
    while l <= r:
        m = (l+r)/2
        if isBadVersion(m):
            r = m
        else:
            l = m + 1
    return l


# Find celebrity in N people, she does not know anyone, everybody knows her
# Minimize number of call to haveAcquaintance(A, B)
# Use stack, everytime pop 2 person, if know than, remove the other
CELEB = 3


def haveAcquaintance(a, b):
    if a == CELEB:
        return False
    if b == CELEB:
        return True
    return random.randint(0, 1) == 1


def findCelebrity(N):
    stack = range(N)

    while len(stack) > 1:
        a, b = stack.pop(), stack.pop()
        if haveAcquaintance(a, b):
            stack.push(b)
        else:
            stack.push(a)

    celeb = stack.pop()
    for i in range(N):
        if (i != celeb and not haveAcquaintance(i, celeb) and
            haveAcquaintance(celeb, i)):
            return -1
    return celeb


# Knight probability problem, find probability knight
# stay on board NxN after K moves
# https://leetcode.com/problems/knight-probability-in-chessboard/description/
class KnightSolution:
    def knightProbability(self, N, K, r, c):
        dp = [[1 for _ in range(N)] for _ in range(N)]
        dir_row = [2, 2, -2, -2, 1, -1, 1, -1]
        dir_col = [1, -1, 1, -1, 2, 2, -2, -2]
        ndir = 8

        for step in range(K):
            s = [[0 for _ in range(N)] for _ in range(N)]
            for i in range(N):
                for j in range(N):
                    for d in range(ndir):
                        x = i + dir_row[d]
                        y = j + dir_col[d]
                        if 0 <= x < N and 0 <= y < N:
                            s[i][j] += dp[x][y]
            dp = s
        return dp[r][c] / float(math.pow(8, K))

print "knight probability...", KnightSolution().knightProbability(3, 2, 0, 0)


# Knight tour problem
# each knight has 8 moves, for every position check position with minimal
# degree to move
class KnightTourSolution:
    dir_row = [2, 2, -2, -2, 1, -1, 1, -1]
    dir_col = [1, -1, 1, -1, 2, 2, -2, -2]
    ndir = 8

    def get_degree(self, board, N, x, y):
        count = 0
        for k in range(self.ndir):
            nx = x + self.dir_row[k]
            ny = y + self.dir_col[k]
            if 0 <= nx < N and 0 <= ny < N and board[nx][ny] == -1:
                count += 1
        return count

    def move(self, board, N, x, y, step):
        _min = self.ndir + 1
        i, j = -1, -1
        for k in range(self.ndir):
            nx = x + self.dir_row[k]
            ny = y + self.dir_col[k]
            if 0 <= nx < N and 0 <= ny < N and board[nx][ny] == -1:
                d = self.get_degree(board, N, nx, ny)
                if d < _min:
                    _min = d
                    i, j = nx, ny
        if i != -1:
            board[i][j] = step
        return i, j

    def knightTour(self, N, r, c):
        board = [[-1 for _ in range(N)] for _ in range(N)]
        board[r][c] = 0
        x, y = r, c

        for i in range(1, N*N):
            x, y = self.move(board, N, x, y, i)
            if x == -1:
                return None
        # Check if closed tour
        for k in range(self.ndir):
            if (x + self.dir_row[k] == r) and (y + self.dir_col[k] == c):
                return board
        return None

print "knight tour...\n", KnightTourSolution().knightTour(8, 0, 0)


# Longest path in tree
# https://leetcode.com/problems/diameter-of-binary-tree/description/
class TreeSolution:
    def _path(self, node):
        if not node:
            return 0

        l = self._path(node.left)
        r = self._path(node.right)
        self.ans = max(self.ans, l + r + 1)
        return max(l, r) + 1

    def longestPathTree(self, root):
        if not root:
            return 0
        self.ans = 1
        self._path(root)
        return self.ans - 1


def addTwoBinary(num1, num2):
    l = (num1, num2) if len(num1) >= len(num2) else (num2, num1)
    p, q = l
    # Padding '0' to smaller number
    q = (len(p) - len(q)) * '0' + q
    a, b = map(int, list(p)), map(int, list(q))
    out = ''
    carry = 0

    for i in range(len(a) - 1, -1, -1):
        s = a[i] + b[i] + carry
        out = str(s % 2) + out
        carry = s / 2

    if carry:
        out = str(carry) + out
    return out

print "add two binary...", addTwoBinary('10011', '1110')


# Find k closet points to 0, 0
class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __cmp__(self, other):
        return cmp(self.x * self.x + self.y * self.y,
                   other.x * other.x + other.y * other.y)

    def __str__(self):
        return '{}.{}'.format(self.x, self.y)


def partition(arr, start, _end):
    pivot = arr[_end]
    idx = start
    for j in range(start, _end+1):
        if arr[j] < pivot:
            arr[j], arr[idx] = arr[idx], arr[j]
            idx += 1
    arr[idx], arr[_end] = arr[_end], arr[idx]
    return idx


def _kClosest(arr, k, start, _end):
    idx = partition(arr, start, _end)
    if k == idx + 1:
        return arr[:idx+1]
    elif k < idx + 1:
        return _kClosest(arr, k, start, idx-1)
    else:
        return _kClosest(arr, k, idx+1, _end)


def kClosest(arr, k):
    return _kClosest(arr, k, 0, len(arr) - 1)

arr = kClosest([Point(3, 4), Point(1, 1), Point(0, 1),
                Point(-1, -2), Point(10, 3)], 2)
print "k closest points...", [str(x) for x in arr]


# Find the maximum overlap number in the interval list
def maximumOverlap(arr):
    points = [x[0] for x in arr]
    _min, _max = min(points), max(points)

    dp = [0 for _ in range(_min, _max+1)]

    for (x, y) in arr:
        dp[x - _min] = 1
        if y + 1 <= _max:
            dp[y+1 - _min] = -1

    count = 0
    res = 0
    idx = -1
    for j in range(_max - _min + 1):
        count += dp[j]
        if count > res:
            res = count
            idx = j
    return idx + _min

print "maximum overlap...", maximumOverlap([(1, 5), (3, 9), (6, 12), (0, 8),
                                            (7, 20), (10, 24)])
