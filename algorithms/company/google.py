from collections import defaultdict, deque, Counter
import random
from random import randint
import itertools
from heapq import *


# Painter problems
# 2 painter with 4 boards length 10, 20, 30, 40. Find minimum unit of time
# to paint given each painter paints continous boards
# C(n, k) is solution
# C(i, 1) = sum(x0..xi)
# C(1, j) = x1
# C(i, j) = max (C(q, j - 1), sum xq...xi)
def painter(boards, k):
    n = len(boards)
    prefix = [0 for x in range(n+1)]
    for i in range(1, n+1):
        prefix[i] = prefix[i-1] + boards[i-1]

    _max = prefix[n]
    dp = [[_max] * (k+1) for x in range(n+1)]

    for i in range(1, n+1):
        dp[i][1] = prefix[i]

    for j in range(1, k+1):
        dp[1][j] = prefix[1]

    for i in range(2, n+1):
        for j in range(2, k+1):
            for x in range(1, i):
                cost = max(dp[x][j-1], prefix[i] - prefix[x])
                if dp[i][j] > cost:
                    dp[i][j] = cost
    return dp[n][k]

print("painter...\n", painter([10, 20, 30, 40], 2))


# Max absolute different of 2 sub array
# Use kaden algorithm finding maximum subarray
# the minimum subarray will locate either left or right side of max subarray
# Reuse kaden algorithm for min subarray by reverting number sign
# Return low and high index of subarray instead
# Calculate prefix sum for quick finding sum
def findMaxSum(nums):
    max_so_far = 0
    max_until = 0
    x, y = 0, 0
    tmpx, tmpy = 0, 0
    for idx, num in enumerate(nums):
        if max_until == 0:
            tmpx = idx

        max_until += num
        tmpy = idx
        if max_so_far < max_until:
            max_so_far = max_until
            x, y = tmpx, tmpy
        if max_until < 0:
            max_until = 0
    return x, y


# Maximum circular subarray
# Find maximum subarray, compare it with circular subarray
# Find circular by find minimum index subarray, circular is sum of left and
# right of min subarray
# Edge case, array all negative numbers, then only return maximum subarray
# https://leetcode.com/contest/weekly-contest-105/problems/maximum-sum-circular-subarray/
def maxSubarraySumCircular(A):
    """
    :type A: List[int]
    :rtype: int
    """
    if not A:
        return 0
    n = len(A)
    arr = [-x for x in A]
    idx, idy = findMaxSum(arr)
    s, e = findMaxSum(A)
    s1 = sum(A[:idx])
    s2 = sum(A[idy+1:])
    s3 = sum(A[s:e+1])
    # all negative
    if idx == 0 and idy == n-1:
        return s3
    return max(s1, s2, s1+s2, s3)


def maxAbsolute(nums):
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    n = len(nums)
    prefix = nums[:]
    for i in range(1, n):
        prefix[i] += prefix[i-1]

    p, q = findMaxSum(nums)
    x, y = findMaxSum([-val for val in nums[:p]])
    u, v = findMaxSum([-val for val in nums[q+1:]])
    u, v = u + q + 1, v + q + 1
    vleft = abs(prefix[q] - prefix[q] - prefix[y] + prefix[x])
    vright = abs(prefix[q] - prefix[q] - prefix[v] + prefix[u])
    if vleft > vright:
        return nums[p: q+1], nums[x: y+1]
    else:
        return nums[p: q+1], nums[u: v+1]

print("max absolute...", maxAbsolute([1, 2, -4, 4, 3, 2, -5]))


# Min cut square, A > B, each time cut rectangle to square by equal (B, B) side
# number of cut = A//B
def minCutSquare(A, B):
    if A <= 0 or B <= 0:
        return -1

    if A == B:
        return 1
    if B > A:
        A, B = B, A
    res = 0
    while A > 0 and B > 0:
        v = A // B
        res += v
        A, B = B, A - v*B
    return res

print("min cut square...", minCutSquare(13, 385))


# Find quadruple numbers in array that sum to K
# Use a map to store sum of 2 elements, each time we have sum
# look up map for sum value
def findQuadruple(nums, k):
    if not nums or len(nums) < 4:
        return []

    arr = sorted(nums)
    sumMap = defaultdict(list)
    n = len(nums)
    res = set()

    for i in range(n-1):
        for j in range(i+1, n):
            s = arr[i] + arr[j]
            d = k - s
            if d in sumMap:
                for couple in sumMap[d]:
                    if not couple & {i, j}:
                        u, v = couple
                        res.add(tuple(sorted([arr[u], arr[v], arr[i], arr[j]])))
            sumMap[s].append({i, j})
    return res

print("Find quadruple...", findQuadruple([10, 2, 3, 4, 5, 9, 7, 8], 23))


# Form a palindrome, find minimum number of character to insert for string
# become palindrome
# def formPalindrome
# Solution: transform to longest palindrome subsequence
# num char = length - longest subsequence


# Wiggle sort or zig zag number order
# For each number i: nums[i] > nums[i-1] => it's up, previous need to be down
# nums[i] < nums[i-1] => it's down side, previous need to be up
# https://leetcode.com/problems/wiggle-subsequence/description/
# Use 2 vars up and down to count sequence length
def wiggleMaxLength(nums):
    if not nums:
        return 0

    n = len(nums)
    if n < 2:
        return n

    prev_up = 1
    prev_down = 1

    for i in range(1, n):
        if nums[i] > nums[i-1]:
            prev_up = max(prev_up, prev_down + 1)
        elif nums[i] < nums[i-1]:
            prev_down = max(prev_down, prev_up + 1)

    return max(prev_up, prev_down)


# Sort numbers so that nums[0] <= nums[1] >= nums[2] <= nums[3] ...
# Tranverse array, compare odd pos number to its previous even pos number,
# and next even position number, swap them
def wiggleSort(nums):
    if not nums or len(nums) <= 1:
        return None

    n = len(nums)
    for i in range(1, n, 2):
        if nums[i] < nums[i-1]:
            nums[i], nums[i-1] = nums[i-1], nums[i]
        if i + 1 < n and nums[i] < nums[i+1]:
            nums[i], nums[i+1] = nums[i+1], nums[i]
    return nums


# Smallest partition array that has equal sum
# Use prefix sum to quick calculate sum
# Try from i = 0, run pointer j from i + 1, if find equal sum until end
# return it
# e.g [1, 3, 2, 2, 3, 1] has 2 partition with equal sum
# [1, 3, 2], [2, 3, 1] and [1, 3], [2, 2], [3, 1] -> return 4
def smallestPartition(nums):
    if not nums:
        return 0

    prefix = nums[:]
    n = len(nums)
    for i in range(1, n):
        prefix[i] += prefix[i-1]

    for i in range(n):
        s = prefix[i]

        for j in range(i+1, n):
            ns = prefix[j]
            if ns - s == prefix[i]:
                s = ns
                if j == n - 1:
                    return prefix[i]
    return prefix[n-1]


# Sort an array contains value from 1, n and space (999) by using swap function
# loop from 0, to n-1
# find index of space
# if val at index i = i + 1 or space, continue
# check value at index val - 1 = nums[val - 1] = k
# if k = space, swap val and k
# swap k to space, then swap val to k
def sortSwap(nums):
    if not nums or len(nums) <= 1:
        return

    n = len(nums)
    idx = nums.index(999)

    for i in range(n):
        val = nums[i]
        if val == i+1 or val == 999:
            continue
        k = nums[val - 1]
        if k != 999:
            nums[val - 1], nums[idx] = nums[idx], nums[val - 1]
        nums[i], nums[val - 1] = nums[val - 1], nums[i]
        idx = i
    return nums


# Find rectangle with 4 corners are 1
# for each line store a pair of column with 1, 1 in set
# check if there's existing pairs
def findRectangle(matrix):
    if not matrix:
        return False
    pairDict = {}

    for i in range(len(matrix)):
        cols = []
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                if not cols:
                    cols.append(j)
                    continue
                for index in cols:
                    if (index, j) in pairDict:
                        return True
                    pairDict[(index, j)] = True
    return False


# Number of ways to remove 0 or more characters to transform one
# string to another string
# e.g: abcccdf -> abccdf = 3, aabba -> ab = 4
# Call dp[m][n] = result
# sequence x1..xi, y1...yj
# dp[0][j] = 1, deleting all characters in y
# xi != yj, dp[i][j] = dp[i][j-1] match sequence y1...y(j-1)
# xi == yj, can match either sequence y1...y(j-1) to x1...x(i-1)
# or match y1...y(j-1) to x1...xi
def numWayTransform(x, y):
    if not x:
        return 1

    m, n = len(x), len(y)
    dp = [[0 for i in range(n)] for j in range(m)]

    for i in range(0, m):
        for j in range(i, n):
            if i == 0:
                if x[i] == y[j] and j == 0:
                    dp[i][j] = 1
                elif x[i] == y[j]:
                    dp[i][j] = dp[i][j-1] + 1
                else:
                    dp[i][j] = dp[i][j-1]
            else:
                if x[i] != y[j]:
                    dp[i][j] = dp[i][j-1]
                else:
                    dp[i][j] = dp[i-1][j-1] + dp[i][j-1]

    return dp[m-1][n-1]


# Abbreviation of a string
def bsabbr(arr, idx, s, res):
    if idx == len(s):
        res.append(''.join(arr))
        return

    char = s[idx]
    l = arr[:]
    if l and not l[-1].isalpha():
        l[-1] = str(int(l[-1]) + 1)
    else:
        l.append('1')
    bsabbr(l, idx+1, s, res)
    arr.append(char)
    bsabbr(arr, idx+1, s, res)


def abbreviate(t):
    res = []
    if not t:
        return res

    bsabbr([], 0, t, res)
    return res


# Regex matching + for single character or empty, * for any sequence
# e.g pattern: b+a*ab -> baaabab: True
# C(i, j) is result of string at index i, pattern at index j
# yj = '+' -> C(i, j) = C(i-1, j-1) for single match or
# C(i, j-1) for empty match
# yj = '*' -> C(i, j) = C(i, j-1) for empty sequence or C(i-1, j)
# for multiple match
# Edge case: string is empty, pattern not empty: '' and '+*'
# pattern is empty, string is not empty
def regexMatching(pattern, text):
    if not pattern and not text:
        return True

    m, n = len(text), len(pattern)
    dp = [[False] * (n+1) for _ in range(m+1)]

    dp[0][0] = True

    # edge case string is empty, pattern is not empty
    for j in range(1, n+1):
        if pattern[j-1] in ('*', '+'):
            dp[0][j] = dp[0][j-1]

    for i in range(1, m+1):
        for j in range(1, n+1):
            if pattern[j-1] == '+':
                dp[i][j] = dp[i-1][j-1] or dp[i][j-1]
            elif pattern[j-1] == '*':
                dp[i][j] = dp[i][j-1] or dp[i-1][j]
            elif text[i-1] == pattern[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = False
    return dp[m][n]


#
# Use dictionary, or hashtable to store previous number, check if new number
# divisible to 2 or 5 and the remainder is in set
def product(N):
    if N <= 0:
        return []

    s = {1}

    for i in range(2, N+1):
        if (i % 2 == 0 and (i/2) in s) or (i % 5 == 0 and (i/5) in s):
            s.add(i)
    return list(s)


#
def conflictMeeting(arr):
    if not arr:
        return []

    res = []
    l = [arr[0]]
    arr.sort(key=lambda x: x[0])

    end = arr[0][1]
    for i in range(1, len(arr)):
        if arr[i][0] <= end:
            l.append(arr[i])
        else:
            if len(l) > 1:
                res.append(l)
            l = [arr[i]]
        end = max(end, arr[i][1])
    if len(l) > 1:
        res.append(l)
    return res


# Decompress string
# similar to https://leetcode.com/problems/decode-string/description/
# a(b(c){2}){2}(b){2}d -> abccbccbbd, ((x){3}(y){2}z){2} -> xxxyyzxxxyyz
# Use stack, push string from right, if {, push current number to stack
# if } push current string to stack, if (, pop from stack, if string concat it
# if number, multiple it and push back to stack, break, if 0 concat
# all current string
# Second solution: use recursive approach, with times = 1
def decompress(pattern):
    if not pattern:
        return ''

    stack = []
    n = len(pattern)
    char_buffer = ''
    res = ''
    i = n-1
    while i >= 0:
        char = pattern[i]
        if char == '}':
            if char_buffer:
                stack.append(char_buffer)
                char_buffer = ''
            j = i
            while pattern[i] != '{':
                i -= 1
            stack.append(int(pattern[i+1:j]))
        elif char == ')':
            char_buffer = ''
        elif char == '(':
            while stack:
                val = stack.pop()
                if isinstance(val, int):
                    char_buffer = char_buffer * val
                    break
                else:
                    char_buffer = char_buffer + val
            stack.append(char_buffer)
            char_buffer = ''
        else:
            char_buffer = char + char_buffer
        i -= 1
    stack.append(char_buffer)
    while stack:
        res += stack.pop()
    return res


# Find duplicate subtree in tree
# Use serialize tree inorder, use hashmap to store subtree
class TreeSolution(object):
    def serialize(self, root):
        if not root:
            return ''

        left = self.serialize(root.left)
        right = self.serialize(root.right)
        subtree = left + str(root.val) + right
        if len(subtree) > 1 and subtree in self.table:
            self.result = True
        return subtree

    def is_duplicate(self, root):
        self.table = {}
        self.result = False
        self.serialize(root)
        return self.result


# Count BST nodes that lie in range
# if node value <= i, count from right
# if node value > j, count from left
# count from i, val in left and val + 1, j in right
def getNodeCount(node, l, r):
    if not node:
        return 0

    if node.val <= l:
        return getNodeCount(node.right, l, r)

    if node.val >= r:
        return getNodeCount(node.left, l, r)

    return (getNodeCount(node.left, l, node.val) +
            getNodeCount(node.right, node.val+1, r))


# Special keyboard: key 1 => 'A', key 2 => Ctrl + A, key 3 => Ctrl + C
# key 4: Ctrl + V, can do multiple copy key4
# C(N) is result, C(0) = 0, C(1) = 1, C(i) = max(C(i-1) + 1, C(i-3) * 2)
# Consider user can copy multiple times, use `copy` array to cache the string
# if at happen at index i-1, then can copy again: val = dp[i-1] + copy[i-1]
# copy[i] = copy[i-1]
# if not then, have to issue 3 key: copy[i] = copy[i-3], val = dp[i-3] * 2
def specialKeyboard(N):
    if N <= 0:
        return 0

    dp = [0 for _ in range(N+1)]
    dp[1] = 1
    copy = 1
    for i in range(2, N+1):
        if i < 3:
            dp[i] = dp[i-1] + 1
        else:
            x = dp[i-1] + 1
            y = dp[i-1] + copy
            z = dp[i-3] * 2
            dp[i] = max(x, y, z)
            if dp[i] == z:
                copy = dp[i-3]

    return dp[N]


# Modular exponential a, b, c -> a^b % c
# formular: a*b % c = (a%c)*(b%c) % c
def modularExponential(a, b, c):
    if b == 0:
        return 1

    if b == 1:
        return a % c

    m = modularExponential(a, b/2, c)
    if b % 2 == 0:
        return (m * m) % c
    else:
        return (a * m * m) % c



# Find Nth ugly number
# https://leetcode.com/problems/ugly-number-ii
# First number 1, next number: min(1*2, 1*3 or 1*5) = 2
# next number = 2*2, 1*3, 1*5 = 3; next = 2*2, 2*3, 1*5 = 4
# Use index2, index3, index5 to store next index for number need to multiple
# to 2, 3, 5
def findNthUglyNumber(n):
    ugly = [1]
    index2 = index3 = index5 = 0
    for k in range(1, n):
        val = min(ugly[index2] * 2, ugly[index3] * 3, ugly[index5] * 5)
        if val == ugly[index2] * 2:
            index2 += 1
        if val == ugly[index3] * 3:
            index3 += 1
        if val == ugly[index5] * 5:
            index5 += 1
        ugly.append(val)
    return ugly[-1]




# Super ugly number
# https://leetcode.com/problems/super-ugly-number/description/
# Solve as ugly number, handle duplicate by increasing index of prime, if it's
# smaller than current ugly number
def nthSuperUglyNumber(self, n, primes):
    """
    :type n: int
    :type primes: List[int]
    :rtype: int
    """
    if n <= 0:
        return -1

    if n == 1:
        return 1

    m = len(primes)
    ugly = [1]
    indices = [0 for x in range(m)]
    _max = 2**32 - 1
    for i in range(1, n):
        idx = 0
        curr = _max
        for j in range(m):
            val = ugly[indices[j]] * primes[j]
            # Avoid duplicate by increasing index of prime j
            if val <= ugly[-1]:
                indices[j] += 1
                val = ugly[indices[j]] * primes[j]
            if val < curr:
                curr = val
                idx = j
        ugly.append(curr)
        indices[idx] += 1

    return ugly[-1]


# Number of islands
# https://leetcode.com/problems/number-of-islands/description/
def _bfs(self, grid, i, j):
    if (i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0])
        or grid[i][j] == '0'):
        return
    grid[i][j] = '0'
    _bfs(grid, i+1, j)
    _bfs(grid, i-1, j)
    _bfs(grid, i, j+1)
    _bfs(grid, i, j-1)


def numIslands(self, grid):
    """
    :type grid: List[List[str]]
    :rtype: int
    """
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                count += 1
                _bfs(grid, i, j)

    return count


# Maximum index that A[i] <= A[j]
# e.g 3, 5, 4, 2, 1 => 4 for pair (3, 4)
# Brute force: iterate from each number from end to current number,
# if it's larger, return index j - i
# if sort array: (1,4) (2, 3) (3, 0) (4, 2) (5, 1)
def maximumIndex(nums):
    arr = [(i, j) for (i, j) in enumerate(nums)]
    arr.sort(key=lambda x: x[1])
    i, j = 0, len(nums) - 1
    res = 0
    while i < j:
        if arr[i][0] > arr[j][0]:
            i += 1
        else:
            res = max(res, arr[j][0] - arr[i][0])
            j -= 1
    return res


# Read 4 char
# https://leetcode.com/problems/read-n-characters-given-read4-ii-call-multiple-times
# Everytime call read4(buff) return number of char read
# Write function read n characters using read4, call multiple times
# Use buffStr to store read string, if read string less than n, read more
# string into buffer until no more string
def read4(buff):
    strs = 'abcdefghijkmnopqrtsxyz'
    for i in range(4):
        idx = randint(0, len(strs) - 1)
        buff[i] = strs[idx]
    return i


class Read4Solution(object):
    def __init__(self):
        self.buffStr = []
        self.readStr = 0

    def copy_from(self, buff):
        idx = 0
        length = read4(buff)
        while idx < length:
            self.buffStr.append(buff[idx])
            idx += 1
        self.readStr += length
        return length

    def read(self, buff, n):
        tmp = range(4)
        length = self.copy_from(tmp)

        need_read = 0
        if self.readStr < n:
            need_read = n - self.readStr

        while need_read > 0 and length > 0:
            length = self.copy_from(tmp)
            need_read -= length

        idx = 0
        while idx < min(self.readStr, n):
            buff[idx] = self.buffStr[idx]
            idx += 1
        return idx

r4 = Read4Solution()
# buff = list(range(1028))
# r4.read(buff, 10)
#
# r4.read(buff, 20)
#
# r4.read(buff, 25)
#


# License key formating group by k and uppercase
# https://leetcode.com/problems/license-key-formatting/description/
def licenseKeyFormatting(self, S, K):
    """
    :type S: str
    :type K: int
    :rtype: str
    """
    if not S or K <= 0:
        return ''

    buffer = deque([])
    n = len(S)
    count = 0
    for i in range(n-1, -1, -1):
        if S[i] == '-':
            continue
        buffer.appendleft(S[i])
        count += 1
        if count == K:
            buffer.appendleft('-')
            count = 0
    if buffer and buffer[0] == '-':
        buffer.popleft()

    return ''.join(buffer).upper()


# Division evaluate
# https://leetcode.com/problems/evaluate-division/description/
# Use DFS with hash table storing division pairs
class DivisionSolution:
    def dfs(self, start, end, lookup, status, res):
        if end in lookup[start]:
            return res * lookup[start][end]
        for k in lookup[start]:
            if k not in status:
                r = self.dfs(k, end, lookup, status + [k], res*lookup[start][k])
                if r is not None:
                    return r
        return None

    def calcEquation(self, equations, values, queries):
        """
        :type equations: List[List[str]]
        :type values: List[float]
        :type queries: List[List[str]]
        :rtype: List[float]
        """
        lookup = defaultdict(dict)
        for formular in zip(equations, values):
            a, b = formular[0]
            res = formular[1]
            lookup[a][b] = res
            if res != 0:
                lookup[b][a] = 1.0/res
        out = []

        for query in queries:
            x, y = query
            if x not in lookup or y not in lookup:
                out.append(-1.0)
                continue

            if x == y:
                out.append(1.0)
                continue

            if y in lookup[x]:
                out.append(lookup[x][y])
                continue

            res = self.dfs(x, y, lookup, [], 1.0)
            out.append(res or -1.0)
        return out


# Longest word in dictionary is subsequence
# https://leetcode.com/problems/longest-word-in-dictionary-through-deleting/
# https://techdevguide.withgoogle.com/resources/find-longest-word-in-dictionary-that-subsequence-of-given-string/#!
# give a dictionary find word in dictionary with longest substring of string
# Optmize: preprocess string s to have position of each character
# e.g a -> [1, 2, 3], p -> [10, 11], check substring invole binary search
# in this list
class DictionarySolution:
    # Check if t is subsequence of s
    def isSubstring(self, s, t):
        if len(t) > len(s):
            return False
        i = j = 0
        while i < len(s) and j < len(t):
            if s[i] != t[j]:
                i += 1
            else:
                i += 1
                j += 1
        return j == len(t)

    def longestSubstring(self, words, string):
        words.sort(key=lambda x: len(x), reverse=True)
        for word in words:
            if self.isSubstring(string, word):
                return word
        return None

    def longestSubstring2(self, words, string):
        letter_positions = defaultdict(list)
        for idx, c in enumerate(string):
            letter_positions[c].append(idx)

        for word in sorted(words, key=lambda w: len(w), reverse=True):
            pos = 0
            for letter in word:
                if letter not in letter_positions:
                    break

                possible_positions = [p for p in letter_positions[letter] if p >= pos]
                if not possible_positions:
                    break
                pos = possible_positions[0] + 1
            else:
                return word

ds = DictionarySolution()


# Flatten iterator of iterators
# Use array to remove ended iterator -> not efficient, use queue to enqueue and
# dequeue
class IF:
    def __init__(self, iterators):
        self.queue = deque(iterators)

    def __iter__(self):
        return self

    def next(self):
        if not self.queue:
            raise StopIteration

        while True:
            try:
                it = self.queue.popleft()
                item = next(it)
                self.queue.append(it)
                return item
            except StopIteration:
                continue
            except IndexError:
                raise StopIteration
            else:
                break

# iterators = [iter(range(10)), iter(range(10, 20, 2)), iter(range(30, 50, 3))]
# it = IF(iterators)
# for i in range(25):
#     print("next...", next(it))


# Placing mimes on NxN square
def place_mine(i):
    pass


def placeMime(M, N):
    remaining_mines = M
    remaining_cells = N
    for i in range(0, N):
        chance = float(remaining_mines) / remaining_cells
        if random.uniform(0., 1.) < chance:
            place_mine(i)
            remaining_mines -= 1
            remaining_cells -= 1


# Next closing time
# https://leetcode.com/problems/next-closest-time/description/
# e.g 11:34 ->  11:41
# Run from 1 to 24*60 to find next nearest where all number in allowed numbers
# Optimize: generate list of allowed numbers from 4 numbers
def nextClosestTime(time):
    ans = start = 60 * int(time[:2]) + int(time[3:])
    elapsed = 24 * 60
    allowed = {int(x) for x in time if x != ':'}
    for h1, h2, m1, m2 in itertools.product(allowed, repeat=4):
        hour, minute = 10*h1 + h2, 10*m1 + m2
        if hour < 24 and minute < 60:
            curr = 60*hour + minute
            curr_elapsed = (curr - start) % 24*60
            if 0 < curr_elapsed < elapsed:
                ans = curr
                elapsed = curr_elapsed
    return "{:02d}:{:02d}".format(*divmod(ans, 60))


'''Fruit into basket
https://leetcode.com/problems/fruit-into-baskets/
Same idea as longest substring with most 2 distinct chars
use 2 variables to store 2 type of fruits, when encounter 3rd types update
length, move start pointer to minimum index, otherwise keep updating
index of 2 types
'''
class FruitSolution:
    def totalFruit(self, tree):
        """
        :type tree: List[int]
        :rtype: int
        """
        if not tree:
            return 0

        n = len(tree)
        if n <= 2:
            return n

        p = q = -1
        res = 0
        start = 0
        for i in range(n):
            if p == -1 or tree[i] == tree[p]:
                p = i
            elif q == -1 or tree[i] == tree[q]:
                q = i
            else:
                res = max(i - start, res)
                if p < q:
                    start = p + 1
                    p = i
                else:
                    start = q + 1
                    q = i
        return max(res, i - start + 1)


# Max distance to closest person
# https://leetcode.com/problems/maximize-distance-to-closest-person/
# Use 2 pointers to find left seat and right seat occupied
# if there's no left seat, distance = right, no right seat: n - 1 - s
# else (right - left)//2
class MaxDistanceSolution:
    def maxDistToClosest(self, seats):
        """
        :type seats: List[int]
        :rtype: int
        """
        if not seats:
            return 0

        m = 0
        i = 0
        n = len(seats)
        while i < n:
            while i < n and seats[i] == 1:
                i += 1
            s = i - 1
            while i < n and seats[i] == 0:
                i += 1
            e = i
            print("se...", s, e)
            if s == -1:
                m = max(m, e)
            elif e == n:
                m = max(m, n - 1 - s)
            else:
                m = max(m, (e-s)//2)
        return m


# Backspace string edit
# https://leetcode.com/problems/backspace-string-compare/
class BPSolution:
    def _edit(self, t):
        stack = []
        for char in t:
            if char != '#':
                stack.append(char)
            elif stack:
                stack.pop()
        return ''.join(stack)

    def backspaceCompare(self, S, T):
        return self._edit(S) == self._edit(T)


# Repeated string match
# https://leetcode.com/problems/repeated-string-match/
# calculate time len(B)/len(A) + 1, check if B is substring A*q or A*(q+1)
class StringSolution:
    def repeatedStringMatch(self, A, B):
        if not A and not B:
            return 1
        q = (len(B) - 1) // len(A) + 1
        for i in range(2):
            if B in A*(q+i):
                return q+i
        return -1


# Bull and cows game, count match and unmatch char
# https://leetcode.com/problems/bulls-and-cows/
# Preprocess secret if guess match, increase A, else increase char
# loop through remain guess, if char in A, decrease char
# Optmize when encounter number in secrect, increase count, number in guess
# decrease count, if 2 number same => increase bull, if different, if count(secret)
# < 0, increase cows (as guess decrease it), if count(guess) > 0, increase cows
# (as secret increase it)
class BullCowSolution:
    def getHint(self, secret, guess):

        table = defaultdict(int)
        A = B = 0
        p = []
        for idx in range(len(guess)):
            if secret[idx] == guess[idx]:
                A += 1
            else:
                table[secret[idx]] += 1
                p.append(idx)

        for idx in p:
            if guess[idx] in table and table[guess[idx]] > 0:
                table[guess[idx]] -= 1
                B += 1

        return '{}A{}B'.format(A, B)

    def getHint2(self, secret, guess):
        count = [0 for i in range(10)]
        bull = cow = 0
        for idx in range(len(guess)):
            s = int(secret[idx])
            g = int(secret[idx])
            if s == g:
                bull += 1
            else:
                if count[s] < 0:
                    cow += 1
                if count[g] > 0:
                    cow += 1

                count[s] += 1
                count[g] -= 1
        return '{}A{}B'.format(bull, cow)


# Sentence screen fitting
# https://leetcode.com/problems/sentence-screen-fitting
# Concat all word list to one sentence, everytime find character of ending
# if not empty space, reduce it to space position
class SentenceSolution:
    def wordsTyping(self, sentence, rows, cols):
        """
        :type sentence: List[str]
        :type rows: int
        :type cols: int
        :rtype: int
        """
        s = ' '.join(sentence) + ' '
        n = len(s)
        i = 0
        for r in range(rows):
            i += cols
            while i > 0 and s[i % n] != ' ':
                i -= 1
            i += 1
        return i//n


# Optimal account balancing
# https://leetcode.com/problems/optimal-account-balancing/
# Modeling transaction as tree/graph, each edge is transaction from node A to B
# Simplify graph using path compression technique
# A -> B -> C with value x, y, 3 cases
# x < y: A -> C (x) and B -> C (y - x)
# x > y: A -> C (x) and A -> B (x - y)
# x == y: A -> C (x) number of transaction reduce by 1
# More complicated: A -> B, C -> B and B -> D (x, y, z) as above to simplify
# A -> B -> D and then C -> B -> D
# Return edge: A -> B and B -> A, (x, y) reduce to one transaction
class AccountSolution(object):
    def minTransfers(self, transactions):
        account = defaultdict(dict)
        # Push all transaction into account
        for s, r, a in transactions:
            account[s][r] = a

        need_simplify = True
        new_trans = False

        while need_simplify:
            new_trans = False
            print("current account...", account)
            for sender in account.keys():
                debt = account[sender]
                for receiver in debt.keys():
                    amount = debt.get(receiver, 0)
                    if not amount or receiver not in account:
                        continue

                    print("sender, receiver, amount...", sender, receiver, amount)

                    for next_recv in account[receiver].keys():
                        new_trans = True
                        next_amt = account[receiver].get(next_recv, 0)
                        if not next_amt:
                            continue

                        if next_amt <= amount:
                            amount -= next_amt
                            debt[receiver] = amount
                            if next_recv != sender:
                                debt[next_recv] = next_amt
                            account[receiver].pop(next_recv)
                            if amount == 0:
                                debt.pop(receiver, None)
                                break
                        else:
                            if next_recv != sender:
                                debt[next_recv] = amount
                            account[receiver][next_recv] -= amount
                            debt.pop(receiver, None)
                            break
                    print("account...", account)
                if not debt:
                    account.pop(sender)
            print("after simplify...", account)
            need_simplify = new_trans
        return sum(len(account[x]) for x in account)

    def minTransfers2(self, transactions):
        balance = defaultdict(int)
        for tran in transactions:
            balance[tran[0]] -= tran[2]
            balance[tran[1]] += tran[2]
        bal = [v for k, v in balance.items() if v != 0]
        return self.dfs(0, bal)

    def dfs(self, idx, debt):
        while idx < len(debt) and debt[idx] == 0:
            idx += 1

        res = float('inf')
        prev = 0

        for i in range(idx+1, len(debt)):
            if debt[i] != prev and debt[i] * debt[idx] < 0:
                debt[i] += debt[idx]
                res = min(res, 1 + self.dfs(idx + 1, debt))
                debt[i] -= debt[idx]
                prev = debt[i]
        return res if res < float('inf') else 0

ac = AccountSolution()


# Hand of straight: hand size W contains consecutive
# https://leetcode.com/problems/hand-of-straights/
# Use counter, count element, sorted item
# Loop and decrease consecutive count until has size W consecutive
def isNStraightHand(hand, W):
    """
    :type hand: List[int]
    :type W: int
    :rtype: bool
    """
    c = Counter(hand)
    for i in sorted(c):
        if c[i] > 0:
            for j in range(W, -1, -1):
                c[i + j] -= c[i]
                if c[i + j] < 0:
                    return False
    return True


# Longest consecutive one in 01 matrix
# https://leetcode.com/problems/longest-line-of-consecutive-one-in-matrix
# Brute force: count every row, column diagonal, and anti-diagonal for longest
# DP: use 3D array dp[m][n][4], e.g first dimension:
# dp[i][j][0] = dp[i-1][j][0] + 1 if matrix[i][j] = 1


# Strobogrammatic number
# https://leetcode.com/problems/strobogrammatic-number-ii/
# recursive for pair 00, 11, 69, 88, 96
class StrobogrammaticSolution(object):
    def find(self, left, right, n, res):
        count = len(left) + len(right)
        if count == n:
            res.append(left + right)
            return

        if count == n - 1:
            for num in ('0', '1', '8'):
                res.append(left + num + right)
            return

        for num in ('0', '1', '8'):
            if not left and num == '0':
                continue
            self.find(left + num, num + right, n, res)

        self.find(left + '6', '9' + right, n, res)
        self.find(left + '9', '6' + right, n, res)
        return

    def findStrobogrammatic(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        res = []
        self.find('', '', n, res)
        return res


# https://leetcode.com/problems/find-and-replace-in-string/solution/
# Find and replace string, use array to store result
def findReplaceString(S, indexes, sources, targets):
    """
    :type S: str
    :type indexes: List[int]
    :type sources: List[str]
    :type targets: List[str]
    :rtype: str
    """
    res = []
    prev = 0

    for idx, source, target in sorted(zip(indexes, sources, targets)):
        if idx > prev:
            res.append(S[prev: idx])

        if S[idx: idx + len(source)] == source:
            res.append(target)
        else:
            res.append(S[idx:idx+len(source)])
        prev = idx+len(source)
    if prev < len(S):
        res.append(S[prev:])
    return ''.join(res)


# https://leetcode.com/problems/isomorphic-strings/
# Isomorphic string, use hash map to map from one char to other
def isIsomorphic(s, t):
    """
    :type s: str
    :type t: str
    :rtype: bool
    """
    if len(s) != len(t):
        return False
    m = {}
    for i in range(len(s)):
        if s[i] in m:
            if t[i] != m[s[i]]:
                return False
        else:
            m[s[i]] = t[i]
    # Check if 2 different chars map to same char
    vals = m.values()
    if len(vals) != len(set(vals)):
        return False
    return True


# Minize max distance to gas station
# https://leetcode.com/problems/minimize-max-distance-to-gas-station
# On a horizontal number line, we have gas stations at positions
# add more K station so that the maximum distance between adjacent
# gas stations, is minimized.
# Use a max heap to store distance between station
# everytime add one more station to max distance, recalculate distance and push
# back to heap => run time: nlog(n) + klog(n), space: O(n)
# Optimize: each time adding one station in largest distance, can we add more?
# knowing that the number of added stations should make current distance less
# than next second distance, num += max(ceil(largest/second_largest), 1)
def maxGasStationDistance(stations, K):
    distances = []
    n = len(stations)
    for i in range(n-1):
        d = stations[i+1] - stations[i]
        distances.append([-d, 1, d])

    heapify(distances)
    for j in range(K):
        item = heappop(distances)
        prio, num, curr = item
        num += 1
        heappush(distances, [-(curr/num), num, curr])
    prio, num, curr = heappop(distances)
    return curr/num

print("max distance....", maxGasStationDistance([0, 4, 7, 12, 18, 20], 9))


# Find redundant connection to make tree
# https://leetcode.com/problems/redundant-connection/
# https://leetcode.com/problems/redundant-connection-ii/
# 2 cases: vertex with 2 parents, cycle in tree
def findRedundantConnection(edges):
    """
    :type edges: List[List[int]]
    :rtype: List[int]
    """
    def find(parent, x):
        if x not in parent:
            return x

        if parent[x] != x:
            parent[x] = find(parent, parent[x])
        return parent[x]

    parent = {}
    p, q = None, None

    for edge in edges:
        if not parent.get(edge[1]):
            parent[edge[1]] = edge[0]
        else:
            p = parent[edge[1]], edge[1]
            q = [edge[0], edge[1]]
            edge[1] = 0

    parent = {}

    for u, v in edges:
        if v == 0:
            continue

        ur, vr = find(parent, u), find(parent, v)
        if ur == vr:
            if not p:
                return [u, v]
            return p
        parent[vr] = ur
    return q

print("redundant connection...", findRedundantConnection([[5,2],[5,1],[3,1],[3,4],[3,5]]))


# Find loop in array
# [3, 1, 1, -2, 3, 2] -> loop 1, 1, -2
# edge cases: [-1, 2] -> 2 is not loop
# Use 2 pointers to find if there's loop
# https://leetcode.com/problems/circular-array-loop/


# Cracking the safe
# https://leetcode.com/problems/cracking-the-safe/
def crackSafe(n, k):
    visited = set()
    ans = []

    def cycle(node):
        for x in map(str, range(k)):
            v = node + x
            if v not in visited:
                visited.add(v)
                cycle(v[1:])
                ans.append(x)

    cycle("0" * (n-1))

    return "".join(ans) + "0"*(n-1)

print("cracking the safe...", crackSafe(1, 2))


# Character replacement
# https://leetcode.com/problems/longest-repeating-character-replacement/
# Find longest repeating substring created by replacing k chars
# use counter to count number of character in current window
# if number of difference exceed k, move low pointer to next character
# and substract count, re-calculate number of different chars
# Edge case: number of difference does not exceed k and pointer reach end
# => move back low pointer
# Second solution: sliding window, counter to count char, update most
# frequent char in window, diff = end - start + 1 - maxFrequent
# if diff exceeds k, move s and decrease counter,
# update length = end - start + 1
def characterReplacement(s, k):
    counter = defaultdict(int)
    count = 0
    lo, hi = 0, 1
    curr = s[0]
    ans = 1
    counter[curr] = 1

    while hi < len(s):
        if s[hi] != curr:
            count += 1

        counter[s[hi]] += 1
        if count > k:
            ans = max(ans, hi - lo)
            while lo < hi and s[lo] == curr:
                counter[s[lo]] -= 1
                lo += 1

            curr = s[lo]
            count = hi - lo - counter[curr] + 1

        hi += 1

    while count <= k and lo >= 0:
        if s[lo] != k:
            count += 1
        lo -= 1

    ans = max(ans, hi - lo - 1)

    return ans

print("character replacement...", characterReplacement('DAABBBBC', 3))


# Rectangle area
# https://leetcode.com/problems/rectangle-area/submissions/
# Find overlap area, left = max(A, E) = E
# for right, if overlap right = min(C, G) if non overlap, this is wrong
# min(C, G) = C < left => take max(min(C, G), left) = E
# Same for bottom and top
def computeArea(A, B, C, D, E, F, G, H):
    left = max(A, E)
    right = max(min(C, G), left)
    bottom = max(B, F)
    top = max(min(D, H), bottom)
    return (C-A)*(D-B) + (G-E)*(H-F) - (top - bottom) * (right - left)


# Rectangle area 2
# https://leetcode.com/problems/rectangle-area-ii/solution/
# - Solution 1: sorted x and y list, remap x, y to its index
# use 2D array to fill in cell covered by rectangle, grid[x][y] = 1 for x in
# map[x1], map[x2] and y in mapy[y1], mapy[y2]
# if grid = 1, calculate area, => run time: O(n^3)
# - Solution 2: line sweep
# Consider every rec as 2 layers, (x1, x2) open at y1 and (x1, x2) close at y2
# sort every open and close layers by y, calculate area for each layer y
# (x1, x2), (x3, x4)...xk => area = sum(x different) * (y - previous y)
# Optimize: use segment tree to add or remove layers => nlog(n)


# Maximum vactions days
# https://leetcode.com/problems/maximum-vacation-days
# n destination and k weeks, flights: matrix nxn for n flight connection
# days: nxk maximum days to stay at city i week j
# dp[i][j] is maximum vacations at city i, week j
# dp[i][j] = max(dp[i][j], days[i][j] + dp[x][j-1]) if flights[x][i] or x == i
# base case: j = 0, dp[i][0] = days[i][0] if flights[0][i]
# optimize: j depends on j - 1, just need an array dp[j-1] and dp[j] to store
# previous and current result
def maxVacationDays(flights, days):
    n, k = len(days), len(days[0])

    dp = [[0]*k for _ in range(n)]

    dp[0][0] = days[0][0]
    for i in range(1, n):
        if flights[0][i]:
            dp[i][0] = days[i][0]

    for i in range(n):
        for j in range(1, k):
            for x in range(n):
                if flights[x][i] or x == i:
                    dp[i][j] = max(dp[i][j], days[i][j] + dp[x][j-1])
    ans = 0
    for i in range(n):
        ans = max(ans, dp[i][k-1])

    return ans


# HARD: Bus route
# https://leetcode.com/problems/bus-routes/
def numBusesToDestination(routes, S, T):
    n = len(routes)
    start = set()
    end = set()
    buses = [set(route) for route in routes]
    conns = defaultdict(set)

    for idx, bus in enumerate(buses):
        if S in bus:
            start.add(idx)
        if T in bus:
            end.add(idx)

    for i in range(1, n):
        for j in range(i):
            if buses[i] & buses[j]:
                conns[i].add(j)
                conns[j].add(i)

    def bfs(node):
        q = deque([(node, 1)])
        visit = set([node])
        while q:
            bus, num = q.popleft()
            if bus in end:
                return num

            for conn in conns.get(bus, []):
                if conn not in visit:
                    q.append((conn, num + 1))
                    visit.add(conn)
        return -1

    ans = float('inf')
    for node in start:
        num = bfs(node)
        if num != -1:
            ans = min(ans, num)
    return ans if ans < float('inf') else -1

print("num of buses...", numBusesToDestination([[1, 2, 7], [3, 6, 7]], 1, 6))


# HARD: 24 games
# https://leetcode.com/problems/24-game/
# Back tracking all possible pairs
def judgePoint24(nums):
    from operator import mul, truediv, add, sub

    if not nums:
        return False

    n = len(nums)

    if n == 1 and round(nums[0], 2) == 24.00:
        return True

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            arr = [nums[x] for x in range(n) if x != i and x != j]
            for op in (mul, sub, add, truediv):
                if op in (add, mul) and j > i:
                    continue

                if op is truediv and nums[j] == 0.0:
                    continue

                arr.append(op(nums[i], nums[j]))
                if judgePoint24(arr):
                    return True
                arr.pop()
    return False


# Split BST
# https://leetcode.com/problems/split-bst/
# Recursive call split, if node < v, call to right,
# connect right to less subtree
# if node > v, call to left, connect left to greater subtree
class SplitSolution:
    def splitBST(self, root, v):
        if not root:
            return None, None

        if root.val == v:
            right = root.right
            root.right = None
            return root, right

        elif root.val < v:
            lte, gt = self.splitBST(root.right, v)
            root.right = lte
            return root, gt
        else:
            lte, gt = self.splitBST(root.left, v)
            root.left = gt
            return lte, root
