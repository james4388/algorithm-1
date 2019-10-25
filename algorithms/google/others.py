from collections import defaultdict, deque, Counter
import random


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


# Longest consecutive one in 01 matrix
# https://leetcode.com/problems/longest-line-of-consecutive-one-in-matrix
# Brute force: count every row, column diagonal, and anti-diagonal for longest
# DP: use 3D array dp[m][n][4], e.g first dimension:
# dp[i][j][0] = dp[i-1][j][0] + 1 if matrix[i][j] = 1


# Find loop in array
# [3, 1, 1, -2, 3, 2] -> loop 1, 1, -2
# edge cases: [-1, 2] -> 2 is not loop
# Use 2 pointers to find if there's loop
# https://leetcode.com/problems/circular-array-loop/
