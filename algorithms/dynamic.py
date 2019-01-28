from collections import deque
import sys


MAX = sys.maxint
'''
NOTE:
- Type 1: current state depends on previous state, find max value
by previous state, question: maximum value, revenue e.g rob cutting
- Type 2: compare sequence, 2 cases: xi = yj or xi != yj, question: find
maximum matching e.g: longest common sequence, palidrome, edit string
distance, longest increasing subsequence.
- Type 3: selection process, at each step, it can either choose xi to be in
collection or dump it, question: maximize the value, or count number of ways,
e.g: snapsack problem, coin change problem
- Type 4: partition list into number of subarray
- Type 5: Catalan number, count number of ways: if process x element, then
the remain n-x-1 element could be processed in same way, forming formula:
C(n) = sum C(i) * C(n - i - 1) question: number of expression for n parenthesis
number of full binary tree, number of tree with n leaves
- Type 6: Bell number, count number of ways to partition a set, at element n
it can add to current set or be a single set: S(n+1, k) = k*S(n, k) + S(n, k-1)
k = 1...n, => B(i, j) = B(i-1, i-1) if (j=0) or B(i-1, j-1) + B(i, j-1)
'''

# Rob cutting
# a rob with n inches, for each length has different price p
# find maximum revenue
# p = pi + nk -> max(p) = pi + max(nk)
def robcutting(n, p):
    r = [0 for i in xrange(n+1)]
    s = [0 for i in xrange(n+1)]

    for i in xrange(1, n+1):
        q = -1

        for j in xrange(1, i+1):
            if q < p[j] + r[i - j]:
                q = p[j] + r[i - j]
                s[i] = j

        r[i] = q

    k = n
    while k > 0:
        print s[k]
        k = k - s[k]

    return r, s

p = {1: 1, 2: 5, 3: 8, 4: 9, 5: 10, 6: 17, 7: 17, 8: 20, 9: 24, 10: 30}
n = 8

# print "rob cutting for %s is %s" % (n, robcutting(n, p))


# Matrix chain
# Given n matrix, A1...An
# each size pi
# Find the way to multiply to have minimum scalar computations
# A x B -> cal: A row x B col x A row
# Subproblem: Ai...Aj choose k in i, j -> m(i, j) = m(i, k) + m(k + 1, j) + p(i-1).p(k).p(j)
def matrix_chain(p):
    n = len(p)
    m = [[-1]*n for i in xrange(n+1)]
    s = [[-1]*n for i in xrange(n+1)]

    for i in xrange(1, n):
        m[i][i] = 0

    for l in xrange(2, n):
        for i in xrange(1, n-l+1):
            j = i + l - 1
            for k in xrange(i, j):
                q = m[i][k] + m[k+1][j] + p[i-1]*p[k]*p[j]
                if m[i][j] == -1 or q < m[i][j]:
                    m[i][j] = q
                    s[i][j] = k
    return m, s


# Longest common sequence X = {x1, x2, ...xm} Y = {y1, y2..., yn}
# suppose z is longest sequence = {z1, z2, ..., zk}
# if xm = yn -> subproblem x(m-1), y(n-1)
# if xm != yn -> subproblem x(m), y(n-1) or x(m-1), y(n)
def longest_sequence(x, y):
    m = len(x)
    n = len(y)
    c = [[None]*(n+1) for i in xrange(m+1)]

    for i in xrange(m+1):
        for j in xrange(n+1):
            if i == 0 or j == 0:
                c[i][j] = 0

            elif x[i-1] == y[j-1]:
                c[i][j] = c[i-1][j-1] + 1
            else:
                c[i][j] = max(c[i-1][j], c[i][j-1])

    return c[m][n]


# print "longest sequence.....", longest_sequence('ADCBADCAABD', 'BCAADBCAABDD')


# Longest increasing subsequence X = [3, 10, 2, 1, 4, 20]
# sequence [3, 10, 20]
# Subproblem: let sequence: x1x2...xk....xi, v1v2...vk...vi is longest sub
# if xi > xk: vi = max(vi, vk + 1)
# Return subsequence: s[s1, s2....si], if xi > xk and vi < vk + 1, si = k
def longest_increasing_sequence(arr):
    n = len(arr)
    v = [1 for x in range(n)]
    s = [-1 for x in range(n)]

    _max = 1
    _max_idx = -1

    for i in range(1, n):
        for j in range(i):
            if arr[i] > arr[j]:
                if v[i] < v[j] + 1:
                    v[i] = v[j] + 1
                    s[i] = j
                    if v[i] > _max:
                        _max = v[i]
                        _max_idx = i
    result = deque([])
    idx = _max_idx
    while idx != -1:
        result.appendleft(arr[idx])
        idx = s[idx]

    return _max, list(result)

# print "longest increasing subsequence....", longest_increasing_sequence([50, 3, 10, 7, 40, 80, 5])


# Min editing, given 2 string, find minimum operator to convert str1 into str2
# Process each char, if they are identical, continue
# if different either remove, insert, or replace
def edit_str(str1, str2):
    m, n = len(str1), len(str2)
    i, j = 0, 0
    arr = [[0]*(n+1) for x in range(m+1)]
    # m row, n column
    for i in range(m+1):
        for j in range(n+1):
            if i == 0:
                arr[i][j] = j
            elif j == 0:
                arr[i][j] = i
            elif str1[i-1] == str2[j-1]:
                arr[i][j] = arr[i-1][j-1]
            else:
                arr[i][j] = 1 + min(arr[i][j-1], arr[i-1][j], arr[i-1][j-1])
    return arr[m][n]

# print "edit_str....", edit_str("greeksquad", "ggesksmquad")


# Min cost path, given matrix cost mxn, find path from 0,0 to m,n for least cost
# it can go right, down or diagonal
# cost to reach position i, j: c = c[i, j] + min(c[i-1, j-1], c[i-1, j], c[i, j-1])
def min_path(matrix):
    row = len(matrix)
    col = len(matrix[0])
    c = [[None]*col for i in range(row)]
    c[0][0] = matrix[0][0]

    for i in range(1, col):
        c[0][i] = c[0][i-1] + matrix[0][i]

    for j in range(1, row):
        c[j][0] = c[j-1][0] + matrix[j][0]

    for i in range(1, row):
        for j in range(1, col):
            c[i][j] = matrix[i][j] + min(c[i-1][j-1], c[i-1][j], c[i][j-1])

    return c


# print "Min sum path....", min_path([[1, 2, 3], [4, 8, 2], [1, 5, 3]])


# Calculate number of coin changes options
# Given coin S = {1, 2, 3}, find how many ways to change n = 5
# Subproblems: ({1, 2, 3}, 5) = ({1, 2, 3}, 5 - 3) + ({1, 2}, 5)
def coin_change(s, n):
    if not s or n < 0:
        return 0

    m = len(s)
    matrix = [[0]*m for x in xrange(n+1)]

    for l in range(m):
        matrix[0][l] = 1

    for j in range(1, n+1):
        for k in range(m):
            x = matrix[j - s[k]][k] if j >= s[k] else 0
            y = matrix[j][k-1] if k >= 1 else 0
            matrix[j][k] = x + y

    return matrix[n][m-1]

# print "Coin change...", coin_change([2, 3, 4], 8)


# Snapsack 0-1 problems
# There're n items with weight w {1...n} and values v {1...n}
# A bag can contains maximum W weight, find maximum values with selected item
# Subproblem: at each step we can choose to either add item or dump it
# S(W, items) = maximum v(n) + S(W - w(n), items(n-1)) and S(W, items(n-1))
def snapsack(w, wt, vals):
    if w <= 0 or not wt or not vals:
        return 0

    n = len(wt)
    arr = [[0]*n for i in xrange(w+1)]

    for i in range(1, w+1):
        for j in range(n):
            if j == 0:
                if i >= wt[j]:
                    arr[i][j] = vals[j]
            else:
                if i >= wt[j]:
                    arr[i][j] = max(vals[j] + arr[i - wt[j]][j-1], arr[i][j-1])
                else:
                    arr[i][j] = arr[i][j-1]

    return arr[w][n-1]

# print "Snapsack....", snapsack(5, [1, 2, 3], [6, 10, 12])


# Longest palindrome problem: find LP in sequence X = BBABCBCAB -> BABCBAB
# Subproblem: if first and last equal: L(1, n) = 2 + L(2, n-1)
# else: L(1, n) = max(L(1, n-1), L(2, n))
# 2 dimensions problem: i, j increase vs decrease -> moving window
def longest_palindrome(seq):
    n = len(seq)
    if n <= 1:
        return n

    m = [[0]*n for i in range(n)]
    for i in range(n):
        m[i][i] = 1

    for s in range(2, n+1):
        for i in range(n - s + 1):
            j = i + s - 1
            if seq[i] == seq[j]:
                m[i][j] = 2 + m[i+1][j-1]
            else:
                m[i][j] = max(m[i][j-1], m[i+1][j])
    return m[0][n-1]

# print "longest palindrome...\n", longest_palindrome('BBABCBCAB')


# Optimal BST tree, key = {k1,..., kn}; dummy node = {d0, ..., d(n-1)}
# present probability to find key pk and cannot find qk
# construct tree for least search cost
# e(i, j) = q(i-1) if j = i -1 else min(e(i, r-1) + e(r+1, j) + w(i, j)) i <= j
def optimal_bst(p, q, n):
    e = [[0]*n for x in range(n+1)]
    w = [[0]*n for x in range(n+1)]
    root = [[0]*n for x in range(n)]

    for i in range(1, n+1):
        e[i][i-1] = q[i-1]
        w[i][i-1] = q[i-1]

    for l in range(1, n+1):
        for i in range(1, n - l + 1):
            j = i + l - 1
            e[i][j] = None
            w[i][j] = w[i][j-1] + p[j] + q[j]

            for r in range(i, j+1):
                t = e[i, r-1] + e[r+1, j] + w[i][j]
                if not e[i][j] or t < e[i][j]:
                    e[i][j] = t
                    root[i][j] = r
    return e, root


# Partition list of numbers S into k range to minimize the maximum sum of range
# C(n, k) is min value
# C(n, 1) = sum(x1..xn)
# C(1, k) = x1
# for value i from 1...n, C(i, k) = max( C(i, k-1), sum(xi...xn) )

def partition(arr, k):
    n = len(arr)
    matrix = [[MAX]*(k+1) for i in range(n+1)]
    s = [0] + arr

    for x in range(1, n+1):
        s[x] += s[x-1]

    for x in range(1, n+1):
        matrix[x][1] = s[x]

    for x in range(1, k+1):
        matrix[1][x] = s[0]

    for i in range(2, n+1):
        for j in range(2, k+1):
            for x in range(1, i):
                cost = max(matrix[x][j-1], s[i] - s[x])
                if matrix[i][j] > cost:
                    matrix[i][j] = cost
    for r in matrix:
        print r
    return matrix[n][k]


# print partition([2, 4, 7, 5, 2, 9, 3, 11, 4, 1, 13], 4)


# bytelandian gold coins: https://www.spoj.com/problems/COINS/
# each coin can exchange to n/2, n/3 and n/4 coin (scale down)
# and exchange to dollar for 1:1
# C(n) = max(n, C(n/2) + C(n/3) + C(n/4))
def exchangeBytelandia(n):
    if n <= 0:
        return n

    arr = [0 for _ in range(n+1)]
    arr[1] = 1
    for i in range(2, n+1):
        arr[i] = max(i, arr[i/2] + arr[i/3] + arr[i/4])
    return arr[n]

# print "exchange...", exchangeBytelandia(12)


# Catalan numbers
# C(n+1) = sum C(i)*C(n-i-1)
# Number of expression for n pair parentheses
# Number of full binary trees
# Number of binary search tree with n leaves
def catalan(n):
    if n == 0 or n == 1:
        return 1

    c = [0 for _ in range(n+1)]

    c[0] = 1
    c[1] = 1
    for i in range(2, n+1):
        c[i] = 0

        for j in range(i):
            c[i] += c[j] * c[i-j-1]

    return c[n]


# Bell numbers: count number of ways to partition a set
# {1, 2, 3} -> {{1}, {2}, {3}}, {{1,2}, {3}}, {{1, 3}, {2}}, {{1}, {2, 3}},
# {{1,2,3}}
# S(n, k) is number of ways to put n into k set
# S(n+1, k) = k*S(n, k) + S(n, k-1)
# n+1 number can add into each existing k set (k choice), or be a single
# element set S(n, k-1)
# B(n) = sum S(n, k) k = 1...n
# Bell triangle 1 | 1 2 | 2 3 5 => B(i, j) = B(i-1, i-1) (j = 0)
# or B(i-1, j-1) + B(i, j-1)
def bellNumber(n):
    b = [[0 for _ in range(n+1)] for _ in range(n+1)]
    b[0][0] = 1

    for i in range(1, n+1):
        b[i][0] = b[i-1][i-1]

        for j in range(1, i+1):
            b[i][j] = b[i-1][j-1] + b[i][j-1]
    return b[n][0]


# Coin collections: find how many ways to reach bottom from top left
# for k coins
# https://www.geeksforgeeks.org/number-of-paths-with-exactly-k-coins/
def _pathCount(matrix, m, n, k, dp):
    if m < 0 or n < 0:
        return 0

    if m == 0 and n == 0:
        return matrix[m][n] == k
    if dp[m][n][k] != -1:
        return dp[m][n][k]

    dp[m][n][k] = (_pathCount(matrix, m-1, n, k - matrix[m][n], dp) +
                   _pathCount(matrix, m, n-1, k - matrix[m][n], dp))
    return dp[m][n][k]


def pathCount(matrix, k):
    m = len(matrix)
    n = len(matrix[0])
    dp = [[[-1]*k for _ in range(n)] for _ in range(m)]
    return _pathCount(matrix, m, n, k, dp)


# Coefficient number: C(n, k) = C(n-1, k-1) + C(n-1, k); C(n, 0) = C(n, n) = 0
# Count how many ways to put m balls to n bins
# S(m, n) ways to assign m balls to n bins
# S(m, n) = S(m, n-1) + S(m-1, n-1) + S(m-2, n-1) ... + S(0, n-1))


# Profit table scheme
# https://leetcode.com/problems/profitable-schemes/description/
# G = 10, P=5, [2, 3, 5], [6, 7, 8] count number of scheme selections
# dp[p][g] is number of scheme for profit p and required g people
# dp[p + p0][g + g0] += dp[p][g] for p0, g0 in list
# note p + p0 = min (p + p0, P), g + g0 <= G
def profitableSchemes(G, P, group, profit):
    """
    :type G: int
    :type P: int
    :type group: List[int]
    :type profit: List[int]
    :rtype: int
    """
    if not group or not profit:
        return 0

    dp = [[0] * (G+1) for _ in xrange(P+1)]
    dp[0][0] = 1

    for p, g in zip(profit, group):
        for i in xrange(P, -1, -1):
            for j in xrange(G - g, -1, -1):
                dp[min(i + p, P)][j + g] += dp[i][j]
    return sum(dp[P]) % (10**9 + 7)

# print "Profitable scheme...", profitableSchemes(10, 5, [2, 3, 5], [6, 7, 8])


# Count all palindrome subsequence in a string
# dp[i][j] is the number of palindrome subseq in i, j
# if s[i] == s[j]: dp[i][j] = dp[i+1][j] + dp[i][j-1] + 1
# if s[i] != s[j]: dp[i][j] = dp[i+1][j] + dp[i][j-1] - dp[i+1][j-1]
# result = dp[0][n-1], dp[i][i] = 1
def countPalindromeSubsequence(s):
    if not s:
        return 0

    n = len(s)
    dp = [[0]*n for _ in range(n)]

    for i in range(n):
        dp[i][i] = 1

    for l in range(2, n+1):
        for i in range(n - l + 1):
            j = i + l - 1
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j] + dp[i][j-1] + 1
            else:
                dp[i][j] = dp[i+1][j] + dp[i][j-1] - dp[i+1][j-1]
    return dp[0][n-1] % (10**9 + 7)

# print "palidrome subsequence...\n", countPalindromeSubsequence('bccb')


# Count distinct subsequence in a string
# dp[i] = 2*dp[i-1] - count number of subsequence of same char i before
# ab -> '', a, b, ab
# abb -> '', a, b, ab + b, ab, bb, abb - (b + ab)
def countDistinctSubsequence(s):
    if not s:
        return 1

    n = len(s)
    dp = [0 for _ in range(n+1)]
    dp[0] = 1
    chars = [-1 for _ in range(256)]

    for i in range(1, n+1):
        dp[i] = 2 * dp[i-1]
        v = ord(s[i-1]) - ord('a')
        if chars[v] != -1:
            dp[i] = dp[i] - dp[chars[v]]
        chars[v] = i - 1

    return dp[n]

print "distinct subsequence...", countDistinctSubsequence('abbc')
