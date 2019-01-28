from collections import deque, defaultdict
from random import randint
from bisect import bisect
from tree import TreeNode


# Find if door open after n visits
# first visit: toggle every door
# second visit: toggle 2, 4, 6th ... door
# 1 open, 0 close
# runtime: n/1 + n/2 + n/3 + ...+ n/n = O(nlogn)
# second solution: count divisors of each number => O(n. sqrt(n))
def doorOpen(doors, n):
    if not doors or n < 1:
        return doors

    for i in range(1, n+1):
        for j in range(1, n/i + 1):
            doors[i*j - 1] += 1
    return [x % 2 for x in doors]

# print "door open...", doorOpen([1, 0, 1, 1, 0, 0], 6)


# Linkedin
# [1, 2, [3, 4, [5]], 6] => 1 + 2 + 2*(3 + 4) + 3*5 + 6 = 3 + 14 + 15 + 6 = 38
class Solution(object):
    def _sumNested(self, arr, depth):
        s = 0
        for item in arr:
            if isinstance(item, (list, tuple)):
                s += self._sumNested(item, depth+1)
            else:
                s += item * depth
        return s

    def sumNestedList(self, nums):
        return self._sumNested(nums, 1)

    def collect(self, nums, m, depth):
        if depth > self.maxdepth:
            self.maxdepth = depth
        for num in nums:
            if isinstance(num, (list, tuple)):
                self.collect(num, m, depth+1)
            else:
                m[depth] += num

    def sumNestedReverse(self, nums):
        m = defaultdict(int)
        self.maxdepth = 1
        self.collect(nums, m, 1)
        s = 0
        for depth in m:
            s += (self.maxdepth - depth + 1)*m[depth]
        return s

print "sum nested reverse...", Solution().sumNestedReverse([1, [2, [3]], 4])
# print "sum nested...", Solution().sumNestedList([1, 2, [3, 4, [5]], 6])


# Calculator
# https://leetcode.com/problems/basic-calculator-ii/description/
# put char into array, if it's operator then convert prev string to integer
# Do 2 pass, first pass for * and /, second pass for - and +
def do_calculation(a, b, op):
    if op == '*':
        return a*b
    if op == '/':
        return a / b
    if op == '+':
        return a + b
    if op == '-':
        return a - b


def calculate(s):
    """
    :type s: str
    :rtype: int
    """
    n = len(s)
    if not n:
        return 0

    arr = []
    prev_idx = 0
    for i in range(n):
        if s[i] in ('+', '-', '*', '/'):
            arr.append(int(s[prev_idx: i]))
            arr.append(s[i])
            prev_idx = i + 1
    arr.append(int(s[prev_idx: n]))
    res = []
    j = 0
    while j < len(arr):
        if arr[j] in ('*', '/'):
            val = do_calculation(res[-1], arr[j+1], arr[j])
            res[-1] = val
            j += 1
        else:
            res.append(arr[j])
        j += 1
    out = 0
    j = 0
    while j < len(res):
        if res[j] in ('+', '-'):
            out = do_calculation(out, res[j+1], res[j])
            j += 1
        else:
            out += res[j]
        j += 1

    return out

# print "calculate...", calculate('14-3/2')


# Calculate square root of number
def sqrt(num):
    if num < 0:
        raise TypeError('Number cannot be negative')

    i, j = 0, num
    mid = None
    while i < j:
        mid = (i + j)/2
        if mid * mid == num:
            return mid
        if mid * mid > num:
            j = mid - 1
        else:
            i = mid + 1
    return i

print "sqrt 15...", sqrt(15)


# Find maximum subarray
# keep 2 variable max_so_far and maxutil, add to max util, if maxsofar < maxutil
# update it, if maxutil < 0, set it to zero
def maxSubArraySum(nums):
    max_so_far = -99999999
    max_ending_here = 0

    for i in range(len(nums)):
        max_ending_here = max_ending_here + nums[i]
        if (max_so_far < max_ending_here):
            max_so_far = max_ending_here

        if max_ending_here < 0:
            max_ending_here = 0
    return max_so_far

# print maxSubArraySum([0, -1, -1, 2, 2])


# Maximum product sub-array
# https://leetcode.com/problems/maximum-product-subarray/description/
# [1, 3, -1, 4, -2, -5]
def maxProduct(nums):
    if not nums:
        return 0

    maxh = minh = res = nums[0]

    for num in nums[1:]:
        if num > 0:
            maxh, minh = max(maxh*num, num), min(minh*num, num)
        else:
            maxh, minh = max(minh*num, num), min(maxh*num, num)

        res = max(res, maxh)
    return res


# Number of subset sum to k
# C(i, j) is number of subset at index i has sum j
# We can choose to take i or not, so
# C(i, j) = C(i-1, j) + C(i-1, j - arr[i])
# C(i, 0) always = 1 because we have empty subset
def numSubsetSumK(nums, k):
    if not nums and k:
        return 0
    n = len(nums)
    dp = [[0] * (k+1) for _ in range(n)]
    for x in range(n):
        dp[x][0] = 1

    for i in range(n):
        for j in range(1, k+1):
            if i == 0:
                if nums[i] == j:
                    dp[i][j] = 1
            elif j >= nums[i]:
                dp[i][j] = dp[i-1][j] + dp[i-1][j - nums[i]]
            else:
                dp[i][j] = dp[i-1][j]
    return dp[n-1][k]

print "num subset...", numSubsetSumK([2, 2, 3, 4, 5, 7], 9)


# Print tree by height, layer by layer, leaf = 0
def _printLayer(node, res):
    if not node:
        return -1

    left = _printLayer(node.left, res)
    right = _printLayer(node.right, res)
    h = max(left, right) + 1
    res[h].append(node.val)
    return h


def printTreeLayer(root):
    if not root:
        return []
    res = defaultdict(list)
    _printLayer(root, res)
    return res


# Convert integer to Roman
def integerToRoman(val):
    if val > 3999:
        raise ValueError('Overflow number')
    ten = ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX']
    dozen = ['', 'X', 'XX', 'XXX', 'XL', 'L', 'LX', 'LXX', 'LXXX', 'XC']
    hundred = ['', 'C', 'CC', 'CCC', 'CD', 'D', 'DC', 'DCC', 'DCCC', 'CM']
    thousand = ['', 'M', 'MM', 'MMM']
    return (thousand[val/1000] + hundred[val % 1000 / 100] + dozen[val % 100 / 10]
            + ten[val % 10])

print "integer to roman...", map(integerToRoman, [3, 4, 9, 58, 1994])


# Convert Roman to integer
# compare string i to its next i+1, if equal or larger, add to sum, else
# substract
def romanToInteger(r):
    roman = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    res = 0
    i = 0
    while i < len(r):
        val = roman[r[i]]

        if i + 1 < len(r):
            nextval = roman[r[i+1]]
            if val >= nextval:
                res += val
                i += 1
            else:
                res += nextval - val
                i += 2
        else:
            res += val
            i += 1
    return res

print "roman to integer...", map(romanToInteger, ['III', 'IV', 'IX', 'LVIII', 'MCMXCIV'])


# print all factors of number
# loop from 1 to sqrt(number)
# First common ancestor of 2 nodes (not itself)
# Design survey website client site, provide class, break down problem, api

# Random by frequency
def randomByFreq(nums, freq):
    n = len(freq)
    prefix = [0 for _ in range(n)]
    prefix[0] = freq[0]
    for i in range(1, n):
        prefix[i] = prefix[i-1] + freq[i]

    v = randint(0, prefix[n-1]) % prefix[n-1]
    pos = bisect(prefix, v)
    return nums[pos]

print "random...", randomByFreq([2, 3, 4, 5, 6], [10, 8, 14, 9, 5])


# Find k closet elements in array
# use binary search for value with minimum distance
# find first index low, calculate value x - mid value and mid+k value - x
# if larger than increase low, mid + 1 else decrease high
# [2, 4, 6, 7, 10, 13, 19, 21], k = 3, x = 9 -> 6, 7, 10
# [5, 8] , k = 1, x = 9 -> 5
def findClosestElements(arr, k, x):
    if not arr:
        return []

    lo, hi = 0, len(arr) - k

    while lo < hi:
        mid = (lo + hi)/2
        if x - arr[mid] > arr[mid + k] - x:
            lo = mid + 1
        else:
            hi = mid
    return arr[lo: lo + k]

print "Find k closest value...", findClosestElements([2, 4, 6, 7, 10, 13,
                                                      19, 21], 3, 9)


# Print matrix bottom up left right
# 1 2 3
# 4 5 6
# 7 8 9
# 0 1 2
# return 7 4 1 2 3 -> 8 5 6 -> 9
# i = m-1 -> 0 + j, 0 + j + 1 -> n-1
def printMatrixSNWE(matrix):
    if not matrix:
        return []
    m, n = len(matrix), len(matrix[0])
    res = []
    l = min(m, n)
    for layer in range(l):
        for i in range(m - 1, layer - 1, -1):
            res.append(matrix[i][layer])

        for j in range(layer + 1, n):
            res.append(matrix[layer][j])
    return res

print "print matrix snwe...", printMatrixSNWE([[1, 2, 3], [4, 5, 6],
                                               [7, 8, 9]])


# Find triplet in array: a + b > c, a + c > b, b + c > a
# Sort array, find position where a+b > c, i, j
# e.g: 3, 4, 5
def findTriplet(nums):
    nums.sort()
    n = len(nums)
    count = 0
    for i in range(n-2):
        k = i + 2
        for j in range(i+1, n-1):
            while k < n and nums[i] + nums[j] > nums[k]:
                k += 1
        count += k - j - 1
    return count


# Give a list of interval, insert new interval and merge them, get cover total
# Use interval linkedlist for non-overlap, 0-6 => 9-12 => 15-18 increasing order
# Insert new interval 3-9, if overlap merge with current and next until no
# overlap, 0-6 merge 3-9 => 0-9 set interval = 0-9 and merge it with later node
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.next = None

    def isOverlap(self, other):
        return self.end >= other.start

    def merge(self, other):
        if not self.isOverlap(other):
            return
        self.start = min(self.start, other.start)
        self.end = max(self.end, other.end)
        self.next = other.next
        del other

    def __len__(self):
        return self.end - self.start

    def __str__(self):
        return '{}-{}'.format(self.start, self.end)


class IntervalStore(object):
    def __init__(self):
        self.head = None
        self.cover = 0

    def insert(self, interval):
        self.cover = max(self.cover, len(interval))

        if not self.head:
            self.head = interval
            return

        prev = tmp = self.head
        i = interval
        # Find where overlap happen
        while tmp and i.end >= tmp.start:
            if tmp.isOverlap(i):
                tmp.start = min(tmp.start, i.start)
                tmp.end = max(tmp.end, i.end)
                self.cover = max(self.cover, len(tmp))
                while tmp.next and tmp.isOverlap(tmp.next):
                    tmp.merge(tmp.next)
                return
            prev = tmp
            tmp = tmp.next
        # No overlap happen:
        if tmp == self.head:
            interval.next = self.head
            self.head = interval
        elif tmp is None:
            prev.next = interval
        else:
            prev.next = interval
            interval.next = tmp

    def __str__(self):
        m = []
        tmp = self.head
        while tmp:
            m.append(str(tmp))
            tmp = tmp.next
        return ' => '.join(m)

istore = IntervalStore()
istore.insert(Interval(0, 6))
istore.insert(Interval(9, 12))
istore.insert(Interval(15, 18))
istore.insert(Interval(3, 9))
istore.insert(Interval(5, 14))
print "store...", istore


# Give a list of flower (0, 1), 2 flower cannot be planted adjeciently, check
# if can plan more number of flower
def canPlanFlower(flowers, k):
    n = len(flowers)
    count = 0
    for i in range(n):
        if flowers[i] == 1:
            continue

        if i == 0:
            if i + 1 < n and flowers[i+1] == 0:
                count += 1
                flowers[i] = 1
        elif i == n - 1:
            if i - 1 >= 0 and flowers[i - 1] == 0:
                count += 1
                flowers[i] = 1
        else:
            if flowers[i-1] == 0 and flowers[i+1] == 0:
                count += 1
                flowers[i] = 1
    return k <= count


# Give a list of (child parent isLeft) build a binary tree from it
# return root, node with parent [10, 50, false], [20, 50, true], [5, 10, true]
# return node 50
def buildTree(childParents):
    root = None
    mapping = {}
    for item in childParents:
        child, parent, isLeft = item
        childNode = TreeNode(child)
        if parent in mapping:
            parentNode = mapping[parent][0]
        else:
            parentNode = TreeNode(parent)
            mapping[parent] = (parentNode, None)
        if isLeft:
            parentNode.left = childNode
        else:
            parentNode.right = childNode
        mapping[child] = (childNode, parentNode)
        if not mapping[parent][1]:
            root = parentNode
    return root


# Give number of 1, 2, 3, 4 find number of string can form, no 2 adjecent
# number are the same
# x1, x2, x3, x4 is number of digits for 1, 2, 3, 4
# string length: x1 + x2 + x3 + x4 = n
# for position 0 => 4 choices
# position 1 => 3 choices
# position 2 =>  3 choices
# example: only 3 numbers 1, 2, 3 and digits = 1, 2, 1
# 2123 2132    2312    2321 => 1 1 1 => 2 choice
# 1232 => 0 2 1 => 1 choice
# 3212 => 1 2 0 => 1 choice
def _count(arr, k, prev, memo, res):
    if k == 0:
        return res

    s = str(arr)
    if s in memo:
        return memo[s]

    for i, v in enumerate(arr):
        if arr[i] > 0 and i != k:
            res *= _count(arr[:i] + [v-1] + arr[i+1:], k - 1, i, memo, res)
    memo[s] = res
    return res


def countNumber(arr):
    n = sum(arr)
    memo = {}
    res = 1
    res = _count(arr, n, -1, memo, res)
    return res

# print "count number...", countNumber([1, 2, 1])

# Game 100, 2 player choose number 1...10, first player cause total exceed
# 100 wins, what if cannot re-use number


# Follow matrix, find influencer: followed by everyone, not following anyone
# 2 phases: find candidate for influencer and test if it's influencer
# Brute force: O(n^2)
def findInfluencer(matrix):
    if not matrix:
        return -1
    n = len(matrix)
    influencer = 0

    for i in range(n):
        # if no one follow this one, or this one follow someone,
        # change candidate
        if matrix[i][influencer] == 0 or matrix[influencer][i] == 1:
            influencer = i

    # Test current candidate
    for i in range(n):
        if matrix[i][influencer] == 0 or matrix[influencer][i] == 1:
            return -1
    return influencer
# Find all repeating substring with length k
# Find min difference in 2 array, follow up: if 2 array sorted


# Find repeated half time number in array
# Majority voting, choose current number, if next number equal current one
# increase count else decrease count, if count = 0, change candidate
def majorityElement(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if not nums:
        return None

    count = idx = 0
    for i in range(len(nums)):
        if not count:
            idx = i
            count = 1
        else:
            count += 1 if nums[idx] == nums[i] else -1
    return nums[idx]


# Search value in rotated sorted array
# check mid value, if mid value > right value, then mid in first half rotated
# then check if target in first half
# if mid value less than right, then mid second half
# check if target in second half
def searchRotated(nums, target):
    n = len(nums)
    l, r = 0, n-1
    while l <= r:
        mid = l + (r-l)/2
        if nums[mid] == target:
            return mid

        if nums[mid] > nums[r]:
            if nums[l] <= target < nums[mid]:
                r = mid - 1
            else:
                l = mid + 1
        else:
            if nums[mid] < target <= nums[r]:
                l = mid + 1
            else:
                r = mid - 1
    return -1


# Permutation of a string or array
# Use backtracking
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


# Check if 2 strings are isomorphic, if one char from a can map to other char
# in b, no 2 char map to same char, char can map to itself
def isomorpheous(a, b):
    if len(a) != len(b):
        return False
    dicti = {}
    for i in xrange(len(a)):
        if a[i] in dicti:
            if b[i] != dicti[a[i]]:
                return False
        else:
            dicti[a[i]] = b[i]
    # Check if 2 different chars map to same char
    if len(dicti.values()) != len(set(dicti.values())):
        return False
    return True


# Check if a string is valid number
# if 1 char: it has to be digit
# first char has to be + - or digit
# before . is digit, after . is digit or empty
# before e has to be digit,
def validChar(char):
    return char in ('+', '-', '.', 'e') or char.isdigit()


def isValidNumber(numstr):
    s = numstr.strip()
    if len(s) == 1 and not s[0].isdigit():
        return False

    if s[0] not in ('+', '-', '.') and not s[0].isdigit():
        return False

    hasDot = False
    hasE = False

    for idx in range(1, len(s)):
        if not validChar(s[idx]):
            return False

        if s[idx] == '.':
            if hasDot or hasE:
                return False
            hasDot = True
            if idx == len(s) - 1:
                return False
            if not s[idx+1].isdigit():
                return False

        elif s[idx] == 'e':
            if hasE:
                return False

            hasE = True
            if not s[idx-1].isdigit():
                return False

            if idx == len(s) - 1:
                return False

        elif (s[idx] == '-' or s[idx] == '+') and s[idx-1] != 'e':
            return False

    return True

print "isNumber....", isValidNumber(" -1e+1.0")
# Design a REST API
# Design hangman game, Amazon shopping cart
# Design T9 dictionary


# Lowest common ancestor
# Case 1: without parent pointer
# use parent dictionary
# iterative => recursive
def LCA(root, node1, node2):
    parent = {root: None}
    queue = deque([root])
    while queue and (node1 not in parent or node2 not in parent):
        n = queue.popleft()
        if n.left:
            parent[n.left] = n
            queue.append(n.left)
        if n.right:
            parent[n.right] = n
            queue.append(n.right)

    ancestors = {}
    while node1:
        ancestors[node1] = 1
        node1 = parent[node1]

    while node2:
        if node2 in ancestors:
            return node2
        node2 = parent[node2]
    return node2


def LCA2(root, node1, node2):
    if root in (None, node1, node2):
        return root

    left = LCA2(root.left, node1, node2)
    right = LCA2(root.right, node1, node2)

    if left and right:
        return root

    return left if left else right


# Case 2 has parent pointer
def LCA3(root, node1, node2):
    parents = {}
    while node1:
        parents[node1] = 1
        node1 = node1.parent

    while node2:
        if node2 in parents:
            return node2
        node2 = node2.parent
    return node2


# Min stack
class MinStack(object):
    def __init__(self):
        self.stack = []
        self.min = None

    def push(self, x):
        if not self.stack:
            self.min = x
            self.stack.append(x)
        else:
            self.stack.append(x - self.min)
            if x < self.min:
                self.min = x

    def pop(self):
        if not self.stack:
            return
        val = self.stack.pop()
        if val < 0:
            # min now is x so prev min = x - (x - prevmin) = prevmin
            self.min = self.min - val

    def peak(self):
        if not self.stack:
            return None

        val = self.stack[-1]
        # val = x - min => x = val + min
        if val > 0:
            return val + self.min
        # min = x => return min
        else:
            return self.min

    def getMin(self):
        return self.min
