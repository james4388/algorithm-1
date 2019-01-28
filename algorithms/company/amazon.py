from collections import deque, defaultdict
from heapq import heappop, heappush


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


# jump game
# Give list of integer, value is number of jump at index, determine if can
# reach destination
# https://leetcode.com/problems/jump-game/description/
# [2, 3, 1, 1, 4] -> True => jump from index 0 -> 1 -> 4
# at each index, calculate how far can jump: d = index + num[index]
# if d less than current index: return false, or d > n - 1 return True
def canJump(nums):
    if not nums:
        return True

    distance = 0
    for i in range(len(nums)):
        if i > distance:
            return False
        d = i + nums[i]
        distance = max(distance, d)
        if distance >= len(nums) - 1:
            return True
    return False

print "can jump...", map(canJump, [[], [0], [1, 0, 0], [2, 3, 1, 1, 4]])


# Find number minimum jump
# https://leetcode.com/problems/jump-game-ii/description/
# [2, 3, 1, 1, 4] => 2 jumps
# Solution1:Use BFS for each index, (index + 1) ... (index + val)
# Solution3: at each index update number of jumps for next index
# e.g at index 2: can be jumps from index 0 or index 1 => num jump = 1
# index 0 => 0
# Edge case: value = 0 ? need to consider value more than 0 only
# what if no jumps found ? value = n + 1 => return -1
# empty array, array has 1 element
# 2, 3, 1, 1, 4 -> number of jump 0, 1, 1, 2, 2
# Solution 3: calculate max distance, if i go out of max, increase jump,
# update max distance
def jump(nums):
    if not nums or len(nums) <= 1:
        return 0
    queue = deque([(0, 0)])

    res = len(nums) + 1
    while queue:
        idx, step = queue.popleft()
        if idx + nums[idx] >= len(nums) - 1:
            res = step + 1
            break
        for i in range(1, nums[idx] + 1):
            queue.append((idx+i, step+1))
    # In case: no jumps found
    return res if res < len(nums) + 1 else -1


def jump2(nums):
    if not nums or len(nums) <= 1:
        return 0
    n = len(nums)
    dp = [n+1 for _ in range(n)]
    dp[0] = 0
    for i in range(n):
        for j in range(i):
            if dp[j] > n:
                continue
            if j + nums[j] >= i:
                dp[i] = min(dp[i], dp[j] + 1)
    return dp[n-1] if dp[n-1] < n+1 else -1


def jump3(nums):
    if not nums or len(nums) <= 1:
        return 0

    n = len(nums)
    distance = 0
    _max = 0
    res = 0
    i = 0
    while distance >= i:
        res += 1
        while i <= distance:
            _max = max(_max, i + nums[i])
            i += 1
            if _max >= n - 1:
                return res
        distance = _max
    return -1

# print "minimum of jumps...", map(jump3, [[], [0], [1, 0, 0], [2, 3, 1, 1, 4]])


# HARD: Trap rain water
# https://leetcode.com/problems/trapping-rain-water/description/
# Brute force: At each index i, find maximum of left side,
# maximum of right sides, take the
# minimum and minus to current idx height min(max_left, max_right) - height
# it repeats everytime to find max left and max right => pre-compute max
# Use stack: push height into stack, if current > height, pop stack top and
# calculate distance
def trap(height):
    n = len(height)
    ans = 0
    current = 0
    stack = []
    while current < n:
        while stack and height[current] > height[stack[-1]]:
            idx = stack.pop()
            if not stack:
                break
            distance = current - stack[-1] - 1
            h = min(height[current], height[stack[-1]]) - height[idx]
            ans += distance * h

        stack.append(current)
        current += 1
    return ans

print "Trap rain water...", trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1])


# HARD: Max points on line
# https://leetcode.com/problems/max-points-on-a-line/description/
class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


def maxPoints(points):
    """
    :type points: List[Point]
    :rtype: int
    """
    if not points:
        return 0
    n = len(points)
    if n < 2:
        return n

    table = defaultdict(set)
    res = 2

    for i in range(1, len(points)):
        for j in range(i):
            m = points[i].y - points[j].y
            n = points[i].x - points[j].x
            q = points[i].y * points[j].x - points[j].y * points[i].x
            if n != 0:
                if n < 0:
                    n, m, q = -n, -m, -q
                m = m/float(n)
                q = q/float(n)
                k = (m, q)
            else:
                k = points[i].x
            table[k].add(i)
            table[k].add(j)

            res = max(res, len(table[k]))
    return res

print maxPoints([Point(0, 0), Point(1, 1), Point(-1, -1)])


# HARD: Skyline each building [x, y, height], print the contour line of buildings
# https://leetcode.com/problems/the-skyline-problem/description/
# Naive approach for each position from x left to max right
# Find the maximum height at each point => O(n^2)
# Optimize find maximum height at each critical points
def getSkyline(buildings):
    if not buildings:
        return []

    n = len(buildings)
    output = []
    table = {}
    for i in range(n):
        x, y, h = buildings[i]

        for point in table:
            if x <= point < y:
                table[point] = max(table[point], h)
        table[x] = h
        table[y] = 0

    curr = 0
    for k in sorted(table.keys()):
        if table[k] != curr:
            output.append([k, table[k]])
            curr = table[k]
    return output

print "get sky line...", getSkyline([[2, 9, 10], [3, 7, 15], [5, 12, 12],
                                     [15, 20, 10], [19, 24, 8]])


# HARD: Median of two sorted array
# https://leetcode.com/problems/median-of-two-sorted-arrays/solution/
# divide 2 parts equal length: search in 0, m (min size) with i and j
# if A[i] >= B[j-1] and B[j] >= A[i-1] then we found i
# else increase i, decrease j
# median = max left + min right / 2 => max(A[i-1], B[j-1]) + min(A[i], B[j])/2
def median(A, B):
    m, n = len(A), len(B)
    if m > n:
        A, B, m, n = B, A, n, m
    if n == 0:
        raise ValueError

    imin, imax, half_len = 0, m, (m + n + 1) / 2
    while imin <= imax:
        i = (imin + imax) / 2
        j = half_len - i
        if i < m and B[j-1] > A[i]:
            # i is too small, must increase it
            imin = i + 1
        elif i > 0 and A[i-1] > B[j]:
            # i is too big, must decrease it
            imax = i - 1
        else:
            # i is perfect

            if i == 0:
                max_of_left = B[j-1]
            elif j == 0:
                max_of_left = A[i-1]
            else:
                max_of_left = max(A[i-1], B[j-1])

            if (m + n) % 2 == 1:
                return max_of_left

            if i == m:
                min_of_right = B[j]
            elif j == n:
                min_of_right = A[i]
            else:
                min_of_right = min(A[i], B[j])

            return (max_of_left + min_of_right) / 2.0


# HARD: Word search in 2D matrix
# give alist of word, find each word that appear in matrix
# https://leetcode.com/problems/word-search-ii/description/
# Use Trie to store list word and use DFS
class TrieNode:
    def __init__(self):
        self.children = [None for _ in range(26)]
        self.word = None


class Solution:
    def buildTrie(self, words):
        root = TrieNode()
        for word in words:
            p = root
            for char in word:
                idx = ord(char) - ord('a')
                if not p.children[idx]:
                    p.children[idx] = TrieNode()
                p = p.children[idx]
            p.word = word
        return root

    def dfs(self, matrix, i, j, p, res):
        c = matrix[i][j]
        if c == '#' or not p.children[ord(c) - ord('a')]:
            return
        p = p.children[ord(c) - ord('a')]
        if p.word:
            res.append(p.word)
            p.word = None

        matrix[i][j] = '#'
        if i > 0:
            self.dfs(matrix, i - 1, j, p, res)
        if j > 0:
            self.dfs(matrix, i, j - 1, p, res)
        if i < len(matrix) - 1:
            self.dfs(matrix, i + 1, j, p, res)
        if j < len(matrix[0]) - 1:
            self.dfs(matrix, i, j+1, p, res)
        matrix[i][j] = c

    def findWords(self, board, words):
        trie = self.buildTrie(words)
        res = []

        for i in range(len(board)):
            for j in range(len(board[0])):
                self.dfs(board, i, j, trie, res)
        return res


# Count smaller numbers to the right of current number
# https://leetcode.com/problems/count-of-smaller-numbers-after-self/description/
# Brute force: for each number loop and count to the right
# [5,2,6,1] -> [2, 1, 1, 0] => O(n^2)
# Use modified merge sort, sort by value and keep its index, every time we
# merge, update number of smaller = len(right) if value left > value right
def countSmaller(nums):
    def sort(enum):
        half = len(enum) / 2
        if half:
            left, right = sort(enum[:half]), sort(enum[half:])
            for i in range(len(enum))[::-1]:
                if not right or left and left[-1][1] > right[-1][1]:
                    smaller[left[-1][0]] += len(right)
                    enum[i] = left.pop()
                else:
                    enum[i] = right.pop()
        return enum
    smaller = [0] * len(nums)
    sort(list(enumerate(nums)))
    return smaller

# print "count smaller...", countSmaller([5,2,6,1])


# Find third largest in sliding window
def thirdLargest(nums):
    res = []
    minheap = []
    maxheap = []

    for num in nums:
        heappush(maxheap, -num)
        val = - heappop(maxheap)
        heappush(minheap, val)
        if len(minheap) > 3:
            v = heappop(minheap)
            heappush(maxheap, -v)
        if len(minheap) == 3:
            res.append(minheap[0])
    return res


def thirdLargest2(nums):
    res = []
    minheap = []
    for num in nums:
        if len(minheap) < 3 or (minheap and num > minheap[0]):
            heappush(minheap, num)
        if len(minheap) > 3:
            heappop(minheap)
        if len(minheap) == 3:
            res.append(minheap[0])
    return res

print "third largest...", thirdLargest2([10, 5, 3, 6, 9, 12, 24, 11])


# Optimize flights forward and return
def bstLower(val, arr):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (hi + lo)/2
        if arr[mid][1] <= val and (mid == (len(arr) - 1) or arr[mid+1] > val):
            return arr[mid]
        elif arr[mid][1] <= val:
            hi = mid - 1
        else:
            lo = mid + 1
    return arr[mid]


def flight(forwards, returns, target):
    returns.sort(key=lambda x: x[1])
    s = 0
    pairs = []

    for flight in forwards:
        val = target - flight[1]
        v = bstLower(val, returns)
        if v[1] + flight[1] == s:
            pairs.append((flight[0], v[0]))
        elif s < v[1] + flight[1] < target:
            s = v[1] + flight[1]
            pairs = [(flight[0], v[0])]
    return pairs

print "flight...", flight([(1, 2000), (2, 4000), (3, 6000)], [(1, 2000), (2, 4000), (3, 5000)], 7000)
