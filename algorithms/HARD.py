import sys
from collections import defaultdict, deque, Counter
import itertools
from heapq import heappush, heappop
from BinaryIndexedTree import BinaryIndexedTree


# Find largest rectangle area
# https://leetcode.com/problems/largest-rectangle-in-histogram/description/
# example: [2, 1, 5, 6, 2, 3] -> min(5, 6) * 2 = 10
# Bruteforce: find all min in each sub array and multiple with length O(n^2)
# Use stack: if current height > top stack height, push current into stack
# otherwise, pop top stack and calculate area = height[current] * distance
def largestRectangleArea(heights):
    res = 0
    if not heights:
        return res
    stack = []
    i = 0
    while i < len(heights):
        if not stack or heights[i] >= heights[stack[-1]]:
            stack.append(i)
            i += 1
        else:
            idx = stack.pop()
            l = i - stack[-1] - 1 if stack else i
            area = heights[idx] * l
            res = max(res, area)

    while stack:
        idx = stack.pop()
        l = len(heights) - stack[-1] - 1 if stack else len(heights)
        area = heights[idx] * l
        res = max(res, area)
    return res

print "largest rectangle area...", largestRectangleArea([2, 1, 5, 6, 2, 3])


# Word boggle, find words from dictionary in 2D matrix
# HARD: Word search in 2D matrix
# give alist of word, find each word that appear in matrix
# https://leetcode.com/problems/word-search-ii/description/
# Use Trie to store list word and use DFS
class TrieNode:
    def __init__(self):
        self.children = [None for _ in range(26)]
        self.word = None
        self.possibles = []


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


# Word break 2
# memorize the breaking location then using dfs to return list of string
# https://leetcode.com/problems/word-break-ii/description/
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


# Super egg drop
# https://leetcode.com/problems/super-egg-drop/description/
# K egg and N floor, find minimum number of moves to know break floor
# if egg breaks, cannot use it again
# Call C(K, N) is the answer
# for 1 egg drops at flow i: it's not break, then try with N - i upper floor
# if it break, then try with i - 1 floor
# base case: C(1, i) = i, C(0, i) = 0, C(j, 1) = 1
# C(K, N) = 1 + max C(K-1, i-1), C(K, N-i) for i in 1, ..., N
# optimize: dp[k][n] goes up as n increase, dp[k-1][x-1] increase, dp[k][n-x]
# decrease, optimal point is in middle, dp[k-1][x-1] >= dp[k][n-x]
# save value of x for next run to reduce runtime to O(kn)
def superEggDrop(K, N):
    if N <= 0:
        return 0
    dp = [[0] * (N+1) for x in range(K+1)]

    for j in range(1, N+1):
        dp[1][j] = j
        dp[0][j] = 0

    for i in range(1, K+1):
        dp[i][1] = 1

    for k in range(2, K+1):
        for n in range(2, N+1):
            dp[k][n] = 99999
            for x in range(1, n+1):
                val = 1 + max(dp[k-1][x-1], dp[k][n-x])
                dp[k][n] = min(dp[k][n], val)

    return dp[K][N]

print "egg drops...", superEggDrop(3, 14)


# Insert interval, give a list of sorted intervals, insert new interval into
# list and merge if overlap
# HARD: https://leetcode.com/problems/insert-interval/description/
# Loop through list, find all overlap with new interval, and merge them
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __str__(self):
        return '{}->{}'.format(self.start, self.end)


class IntervalSolution(object):
    def isOverlap(self, x, y):
        return (x.end >= y.start) and (y.end >= x.start)

    def merge(self, x, y):
        return Interval(min(x.start, y.start), max(x.end, y.end))

    def insert(self, intervals, newInterval):
        if not intervals:
            return [newInterval]
        res = []
        i, n = 0, len(intervals)
        while i < n:
            if newInterval.end < intervals[i].start:
                break

            if self.isOverlap(newInterval, intervals[i]):
                newInterval = self.merge(newInterval, intervals[i])
            else:
                res.append(intervals[i])
            i += 1

        res.append(newInterval)
        while i < n:
            res.append(intervals[i])
            i += 1
        return res

    def printIntervals(self, intervals):
        l = [str(x) for x in intervals]
        print("intervals...", l)


# Longest substring at most k distinct chars
# https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters
# Use hashmap to store later index of characters, and count number of distinct
# characters, if count exceed k, move lower pointer, check char it passes, if
# index of character equal index in hashmap, decrease count
def longestAtMostKDistinctChars(text, k):
    if not text:
        return 0
    if k <= 1:
        return k

    low, high = 0, 0
    n = len(text)
    table = {}
    count = 0
    res = 0
    while high < n:
        char = text[high]
        if char not in table:
            count += 1
        table[char] = high
        if count > k:
            while count > k:
                lowchar = text[low]
                if low == table[lowchar]:
                    count -= 1
                    table.pop(lowchar)
                low += 1
        else:
            res = max(res, high - low + 1)
        high += 1
    return res

print "longest substring at most k distinct", longestAtMostKDistinctChars("eceded", 2)


# https://leetcode.com/problems/longest-substring-with-at-most-two-distinct-characters
# Optimze for longest substring at most 2 distinct chars, update index of 2
# chars as moving along, if encounter 3rd char that different than 2 chars,
# move low pointer to minimum index of 2 indexes, replace low index chars with
# 3rd chars
def longestSubstringAtMost2DistinctChars(text):
    if not text:
        return 0

    chars = ['', '']
    indices = [-1, -1]
    n = len(text)
    low, high = 0, 0
    curr, res = 0, 0
    while high < n:
        char = text[high]
        if char == chars[0] or not chars[0]:
            curr += 1
            indices[0] = high
            chars[0] = char
        elif char == chars[1] or not chars[1]:
            curr += 1
            indices[1] = high
            chars[1] = char
        # Encounter 3rd char
        else:
            # Find lower indices char, move low pointer to pass this position
            # update current length
            res = max(res, curr)
            minIdx = 0 if indices[0] < indices[1] else 1
            low = indices[minIdx] + 1
            chars[minIdx] = text[high]
            indices[minIdx] = high
            curr = high - low + 1
        high += 1

    return max(res, curr)

print "longest substring at most 2 distinct chars...", longestSubstringAtMost2DistinctChars('eaaddaccccddc')


# Query range 2D array
# HARD: https://leetcode.com/problems/range-sum-query-2d-immutable/description/
# Store sum from 0,0 to i, j by sum[i][j] = sum[i-1][j] + sum[i][j-1] -
# sum[i-1][j-1] + matrix[i][j]
# Query: r1, c1 to r2, c2 = sum[r2][c2] - sum[r2][c1] -
# sum[r1][c2] + sum[r1][c1]
# Follow up: support update(r, c, value), calculate difference and add up to
# sub array with index equal or larger than current position
# Second solution: do prefix for each row, calculate sum by sum prefix for each
# row, update can do by re-calculate prefix for that row only
class NumMatrix(object):
    def __init__(self, matrix):
        m, n = len(matrix), len(matrix[0])
        prefix = [[0]*(n+1) for i in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                prefix[i][j] = (prefix[i-1][j] + prefix[i][j-1] -
                                prefix[i-1][j-1] + matrix[i-1][j-1])
        self.prefix = prefix
        self.nums = matrix
        self.row = m
        self.col = n

    def sumRegion(self, r1, c1, r2, c2):
        if (not 0 <= r1 < self.row or not 0 <= r2 < self.row
            or not 0 <= c1 < self.col or not 0 <= c2 < self.col):
            return None
        prefix = self.prefix
        return (prefix[r2+1][c2+1] - prefix[r2+1][c1] -
                prefix[r1][c2+1] + prefix[r1][c1])

    def update(self, r, c, value):
        prefix = self.prefix
        nums = self.nums
        delta = value - nums[r][c]
        nums[r][c] = value
        for i in range(r+1, self.row+1):
            for j in range(c+1, self.col+1):
                prefix[i][j] += delta


arr = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
nm = NumMatrix(arr)
print "query by region sum...\n", nm.sumRegion(1, 1, 2, 2)
print "update matrix...\n", nm.update(1, 1, 10)
print "query by region sum after...\n", nm.sumRegion(1, 1, 2, 2)


# Guess words
# HARD: https://leetcode.com/problems/guess-the-word/
# use random algorithm, every time guess a random word, keep the list with
# same match
# Minimax: compare 2 words in lists, and guess word with minimum match with
# the rest
class GuessSolution:
    def match(self, a, b):
        match = 0
        for i in range(len(a)):
            if a[i] == b[i]:
                match += 1
        return match

    def findSecretWord(self, wordlist, master):
        n = 0
        while n < 6:
            count = Counter(w1 for w1, w2 in itertools.permutations(wordlist, 2) if self.match(w1, w2) == 0)
            guess = min(wordlist, key=lambda w: count[w])
            n = master.guess(guess)
            wordlist = [w for w in wordlist if self.match(w, guess) == n]


# Word square
# HARD: https://leetcode.com/problems/word-squares
# Use trie to store each word, for each word, check if every letter of it
# has other word with begin letter, use dfs to find list of words
class WordSquareSolution:
    def buildTrie(self, words):
        root = TrieNode()
        for word in words:
            p = root
            for char in word:
                idx = ord(char) - ord('a')
                if not p.children[idx]:
                    p.children[idx] = TrieNode()
                p = p.children[idx]
                p.possibles.append(word)
            p.word = word
        return root

    def dfs(self, wordlist, root, n):
        if len(wordlist) == n:
            return wordlist

        # next word index
        curr = len(wordlist)
        node = root
        for i in range(curr):
            char = wordlist[i][curr]
            idx = ord(char) - ord('a')
            if node.children[idx]:
                node = node.children[idx]
            else:
                return None

        if not node or not node.possibles:
            return None

        possibles = node.possibles
        for p in possibles:
            res = self.dfs(wordlist + [p], root, n)
            if res:
                return res
        return None

    def wordSquare(self, words):
        if not words:
            return []
        trie = self.buildTrie(words)
        candidates = []

        for word in words:
            for i in range(1, len(word)):
                idx = ord(word[i]) - ord('a')
                if not trie.children[idx]:
                    break
            else:
                candidates.append(word)

        res = []
        size = len(words[0])
        for word in candidates:
            l = self.dfs([word], trie, size)
            if l:
                res.append(l)
        return res

wq = WordSquareSolution()
print "square word...", wq.wordSquare(['area', 'ball', 'dear', 'lady', 'lead', 'yard'])


# Trap rain water
# HARD: https://leetcode.com/problems/trapping-rain-water/solution/
# Brute force: the amount of trap rain water equal to minimum of max left column
# and max right column - height of current column, go 2 passes to find max
# left and max right, => run time O(n^2)
# Use 2 max array to avoid run max left and max right multiple times
def trapRainWater(heights):
    if not heights:
        return 0
    length = len(heights)
    maxLeft = [0 for _ in range(length)]
    maxLeft[0] = heights[0]
    for i in range(1, length):
        maxLeft[i] = max(maxLeft[i-1], heights[i])

    maxRight = [0 for _ in range(length)]
    maxRight[length-1] = heights[length-1]
    for i in range(length-2, -1, -1):
        maxRight[i] = max(maxRight[i+1], heights[i])
    res = 0

    for i in range(length):
        res += min(maxLeft[i], maxRight[i]) - heights[i]

    return res

print "trap rain water...", trapRainWater([1,3,2,4,1,3,1,4,5,2,2,1,4,2,2])


# HARD: Closest K binary search tree value
# https://leetcode.com/problems/closest-binary-search-tree-value-ii/description/
# Use 2 stacks to store predecessor and successor as traversing
# Compare 2 stacks top values and choose nearest values
# As getting one value from stack, continue to push right successor or left
# predecessor to current stack
class ClosestSolution:
    def closestKValues(self, root, target, k):
        """
        :type root: TreeNode
        :type target: float
        :type k: int
        :rtype: List[int]
        """
        if not root:
            return []

        predecessor = []
        successor = []
        result = []
        curr = root
        while curr:
            if curr.val >= target:
                successor.append(curr)
                curr = curr.left
            else:
                predecessor.append(curr)
                curr = curr.right
        while k > 0:
            if not predecessor and not successor:
                break
            if not predecessor:
                self.getSuccessor(successor, result)
            elif not successor:
                self.getPredecessor(predecessor, result)
            elif abs(predecessor[-1].val - target) > abs(successor[-1].val - target):
                self.getSuccessor(successor, result)
            else:
                self.getPredecessor(predecessor, result)
            k -= 1
        return result

    def getPredecessor(self, stack, result):
        node = stack.pop()
        result.append(node.val)
        curr = node.left
        while curr:
            stack.append(curr)
            curr = curr.right

    def getSuccessor(self, stack, result):
        node = stack.pop()
        result.append(node.val)
        curr = node.right
        while curr:
            stack.append(curr)
            curr = curr.left


# Minimum cost to hire K worker
# https://leetcode.com/problems/minimum-cost-to-hire-k-workers/description/
# Brute force: every time choose worker as captain, calculate other workers as
# ratio with current work, sort the list and sum prices
# Optimize: use max heap for quality, calculate ratio price/quality and sort
# in order, every time put a quality into heap, if heap larger than K, then pop
# item with larger quality out
class HireWorkerSolution:
    def mincostToHireWorkers(self, quality, wage, K):
        """
        :type quality: List[int]
        :type wage: List[int]
        :type K: int
        :rtype: float
        """
        if not quality or not wage or K <= 0:
            return -1

        if K == 1:
            return min(wage)

        workers = sorted([(float(w)/q, w, q) for w, q in zip(wage, quality)])
        pool = []
        sumq = 0
        ans = float('inf')

        for ratio, w, q in workers:
            heappush(pool, -q)
            sumq += q

            if len(pool) > K:
                sumq += heappop(pool)

            if len(pool) == K:
                ans = min(ans, sumq*ratio)
        return ans


# K empty slots
# HARD: https://leetcode.com/problems/k-empty-slots
# Given bloom array: [1, 3, 2] day 1 flower 1, day 2 flower 3, day 3 flower 2
# Given k find which day, there's two blooming and k in between not blooming
# Convert to array: days[x] = i for flower x bloom at day i
# Find interval: left, right which is minimum bloom day of this interval,
# solution 1: use sliding min queue, min of window > max(left, right)
# solution 2: sliding window, for interval left, right, if found days[i] < left
# or days[i] < right, update the window to i, i+k+1
def kEmptySlots(flowers, k):
    days = [0]*len(flowers)

    for day, flower in enumerate(flowers, 1):
        days[flower - 1] = day

    left, right = 0, k + 1
    res = len(flowers) + 1
    while right < len(days):
        for i in range(left, right+1):
            if days[i] < days[left] or days[i] < days[right]:
                left, right = i, i + k + 1
                break
        else:
            res = min(res, max(left, right))
            left, right = right, right + k + 1
    return res if res < len(flowers) + 1 else -1


# HARD: Reverse pair:
# https://leetcode.com/problems/reverse-pairs/
# Use binary tree insert
class Node(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        self.count_ge = 1


def insert(head, val):
    if not head:
        return Node(val)

    if val == head.val:
        head.count_ge += 1
    elif val < head.val:
        head.left = insert(head.left, val)
    else:
        head.count_ge += 1
        head.right = insert(head.right, val)
    return head


def search(head, val):
    if not head:
        return 0

    if val == head.val:
        return head.count_ge
    elif val < head.val:
        return head.count_ge + search(head.left, val)
    else:
        return search(head.right, val)


class ReversePairSolution:
    def reversePairs(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        n = len(nums)
        head = None
        count = 0
        for i in range(n):
            count += search(head, nums[i]*2 + 1)
            head = insert(head, nums[i])
        return count


def index(arr, val):
    lo, hi = 0, len(arr) - 1
    while lo < hi:
        mid = lo + (hi - lo)//2
        if arr[mid] >= val:
            hi = mid
        else:
            lo = mid + 1
    return lo


# Second solution use BIT
class ReversedSolution2:
    def reversePairs(self, nums):
        table = sorted(nums)
        n = len(table)
        tree = BinaryIndexedTree(n)
        count = 0
        for i in range(n-1, -1, -1):
            idx = index(table, nums[i]/2)
            count += tree.search(idx)
            curr = index(table, nums[i])
            tree.insert(curr, 1)
        return count


# Third solution use merge sort
def mergeCount(nums, left, right):
    if left >= right:
        return 0

    mid = left + (right - left)//2
    res = mergeCount(nums, left, mid) + mergeCount(nums, mid+1, right)
    j = mid + 1
    for i in range(left, mid+1):
        while j <= right and nums[i]/2.0 > nums[j]:
            j += 1
            res += j - (mid + 1)
    # merge 2 parts
    return res


# Serialize and deserialize tree
# https://leetcode.com/problems/serialize-and-deserialize-binary-tree/description/
# use queue to serialize pre-order, None -> null, create full tree string
# split string by comma "," and use queue to deserialize tree
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        out = []
        q = deque([root])
        while q:
            item = q.popleft()
            if not item:
                out.append('null')
                continue
            out.append(str(item.val))
            q.append(item.left)
            q.append(item.right)
        return ','.join(out)

    def makeNode(self, val):
        return Node(int(val)) if val != 'null' else None

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        arr = data.split(',')
        root = self.makeNode(arr[0])
        q = deque([root])
        i = 1
        while q and i < len(arr):
            item = q.popleft()
            if not item:
                continue
            item.left = self.makeNode(arr[i])
            item.right = self.makeNode(arr[i+1])
            q.append(item.left)
            q.append(item.right)
            i += 2
        return root


# Median finder in infinite stream
# https://leetcode.com/problems/find-median-from-data-stream/description/
# Use 2 heaps, small and large heaps, add number to large heap and pop small to
# small heap
class MedianFinder(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.low = []
        self.high = []
        self.count = 0

    def addNum(self, num):
        """
        :type num: int
        :rtype: void
        """
        self.count += 1
        heappush(self.high, num)
        k = heappop(self.high)
        heappush(self.low, -k)
        if len(self.low) > len(self.high):
            m = heappop(self.low)
            heappush(self.high, -m)

    def findMedian(self):
        """
        :rtype: float
        """
        if self.count % 2 == 1:
            return float(self.high[0])
        return (self.high[0] - self.low[0]) / 2.0


# Word pattern 2
# HARD: https://leetcode.com/problems/word-pattern-ii/
# Use backtracking together with hash table to check valid, use another hash
# map to avoid duplicate, a and b map to same text e.g
class PatternSolution:
    def bs(self, pattern, pidx, text, sidx, table, rev):
        if pidx == len(pattern) and sidx == len(text):
            return True
        elif pidx == len(pattern) or sidx == len(text):
            return False
        char = pattern[pidx]
        print(table, pidx, sidx, rev)
        if char in table:
            s = table[char]
            if text[sidx: sidx + len(s)] == s:
                return self.bs(pattern, pidx + 1, text, sidx + len(s), table, rev)
            else:
                return False
        else:
            for idx in range(sidx + 1, len(text)+1):
                part = text[sidx: idx]
                if part in rev:
                    continue

                table[char] = part
                rev[part] = char

                if self.bs(pattern, pidx + 1, text, idx, table, rev):
                    return True
                rev.pop(part, None)
                table.pop(char, None)
        return False

    def wordPatternMatch(self, pattern, str):
        """
        :type pattern: str
        :type str: str
        :rtype: bool
        """
        return self.bs(pattern, 0, str, 0, {}, {})

wp = PatternSolution()
print "word pattern match...", wp.wordPatternMatch("aba", "aaaa")


# Find maximum values in sliding windows
# https://leetcode.com/problems/sliding-window-maximum/description/
# example: [1, 3, -1, -3, 5, 3, 6, 7], k= 3 -> 3 -> 3, 3, 5, 5, 6, 7
# Bruteforce: find max in k window -> O(nk)
# Use a deque to store maximum value idx at first index, second max idx
# whenever add elem to queue compare it with the rear value, if larger than,
# remove rear, when max idx fallout of window pop it out
# (keep max values in each sub array)
def maxSlidingWindow(nums, k):
    if not nums or k <= 0 or k > len(nums):
        return []
    queue = deque()
    output = []

    for i in range(k):
        while queue and (nums[i] >= nums[queue[-1]]):
            queue.pop()
        queue.append(i)
    output.append(nums[queue[0]])

    for i in range(k, len(nums)):
        while queue and (queue[0] <= i - k):
            queue.popleft()

        while queue and (nums[i] >= nums[queue[-1]]):
            queue.pop()
        queue.append(i)
        output.append(nums[queue[0]])
    return output

print "max sliding window...", maxSlidingWindow([7, 2, 4], 2)


# Alien dictionary
# https://www.geeksforgeeks.org/given-sorted-dictionary-find-precedence-characters/
# Create graph with each  vertex is char, compare each word pair, if missmatch
# add it as edge
class Graph(object):
    def __init__(self):
        self.adjections = defaultdict(set)

    def addEdge(self, _from, to):
        self.adjections[_from].add(to)

        if to not in self.adjections:
            self.adjections[to] = set()

    def runDfs(self, vertex):
        print "run dfs..", vertex, self.status
        # Discover
        self.status[vertex] = 1

        for adj in self.adjections[vertex]:
            if self.status[adj] == 0:
                self.runDfs(adj)

        self.status[vertex] = 2
        self.stack.append(vertex)

    def topo_sort(self):
        self.stack = []
        self.status = {k: 0 for k in self.adjections}
        for vertex in self.adjections:
            if self.status[vertex] == 0:
                self.runDfs(vertex)
        return self.stack

    def __str__(self):
        m = ['{} -> {}'.format(k, v) for (k, v) in self.adjections.items()]
        return '\n'.join(m)


class AlienDictionary(object):
    def buildGraph(self, words):
        graph = Graph()

        for pair in zip(words, words[1:]):
            a, b = pair
            for i in range(min(len(a), len(b))):
                if a[i] != b[i]:
                    graph.addEdge(a[i], b[i])
                    break
        return graph

    def findOrder(self, words):
        graph = self.buildGraph(words)
        print "graph...", graph
        topo = graph.topo_sort()
        return topo[::-1]


# Smallest range to cover all sorted arrays list element
# https://leetcode.com/problems/smallest-range
# Use a min heap to push each elements from list
# Each time pop a smallest element, calculate range = maxValue - pop item value
# increment next index and push into heap
class SmallestRangeSolution:
    def smallestRange(self, nums):
        """
        :type nums: List[List[int]]
        :rtype: List[int]
        """
        k = len(nums)
        heap = []
        res = float('inf')
        out = [None, None]
        maxVal = float('-inf')

        for i in range(k):
            heappush(heap, (nums[i][0], i, 0))
            maxVal = max(maxVal, nums[i][0])

        while heap:
            v, i, j = heappop(heap)

            if maxVal - v < res:
                res = maxVal - v
                out[0], out[1] = v, maxVal

            if j >= len(nums[i]) - 1:
                break

            heappush(heap, (nums[i][j+1], i, j+1))
            maxVal = max(maxVal, nums[i][j+1])
        return out


# Basic calculator with () and + -
# https://leetcode.com/problems/basic-calculator/
# Recursive call calculate if character is ( and return call when character is
# ) or end of char
# Bug: handle end of string
class BasicCalculatorSolution:
    def _calc(self, s, idx, total):
        if idx == len(s) or s[idx] == ')':
            return total, idx + 1

        curr = ''
        sign = 1

        while idx < len(s):
            char = s[idx]
            # print("total, idx, char...", total, idx, char)
            if char == ' ':
                idx += 1
                continue
            if char == '(':
                val, nidx = self._calc(s, idx+1, 0)
                total += val * sign
                idx = nidx
                sign = 1

            elif char.isdigit():
                curr += char
            elif char in ('+', '-'):
                if curr:
                    total += int(curr) * sign
                    curr = ''
                sign = 1 if char == '+' else -1
            else:
                break
            idx += 1
        if curr:
            total += int(curr) * sign
        return total, idx

    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        total, _ = self._calc(s, 0, 0)
        return total


# Median of 2 sorted array
# https://leetcode.com/problems/median-of-two-sorted-arrays/
# A, B is 2 array, suppose A < B
# median A0...Ai-1 and Ai....Am
# median B0...Bj-1 and Bj....Bn
# Find i, j for A(i-1) < Bj and B(j-1) < Ai
# i + j = m + n - i - j or (m + n + 1 - i -j) => j = (m + n + 1)//2 - i
# if A(i-1) > B(j) => decrease i => increase j
# if B(j-1) > A(i) => increase i => decrease j
class MedianSolution:
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        A, B = nums1, nums2
        m, n = len(nums1), len(nums2)
        if m > n:
            A, B, m, n = nums2, nums1, n, m

        imin, imax = 0, m
        while imin <= imax:
            i = (imin + imax)//2
            j = (m + n + 1)//2 - i

            if i > 0 and A[i-1] > B[j]:
                imax = i - 1
            elif i < m and B[j-1] > A[i]:
                imin = i + 1
            else:
                left, right = None, None
                if i <= 0:
                    left = B[j-1]
                elif j <= 0:
                    left = A[i-1]
                else:
                    left = max(A[i-1], B[j-1])

                if (m + n) % 2 == 1:
                    return left

                if i >= m:
                    right = B[j]
                elif j >= n:
                    right = A[i]
                else:
                    right = min(A[i], B[j])

                return (left + right)/2.0
        return None


# Put N Queen on board for them not conflict
# https://leetcode.com/problems/n-queens/
# Use backtracking, store all location previous set, check if new Queen not
# violate condition x, y, x+y, y-x; backtrack for next (x+1, 0)
# Optimize: observe that each Queen has to be on each different row
# need to store only column that Queen occupies, and backtrack to next row
# and check for previous Queens
class QueenSolution:
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        used = []
        res = []

        def bs(m):
            if m == n:
                vals = []
                for v in used:
                    vals.append('.' * v + 'Q' + '.' * (n-v-1))
                res.append(vals)
                return

            # Consider only column
            for col in range(n):
                for row, u in enumerate(used):
                    if col == u or m + col == row + u or u - row == col - m:
                        break
                else:
                    used.append(col)
                    bs(m+1)
                    used.pop()

        bs(0)
        return res


# Longest valid parenthesis
# https://leetcode.com/problems/longest-valid-parentheses/
# consider simple case )()(), if char ( length = 0, if char ) check previous
# char, if it is (, then there's match, dp[i] = 2 + dp[i-2]
# otherwise if there's valid sequence dp[i-1] > 0, check char before valid
# sequence i - dp[i-1] - 1, if match (
# then dp[i] = 2 + dp[i-1] + dp[i - dp[i-1] - 2]
class LongestParenthesesSolution(object):
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        n = len(s)
        dp = [0 for i in range(n)]
        res = 0

        for i in range(n):
            if s[i] == '(':
                dp[i] = 0
            if s[i] == ')' and i > 0:
                if s[i-1] == '(':
                    dp[i] = 2 + dp[i-2]
                elif dp[i-1] > 0:
                    prev = i - dp[i-1] - 1
                    if prev >= 0 and s[prev] == '(':
                        dp[i] = 2 + dp[i-1] + dp[prev - 1]

            res = max(res, dp[i])
        return res


# Sudoku solver
# https://leetcode.com/problems/sudoku-solver/
# Use 3 arrays to store used numbers for rows, columns and sub squares
# each row (or columns, sub square) is 10 bit numbers to indicate which number
# in row by set and unset bit, to check which sub square that current cell
# is in = i/3*3 + j/3
# to set bit n: v | 1 << n, to unset bit: v & ~(1 << n), to check bit:
# v >> n & 1
class SudokuSolution(object):
    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        vals = [[0]*9 for i in range(3)]
        q = []

        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    q.append((i, j))
                else:
                    v = 1 << int(board[i][j])
                    s = i/3 * 3 + j/3
                    vals[0][i] |= v
                    vals[1][j] |= v
                    vals[2][s] |= v

        def backtrack(idx):
            if idx >= len(q):
                return True

            x, y = q[idx]
            s = x/3 * 3 + y/3
            for v in range(1, 10):
                t = 1 << v
                if ((vals[0][x] >> v & 1 != 0) or (vals[1][y] >> v & 1 != 0) or
                    (vals[2][s] >> v & 1 != 0)):
                    continue
                vals[0][x] |= t
                vals[1][y] |= t
                vals[2][s] |= t

                board[x][y] = str(v)
                if backtrack(idx + 1):
                    return True

                board[x][y] = '.'
                t = ~t
                vals[0][x] &= t
                vals[1][y] &= t
                vals[2][s] &= t
            return False

        backtrack(0)


# Palindrome pair https://leetcode.com/problems/palindrome-pairs/
# Find all pairs in words list that combine to form palindrome
# use hash map to store all words
# s1 = ba, s2 = abc => abc + ba = abcba, reversed prefix ab
# check if exist in hash table and suffix is also palindrome
# s1 = cb, s2 = abc => cb + abc = cbabc, reversed suffix cb, do same thing
# edge case: empty string => e.g '' and 'a'
# whole string: 'abcd' and 'dcba', check prefix and suffix from 0 to n
class PalindromePairSolution(object):
    def palindromePairs(self, words):
        """
        :type words: List[str]
        :rtype: List[List[int]]
        """

        def isPalindrome(s):
            if len(s) <= 1:
                return True

            i, j = 0, len(s) - 1
            while i <= j:
                if s[i] != s[j]:
                    return False
                i += 1
                j -= 1
            return True

        table = {word: i for i, word in enumerate(words)}
        res = []
        for word, idx in table.iteritems():
            n = len(word)
            for i in range(n+1):
                pre = word[:i]
                suf = word[i:]

                if isPalindrome(pre):
                    r = suf[::-1]
                    if r != word and r in table:
                        res.append([table[r], idx])
                if i != n and isPalindrome(suf):
                    r = pre[::-1]
                    if r != word and r in table:
                        res.append([idx, table[r]])
        return res


# Minimize largest continuous sum array
# https://leetcode.com/problems/split-array-largest-sum/
# Call dp[i][j] is largest sum at index i, split into j array
# dp[i][j] = min( max(dp[x][j-1], sum x...i for x in range j - 1, i - 1)
# Base case: dp[i][1] = sum 0...i
# dp[1][j] = arr[0]
# Optimize: use binary search
# min value is max(arr), max value is sum(arr)
# use bst for value of mid = (lo + high)/2 and check if can make more than
# m cut then set high = mid, if cannot set lo = mid + 1
# Run time: check if can cut O(n), binary search O(log(sum(arr)))
class SplitArraySolution:
    def splitArray(self, nums, m):
        """
        :type nums: List[int]
        :type m: int
        :rtype: int
        """
        n = len(nums)
        dp = [[sys.maxint for i in range(m)] for j in range(n+1)]
        dp[0][0] = 0

        acc = 0
        for i in range(1, n+1):
            acc += nums[i-1]
            dp[i][0] = acc

        for j in range(m):
            dp[0][j] = 0

        for i in range(1, n+1):
            for j in range(1, m):
                for x in range(j-1, i):
                    dp[i][j] = min(dp[i][j], max(dp[x][j-1], dp[i][0] - dp[x][0]))
        return dp[n][m-1]

    # Binary search solution
    def splitArray2(self, nums, m):
        def valid(mid):
            cnt = 0
            current = 0
            for n in nums:
                current += n
                if current > mid:
                    cnt += 1
                    if cnt >= m:
                        return False
                    current = n
            return True

        l = max(nums)
        h = sum(nums)

        while l < h:
            mid = l+(h-l)/2
            if valid(mid):
                h = mid
            else:
                l = mid+1
        return l

s = SplitArraySolution()
print "split array...", s.splitArray([7, 2, 5, 10, 8], 4)


# Sliding Puzzle problem
# https://leetcode.com/problems/sliding-puzzle/
# Run bfs on board using queue
class SlidingPuzzleSolution(object):
    def slidingPuzzle(self, board):
        """
        :type board: List[List[int]]
        :rtype: int
        """

        finished = [1, 2, 3, 4, 5, 0]
        states = set()

        v = board[0] + board[1]
        pos = v.index(0)
        ans = 0
        q = deque([(pos, v)])

        while q:
            nq = deque()
            while q:
                pos, s = q.popleft()
                if s == finished:
                    return ans
                states.add(tuple(s))

                if pos % 3 > 0:
                    t = s[:]
                    t[pos], t[pos-1] = t[pos-1], t[pos]
                    if tuple(t) not in states:
                        nq.append((pos-1, t))

                if pos % 3 < 2:
                    t = s[:]
                    t[pos], t[pos+1] = t[pos+1], t[pos]
                    if tuple(t) not in states:
                        nq.append((pos+1, t))

                if pos < 3:
                    t = s[:]
                    t[pos], t[pos+3] = t[pos+3], t[pos]
                    if tuple(t) not in states:
                        nq.append((pos+3, t))

                if pos >= 3:
                    t = s[:]
                    t[pos], t[pos-3] = t[pos-3], t[pos]
                    if tuple(t) not in states:
                        nq.append((pos-3, t))
            q = nq
            ans += 1
        return -1


# HARD: race car
# https://leetcode.com/problems/race-car/
# A: accelerate speed, pos += speed, speed *= 2
# R: reverse, if positive speed = -1, negative speed = 1
# Give target find short list of commands
# https://leetcode.com/problems/race-car/
# Use BFS to store location and current speed, runtime: 2^n
# There's overlap: use hashset to store
def racecar(target):
    if not target:
        return 0

    q = [(0, 1)]
    visited = {(0, 1)}
    lvl = 0

    while q:
        nq = []
        print("current queue...", q)
        for item in q:
            pos, speed = item
            if pos == target:
                return lvl

            acc = (pos + speed, speed*2)
            rev = (pos, -1 if speed > 0 else 1)

            if acc not in visited and acc[0] > 0:
                nq.append(acc)
                visited.add(acc)

            if rev not in visited and rev[0] > 0:
                nq.append(rev)
                visited.add(rev)
        q = nq
        lvl += 1
    return -1

print("race car...", racecar(5))
