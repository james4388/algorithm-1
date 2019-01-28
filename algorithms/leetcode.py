import math
from linkedlist import ListNode, LinkedList
from collections import defaultdict, deque, Counter
from Queue import Queue
from tree import TreeNode
import sys
import string

MAX = sys.maxint
MIN = -sys.maxint - 1


# Given 2 number x,y find the distance bit
# e.g 1 = 0001, 4 = 0100 -> 2
def hammingDistance(x, y):
    distance = 0

    _min = min(x, y)
    _max = max(x, y)

    while _min > 0:
        p = _min % 2
        q = _max % 2
        if p != q:
            distance += 1

        _min = _min/2
        _max = _max/2

    while _max > 0:
        q = _max % 2
        if q != 0:
            distance += 1
        _max = _max/2

    return distance


# print "Hamming distance ....", hammingDistance(27, 4)


# Game 100, 2 players can choose 1...10 to add up to 100
# first player add up to total win
# Given max number and total check if first player can force win
# S list of numbers, n total
# Sub problem: S - s(i), n - s(i) if true then parent = false
# Note each number is not re-used
# https://leetcode.com/problems/can-i-win/discuss/95292/Python-solution-easy-to-understand
def chooseNumber(nums, total, memo):
    key = str(nums)
    if key in memo:
        return memo[key]

    if nums[-1] >= total:
        return True

    for i in range(len(nums)):
        if not chooseNumber(nums[:i] + nums[i+1:], total - nums[i]):
            memo[key] = True
            return True

    memo[key] = False
    return False


def canWinGame(maxChoose, total):
    # Total values not equal desired value
    if (maxChoose + 1)*maxChoose/2 < total:
        return False

    return chooseNumber(range(1, maxChoose + 1), total, {})


# Add two numbers in 2 reversed order
def addTwoNumbers(self, l1, l2):
    """
    :type l1: ListNode
    :type l2: ListNode
    :rtype: ListNode
    """
    r = 0
    prev = None
    head = None
    p, q = l1, l2
    while (p is not None or q is not None):
        i = p.val if p else 0
        j = q.val if q else 0
        x = i + j + r
        r = x/10
        x = x % 10
        n = ListNode(x)
        prev = n

        if p:
            p = p.next
        if q:
            q = q.next

    # Edge case, when 2 last number sum > 10
    if r != 0:
        n = ListNode(r)
        if prev:
            prev.next = n

    return head

# l1 = ListNode(0, ListNode(8))
# l2 = ListNode(1)
# addTwoNumbers(l1, l2)


# Longest substring without repeating
# Loop through string, for each char, check its previous chars
# if duplicate, start checking from non-duplicate char
def longestSubstring(s):
    idx = 0
    _max = 0
    for i in range(0, len(s)):
        # Find nearest duplicate char
        # Improve: use dict to store last index of char
        v = s.rfind(s[i], idx, i)
        if v == -1:
            _max = max(i - idx + 1, _max)
        else:
            idx = v + 1

    return _max

# print "longest substring...", longestSubstring('dvdf')


# Longest palidromatic substring
# Given string x1 x2 ... xk is palindrome substring
# x1 = xk, L = 2 + L(x2, x(k-1))
# x1 != xk, L = L(x2, x(k-1))
# dynamic moving window
def longestPalindromSubstring(s):
    n = len(s)
    arr = [[0]*n for x in range(n)]
    _max = 0
    idx = None

    for i in range(n):
        arr[i][i] = 1

    for l in range(2, n+1):
        for i in range(n - l + 1):
            j = i + l - 1
            if s[i] == s[j]:
                # There's no substring
                if j == i+1:
                    arr[i][j] = 2
                # Check if substring is palindrome
                elif arr[i+1][j-1] != 0:
                    arr[i][j] = arr[i+1][j-1] + 2

                if arr[i][j] > _max:
                    _max = arr[i][j]
                    idx = (i, j)
            else:
                arr[i][j] = 0

    return _max, idx

# print "longest Palindrom...", longestPalindromSubstring("babad")


# Count pair with difference k
# Given array count number of pair, non-duplicate with difference value k
# Solution sort array, use bst search time nlogn
def bstSearch(nums, n, low, high):
    if low > high:
        return -1

    mid = low + (high - low)/2
    if nums[mid] == n:
        return mid

    elif nums[mid] > n:
        return bstSearch(nums, n, low, mid-1)
    else:
        return bstSearch(nums, n, mid+1, high)

    return -1


def countPair(arr, k):
    nums = sorted(arr)
    count = 0
    n = len(nums)
    for i in range(n):
        if bstSearch(nums, nums[i] + k, i + 1, n - 1) != -1:
            count += 1

    return count

# print "count pair....", countPair([1, 3, 4, 5, 2, 8], 3)


# Merge k sorted linkedlist
# https://leetcode.com/problems/merge-k-sorted-lists/description/
# use heap to store all linkedlist
# or: divide and conquer array into 2 half
def heapify(arr, k):
    if not arr:
        return arr

    l = 2*k + 1
    r = 2*k + 2

    idx = k
    if l < len(arr) and arr[idx].val > arr[l].val:
        idx = l

    if r < len(arr) and arr[idx].val > arr[r].val:
        idx = r

    if idx != k:
        arr[k], arr[idx] = arr[idx], arr[k]
        heapify(arr, idx)

    return arr


def build_heap(arr):
    n = len(arr)
    for x in range(n/2, -1, -1):
        heapify(arr, x)

    return arr


def pop_heap(arr):
    arr[0], arr[-1] = arr[-1], arr[0]
    return arr[:-1]


def mergeKLists(lists):
    s = [x for x in lists if x is not None]
    s = build_heap(s)
    dummy = ListNode(None)
    tmp = dummy
    while len(s) > 0:
        n = s[0]
        if n is not None:
            node = ListNode(n.val)
            tmp.next = node
            tmp = node
            if n.next is None:
                s = pop_heap(s)
            else:
                s[0] = n.next
            heapify(s, 0)

    head = dummy.next

    return head


# m = LinkedList.from_array([1, 4, 5])
# n = LinkedList.from_array([1, 3, 4])
# o = LinkedList.from_array([2, 6])
# lists = [m.head, n.head, o.head]
# mergeKLists(lists)

# Reverse k group in linked list
# https://leetcode.com/problems/reverse-nodes-in-k-group
# Count k element, detach last element and reverse k nodes,
def reverse_linkedlist(head):
    curr = head
    prev = None
    while curr is not None:
        curr.next, prev, curr = prev, curr, curr.next
    return prev, head


def reverseKGroup(head, k):
    curr = head
    prev = None

    while curr is not None:
        point = last = curr
        i = k
        # Get the last node of k range
        while i > 1 and last is not None:
            last = last.next
            i -= 1

        if last is None:
            if prev:
                prev.next = point
            break

        curr = last.next
        # Detach part of linkedlist and do reverse
        last.next = None
        left, right = reverse_linkedlist(point)

        # Concatinate this part to previous part using pre pointer
        if not prev:
            head = left
        else:
            prev.next = left

        prev = right
    return head


def print_list(head):
    tmp = head
    while tmp is not None:
        tmp = tmp.next

# arr = LinkedList.from_array(range(25))
# head = arr.head
# head = reverseKGroup(head, 4)
# print_list(head)


# Find index of all concatinated words list in long string
def is_substr(s, left, right, word_len, m):
    k = left
    while k < right:
        w = s[k: k + word_len]
        if w not in m:
            return False
        m[w] -= 1
        if m[w] == 0:
            m.pop(w)
        k += word_len
    return True


def findSubString(s, words):
    str_len = len(s)
    word_len = len(words[0])
    list_len = len(words)
    window_len = word_len * list_len
    m = defaultdict(int)
    for word in words:
        m[word] += 1

    out = []
    for i in range(str_len - window_len + 1):
        if is_substr(s, i, i + window_len, word_len, m.copy()):
            out.append(i)
    return out


# print findSubString("barfoothefoobarman", ["foo", "bar"])

#
def threeSum(nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    n = len(nums)
    if n < 3:
        return []

    res = []

    l = sorted(nums)
    print "l ....", l
    prev = None

    for i in xrange(n):
        x = l[i]
        if x == prev:
            continue
        m = {}
        dup = set()
        print "value...", x, prev
        for j in xrange(i+1, n):
            y = l[j]
            if y in m and ((m[y], y) not in dup):
                res.append([x, m[y], y])
                dup.add((m[y], y))
                m.pop(y)
            else:
                m[-x - y] = y
        prev = x

    return res

# print "three sum....", threeSum([0,2,2,3,0,1,2,3,-1,-4,2])


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


# print letterCombination("246")


# Remove n-th node from linkedlist
def removeNthFromEnd(head, n):
    """
    :type head: ListNode
    :type n: int
    :rtype: ListNode
    """
    if n < 1:
        return head
    prev = curr = last = head

    for i in range(n-1):
        last = last.next

    while last.next is not None:
        last = last.next
        prev = curr
        curr = curr.next

    if curr == head:
        return head.next

    prev.next = curr.next

    return head


# Generate valid parenthesis for n pair
def insert_pt(string, nopen, nclose, n, out):
    if len(string) == 2*n:
        out.append(string)
        return

    if nclose > nopen:
        return

    if nopen < n:
        insert_pt(string + '(', nopen+1, nclose, n, out)

    if nclose < nopen:
        insert_pt(string + ')', nopen, nclose + 1, n, out)


def generateParenthesis(n):
    if n <= 0:
        return ''
    out = []
    insert_pt('', 0, 0, n, out)
    return out

# print "generate parenthesis...", generateParenthesis(4)


# Next greater permutation
# https://leetcode.com/problems/next-permutation/description/
def nextPermutation(nums):
    n = len(nums)
    if n <= 1:
        return

    j = n - 1
    # Find position where a(j-1) < a(j) to swap
    while j > 0:
        if nums[j] > nums[j-1]:
            break
        j -= 1

    # If array in non-increase order, sort it
    if j == 0:
        nums.sort()
        return

    idx = j
    x = nums[idx - 1]
    # Find value to swap with
    while j < n:
        if nums[j] <= x:
            break
        j += 1

    nums[idx - 1], nums[j - 1] = nums[j-1], nums[idx-1]

    # Reverse the rest of array from swap point
    j = n - 1
    while (idx <= j) and idx < n and j >= 0:
        nums[j], nums[idx] = nums[idx], nums[j]
        idx += 1
        j -= 1

# nums = [1, 2, 5, 8, 6, 3, 1]
# nextPermutation(nums)
# print "nex permutation....", nums


# Search value in rotated sorted array
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

# print "search rotated....", searchRotated([3, 1], 1)


# Search range of value in array
def searchRange(nums, target):
    if not nums:
        return [-1, -1]

    n = len(nums)
    l, r = 0, n-1
    idx = None
    # Search first index that equal to target and larger than left side
    while l <= r:
        mid = l + (r-l)/2

        if nums[mid] >= target:
            r = mid - 1
        else:
            l = mid + 1

    print "first index...", l, r, mid
    if l >= n or nums[l] != target:
        return [-1, -1]
    idx = l
    r = n - 1

    # Search second index that equal to target but less than right side
    while l <= r:
        mid = l + (r-l)/2
        if nums[mid] <= target:
            l = mid + 1
        else:
            r = mid - 1
    print "second index...", l, r, mid
    return [idx, r]

# print "Search range...", searchRange([5,7,7,8,8,10], 8)


# Combination sum of target value
def find_combine(nums, curr, idx, val, out):
    if val == 0:
        out.append(curr)
        return

    elif val < 0:
        return

    while idx < len(nums):
        x = nums[idx]
        if x <= val:
            find_combine(nums, curr + [x], idx, val - x, out)
        idx += 1


def combinationSum(candidates, target):
    out = []
    find_combine(candidates, [], 0, target, out)
    return out

# print "combination sum....", combinationSum([2, 3, 5], 8)


# Permutate list of integers [x1, x2, x3..., xn]
def permute(nums):
    out = [[]]
    for n in nums:
        perms = []
        for perm in out:
            for i in xrange(len(perm)+1):
                perms.append(perm[:i] + [n] + perm[i:])
        out = perms
    return out

# print "permutate....", permute([1, 2, 3])


# Rotate matrix clockwise
# Split it into multi-layer
# layer = n/2
# layer 0: 0,0 -> 0,3 ; 0,3 -> 3,3 ; 3,3 -> 3,0 ; 3,0 -> 0,0
# layer 1: 1,1 -> 1,2 ; 1,2 -> 2,2 ; 2,2 -> 2,1 ; 2,1 -> 1,1
def rotate(matrix):
    n = len(matrix)
    layer = n/2

    for i in xrange(layer):
        for j in xrange(i, n-i-1):
            tmp = matrix[i][j]
            matrix[i][j] = matrix[n-1-j][i]
            matrix[n-1-j][i] = matrix[n-1-i][n-1-j]
            matrix[n-1-i][n-1-j] = matrix[j][n-1-i]
            matrix[j][n-1-i] = tmp
    return matrix


matrix = [
  [5, 1, 9, 11],
  [2, 4, 8, 10],
  [13, 3, 6, 7],
  [15, 14, 12, 16]
]

# print "rotate image....\n", rotate(matrix)


# Find maximum subarray
# keep track of non-negative sub-sequence
def maxSubArraySum(nums):
    max_so_far = MIN
    max_ending_here = 0

    for i in range(len(nums)):
        max_ending_here = max_ending_here + nums[i]
        if (max_so_far < max_ending_here):
            max_so_far = max_ending_here

        if max_ending_here < 0:
            max_ending_here = 0
    return max_so_far

# print maxSubArraySum([0, -1, -1, 2, 2])


# Merge intervals
# https://leetcode.com/problems/merge-intervals/
# Add item into list, compare last list end to new item start and update
# Alternative: use interval tree to achieve without sorting
class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e


def merge(intervals):
    res = []
    for item in sorted(intervals, key=lambda x: x.start):
        if res and res[-1].end >= item.start:
            res[-1].end = max(res[-1].end, item.end)
        else:
            res.append(item)
    return res


# Edit distance
def minDistance(word1, word2):
    m = len(word1)
    n = len(word2)

    table = [[0 for x in range(n+1)] for y in range(m+1)]
    for i in range(n+1):
        table[0][i] = i

    for j in range(m+1):
        table[j][0] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            if word1[i-1] == word2[j-1]:
                table[i][j] = table[i-1][j-1]
            else:
                # 3 operations: insert, delete, replace
                table[i][j] = (min(table[i][j-1], table[i-1][j],
                                   table[i-1][j-1]) + 1)
    return table[m][n]


# print "edit distance....", minDistance('horse', 'ros')


# Sort colors
# https://leetcode.com/problems/sort-colors/
# Use count array to sort colors
def sortColors(nums):
    n = len(nums)
    count = [0, 0, 0]
    for num in nums:
        count[num] += 1

    i = val = 0
    while i < n and val < 3:
        if count[val] > 0:
            nums[i] = val
            count[val] -= 1
            i += 1
        else:
            val += 1

    return nums

# print "sort color...", sortColors([2,0,2,1,1,0])


# Find if word exists in 2D array in horizontal or vertical
# Use backtracking technique, for every char if it's matched with first char
# then search from there
# mark cell to avoid search again -> move and unmake move
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


def exist(board, word):
    """
    :type board: List[List[str]]
    :type word: str
    :rtype: bool
    """
    for i in range(len(board)):
        for j in range(len(board[0])):
            if _exist(board, word, 0, i, j):
                return True
    return False


# Find how many unique binary tree for n nodes
# https://leetcode.com/problems/unique-binary-search-trees/description/
# if tree has no node = 0, 1 node = 1
# make root at index 0, 1, ..., n-1
# we have number of unique trees: F(0).F(n-1), F(1).F(n-1), ..., F(n-1).F(0)
def numTrees(n):
    if n == 0:
        return 0
    if n == 1:
        return 1

    arr = [0 for x in range(n+1)]
    arr[0], arr[1] = 1, 1
    for x in range(2, n+1):
        for y in range(x):
            arr[x] += arr[y]*arr[x-1-y]
    return arr[n]

# print "unique tree...", numTrees(3)


# Generate unique binary tree
# https://leetcode.com/problems/unique-binary-search-trees-ii/description/
# Each time choose node to be root, generate left sub tree and right sub tree
# Overlap sub-problem: ex n = 6 -> 3 is root, left [1, 2] right [4, 6], n = 5
# root = 3 left [1, 2] right [4, 5]
# Use hashtable to store all trees for calculated range (i, j)
class UniqueBSTSolution(object):
    def generate(self, i, j, res):
        if i > j:
            return [None]

        if (i, j) in res:
            return res[(i, j)]

        if i == j:
            node = TreeNode(i)
            res[(i, j)].append(node)
            return [node]

        trees = []
        for k in range(i, j + 1):
            left = self.generate(i, k-1, res)
            right = self.generate(k+1, j, res)
            for l in left:
                for r in right:
                    n = TreeNode(k)
                    n.left = l
                    n.right = r
                    trees.append(n)
        res[(i, j)] = trees
        return trees

    def generateTrees(self, n):
        """
        :type n: int
        :rtype: List[TreeNode]
        """
        res = defaultdict(list)
        self.generate(1, n, res)
        return res[(1, n)]


# Check if tree is symmetric
def _symmetric(left, right):
    if not left and not right:
        return True

    if not left or not right:
        return False

    if left.val != right.val:
        return False

    return (_symmetric(left.left, right.right) and
            _symmetric(left.right, right.left))


def isSymmetric(root):
    if not root:
        return True
    return _symmetric(root.left, root.right)


def iteractiveIsSymmetric(root):
    if not root:
        return True

    lside, rside = deque([root.left]), deque([root.right])

    while lside and rside:
        p, q = lside.popleft(), rside.popleft()
        if not p and not q:
            continue

        if not p or not q:
            return False

        if p.val != q.val:
            return False

        lside += [p.left, p.right]
        rside += [q.right, q.left]
    return True


# Tree level ordering
def levelOrder(root):
    if not root:
        return []
    nodes = [root]
    res = []
    while nodes:
        tmp = []
        vals = []
        for node in nodes:
            vals.append(node.val)
            if node.left:
                tmp.append(node.left)
            if node.right:
                tmp.append(node.right)
        nodes = tmp
        if vals:
            res.append(vals)
    return res


# Tree flatten
def _flatten(node):
    if not node:
        return (None, None)
    n = node
    left = node.left
    right = node.right

    node.left = None
    lside, lastl = _flatten(left)
    if lside:
        node.right = lside
        n = lastl
    rside, lastr = _flatten(right)
    if rside:
        n.right = rside
        n = lastr
    return node, n


def flatten(self, root):
    """
    :type root: TreeNode
    :rtype: void Do not return anything, modify root in-place instead.
    """
    if not root:
        return
    _flatten(root)


# Tree maximum path sum
class TreeMaxPath():
    def _maxPath(self, node):
        if not node:
            return 0

        left = self._maxPath(node.left)
        right = self._maxPath(node.right)

        # Max sub tree
        _max = max(left, right)
        _max = max(node.val, _max + node.val)

        # Max path
        self.res = max(self.res, node.val, node.val + left, node.val + right,
                       left + node.val + right)
        return _max

    def maxPathSum(self, root):
        self.res = MIN
        self._maxPath(root)
        return self.res


# Continuous sub sequence
def longestConsecutive(nums):
    if not nums:
        return 0

    d = {}

    res = 0
    for num in nums:

        if num in d:
            continue

        # Find if there 2 next number
        upper = d.get(num+1, 0)
        lower = d.get(num-1, 0)

        # Update the length
        l = upper + lower + 1
        res = max(res, l)
        d[num] = l

        # Update length for both upper and lower bound
        d[num + upper] = l
        d[num + lower] = l
    return res

# print "longest consecutive...", longestConsecutive([100, 4, 200, 1, 3, 2, 5])


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

# print "max product...", maxProduct([2, -3, -2, 4])


# Majority number in array
# Moore majority voting algorithm
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


# Rob house
def rob(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    n = len(nums)
    if not n:
        return 0
    if n == 1:
        return nums[0]

    prev, cur = nums[0], max(nums[0], nums[1])
    for i in range(2, n):
        prev, cur = cur, max(nums[i] + prev, cur)

    return cur

# print "rob house....", rob([1, 2, 3, 1])


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


# Find if user can finish courses with prerequisites
# https://leetcode.com/problems/course-schedule/
# Make graph, and use dfs to check if can finish course
class CourseSolution(object):
    def finish(self, x, edges):
        self.status[x] = 1

        for y in edges[x]:
            if self.status[y] == 1:
                return False
            if self.status[y] == 0:
                if not self.finish(y, edges):
                    return False
        self.status[x] = 2
        return True

    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        edges = defaultdict(list)
        for (x, y) in prerequisites:
            edges[x].append(y)
            if y not in edges:
                edges[y] = []
        self.status = [0 for i in range(numCourses)]
        for i in range(numCourses):
            # not visit
            if self.status[i] == 0:
                if not self.finish(i, edges):
                    return False
        return True


# Longest mountain
def longestMountain(A):
    """
    :type A: List[int]
    :rtype: int
    """
    _max = 0
    n = len(A)
    if n < 3:
        return _max

    out = [1 for i in range(n)]

    for i in range(1, n):
        if A[i] > A[i-1]:
            out[i] = out[i-1] + 1

    for j in range(n-2, -1, -1):
        if A[j] > A[j+1]:
            if out[j] > 1:
                _max = max(out[j] + out[j+1], _max)

            out[j] = out[j+1] + 1
    return _max


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

# print "straight hand...", isNStraightHand([1,2,3,6,2,3,4,7,8], 3)


# Calculator
# https://leetcode.com/problems/basic-calculator-ii/description/
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


# First missing positive
# https://leetcode.com/problems/first-missing-positive/description/
# use hashmap => space O(n)
# Optimize: put number to its correct position in place
def firstMissingPositive(nums):
    if not nums:
        return 1

    n = len(nums)
    m = {num: 1 for num in nums if num > 0}
    for i in range(1, n+1):
        if i not in m:
            return i
    return n + 1


def firstMissingPositive2(nums):
    if not nums:
        return 1
    n = len(nums)
    i = 0
    while i < n:
        val = nums[i]
        if val > 0 and val <= n and nums[val - 1] != val:
            nums[i], nums[val - 1] = nums[val - 1], nums[i]
        else:
            i += 1
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1

    return n + 1

# print "first missing positive...", firstMissingPositive2([3, 4, -1, 1])


# word ladder
# https://leetcode.com/problems/word-ladder/discuss/40729/Compact-Python-solution
class WordSolution(object):
    def addNextWords(self, word, h, tmp):
        for i in range(len(word)):
            for ch in string.ascii_letters:
                nw = word[:i] + ch + word[i+1:]
                if nw in h:
                    tmp.append(nw)
                    h.pop(nw)

    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        if not wordList:
            return 0

        q = deque([beginWord])
        count = 0
        h = {w: 1 for w in wordList}
        h.pop(beginWord, None)

        while q:
            count += 1
            tmp = []
            while q:
                word = q.popleft()
                if word == endWord:
                    return count
                self.addNextWords(word, h, tmp)
            q = deque(tmp)
        return 0

# print "word ladder...", WordSolution().ladderLength('a', 'b', ['a', 'b', 'c'])


# Surrounded region
# Look for O in edge, and do BFS to flip to H
# Turn remain O to X, H to O
# https://leetcode.com/problems/surrounded-regions/description/
class SurroundSolution(object):
    def bfs(self, board, x, y):
        board[x][y] = 'H'
        q = deque([(x, y)])
        dirs = [(0, -1), (0, 1), (1, 0), (-1, 0)]

        while q:
            i, j = q.popleft()
            for d in dirs:
                nx, ny = i + d[0], j + d[1]
                if (0 < nx < len(board) and 0 < ny < len(board[0]) and
                    board[nx][ny] == 'O'):
                    board[nx][ny] = 'H'
                    q.append((nx, ny))

    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        if not board:
            return
        m = len(board)
        n = len(board[0])

        for i in range(m):
            if board[i][0] == 'O':
                self.bfs(board, i, 0)
            if board[i][n-1] == 'O':
                self.bfs(board, i, n-1)

        for j in range(n):
            if board[0][j] == 'O':
                self.bfs(board, 0, j)

            if board[m-1][j] == 'O':
                self.bfs(board, m-1, j)

        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                elif board[i][j] == 'H':
                    board[i][j] = 'O'


# Partition to k equal sum
# https://leetcode.com/problems/partition-to-k-equal-sum-subsets/description/
# arr = [1, 2, 3, 4, 5], k = 3 => (1, 4), (2, 3), (5)
def _partition(nums, curr, val, k, visited):
    if k == 1:
        return True

    if curr == val:
        return _partition(nums, 0, val, k - 1, visited)

    for i in range(len(nums)):
        if visited[i] or nums[i] + curr > val:
            continue
        visited[i] = True

        if _partition(nums, curr + nums[i], val, k, visited):
            return True

        visited[i] = False
    return False


def canPartitionKSubsets(nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: bool
    """
    if not nums or not k:
        return True
    n = len(nums)
    if k > n:
        return False
    s = sum(nums)
    if s % k:
        return False
    val = s / k
    visited = [False] * n
    return _partition(nums, 0, val, k, visited)

# print canPartitionKSubsets([730,580,401,659,5524,405,1601,3,383,4391,4485,1024,1175,1100,2299,3908], 4)


# Divide integer
def divide(dividend, divisor):
    """
    :type dividend: int
    :type divisor: int
    :rtype: int
    """
    sign = (dividend ^ divisor) >= 0
    res = 0
    if dividend < 0:
        dividend = -dividend
    if divisor < 0:
        divisor = -divisor
    while dividend >= divisor:
        tmp = divisor
        m = 1
        while dividend >= (tmp << 1):
            tmp = tmp << 1
            m = m << 1

        dividend -= tmp
        res += m
    return res if sign else -res

print "divide integer....", divide(7, -3)


# Biggest number
# https://leetcode.com/problems/largest-number/description/
# Python 2: nums.sort(cmp=lambda x, y: ...)
class Comparable(object):
    def __init__(self, num):
        self.num = str(num)

    def __cmp__(self, other):
        return cmp(self.num + other.num, other.num + self.num)

    def __str__(self):
        return self.num


# Edge case nums contains all zeroes
class NSolution(object):

    def largestNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        if not nums:
            return ''

        if not any(nums):
            return '0'
        m = sorted([Comparable(x) for x in nums], reverse=True)
        return ''.join([str(y) for y in m])


# Wiggle sort 2
# https://leetcode.com/problems/wiggle-sort-ii/description/
def wiggleSort(nums):
    """
    :type nums: List[int]
    :rtype: void Do not return anything, modify nums in-place instead.
    """
    if not nums or len(nums) <= 1:
        return
    nums.sort()
    n = len(nums)

    m = n/2
    for i in range(m+1):
        if i % 2 and m+i < n:
            nums[i], nums[m+i] = nums[m+i], nums[i+i]

# print "wiggle sort...", wiggleSort([1, 5, 1, 1, 6, 4])


# Print spiral matrix
# https://leetcode.com/problems/spiral-matrix/description/
def printLayer(matrix, layer, res):
    startx, endx = layer, len(matrix) - layer - 1
    starty, endy = layer, len(matrix[0]) - layer - 1

    i, j = startx, starty
    if startx == endx and starty == endy:
        res.append(matrix[startx][starty])
        return

    while j <= endy:
        res.append(matrix[i][j])
        j += 1

    i, j = startx + 1, endy
    while i <= endx:
        res.append(matrix[i][j])
        i += 1

    i, j = endx, endy - 1
    if startx != endx:
        while j >= starty:
            res.append(matrix[i][j])
            j -= 1

    i, j = endx - 1, starty
    if starty != endy:
        while i > startx:
            res.append(matrix[i][j])
            i -= 1


def spiralOrder(matrix):
    """
    :type matrix: List[List[int]]
    :rtype: List[int]
    """
    if not matrix:
        return []

    res = []
    l = min(len(matrix), len(matrix[0]))
    layers = l/2 + 1 if l % 2 else l/2
    for layer in range(layers):
        printLayer(matrix, layer, res)
    return res

print "spiral matrix....", spiralOrder([[6, 7, 9]])


# calculation
# https://leetcode.com/problems/evaluate-reverse-polish-notation/description/
def doCal(v1, v2, op):
    if op == '+':
        return v1 + v2
    if op == '-':
        return v1 - v2
    if op == '/':
        return v1/v2
    if op == '*':
        return v1*v2


def evalRPN(tokens):
    """
    :type tokens: List[str]
    :rtype: int
    """
    if not tokens:
        return 0
    ops = []
    for token in tokens:
        if token in ('+', '-', '*', '/'):
            v2 = ops.pop()
            v1 = ops.pop()
            ops.append(doCal(v1, v2, token))
        else:
            ops.append(int(token))
        print "ops...", ops
    return ops[0]

print "calculate...", evalRPN(["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"])


# Dynamic, Medium, Perfect square
# https://leetcode.com/problems/perfect-squares/description/
# Use array to store minimum square
def numSquares(n):
    """
    :type n: int
    :rtype: int
    """
    if n <= 0:
        return -1
    dp = range(n+1)

    for i in range(2, n+1):
        val = int(math.sqrt(i))
        for j in range(1, val+1):
            dp[i] = min(dp[i], dp[i-j*j] + 1)
    return dp[n]


# Find minimum sum path in triangle
# https://leetcode.com/problems/triangle/description/
# Process level by level, for each element it can add previous row with same
# column j or j - 1, except at index 0 and n-1
def triangle(numlist):
    if not numlist:
        return -1

    curr = numlist[0]
    n = len(numlist)
    for i in range(1, n):
        tmp = numlist[i]
        for j in range(len(tmp)):
            if j == 0:
                tmp[j] += curr[j]
            elif j == len(tmp) - 1:
                tmp[j] += curr[j-1]
            else:
                tmp[j] += min(curr[j], curr[j-1])
        curr = tmp
    return min(curr)

print "triangle....", triangle([[2], [3, 4], [6, 5, 7], [4, 1, 3, 8]])


# Maximal rectangle, give matrix with 0 and 1 return max rectangle contains
# only 1
# https://leetcode.com/problems/maximal-rectangle/description/
def maximalRectangle(matrix):
    m = len(matrix)
    n = len(matrix[0])
    res = 0
    h = [0 for _ in range(n)]
    for i in range(m):
        stack = []
        for j in range(n):
            if matrix[i][j] == '1':
                h[j] += 1
            else:
                h[j] = 0
            if not stack or h[j] >= h[stack[-1]]:
                stack.append(j)
            else:
                while stack and h[j] < h[stack[-1]]:
                    idx = stack.pop()
                    l = j - stack[-1] - 1 if stack else j
                    area = h[idx] * l
                    res = max(res, area)
                stack.append(j)
        while stack:
            idx = stack.pop()
            l = n - stack[-1] - 1 if stack else n
            area = h[idx] * l
            res = max(res, area)
    return res


# https://leetcode.com/problems/decode-string/description/
# example 3[a2[c]e]2[df] -> acceacceaccedfdf
# use stack, push char into stack, if [ append current num, if ] pop stack
# if pop item is number, then multiple char, if not, concatinate item to
# current char, append char to stack
def decodeString(s):
    """
    :type s: str
    :rtype: str
    """
    stack = []
    num = ''
    char = ''
    for c in s:
        if c.isdigit():
            num += c
            if char:
                stack.append(char)
                char = ''
        elif c == '[':
            stack.append(num)
            num = ''
        elif c == ']':
            while stack:
                n = stack.pop()
                if n.isdigit():
                    char = int(n) * char
                    break
                else:
                    char = n + char
            stack.append(char)
            char = ''
        else:
            char += c
    return ''.join(stack)


# https://leetcode.com/problems/pancake-sorting/
# Pan cake sort
class PanCakeSolution:
    def maxIndexUpTo(self, arr, n):
        idx = 0
        val = arr[0]
        for i in range(1, n):
            if arr[i] > val:
                val = arr[i]
                idx = i
        return idx

    def pancakeSort(self, A):
        """
        :type A: List[int]
        :rtype: List[int]
        """
        n = len(A)
        ans = []
        if not A:
            return ans

        for i in range(n):
            idx = self.maxIndexUpTo(A, n - i)
            # Already in order
            if idx == n - i - 1:
                continue
            # Flip to move max to front
            if idx != 0:
                A[:idx+1] = A[:idx+1][::-1]
                ans.append(idx + 1)
            # Flip to move to end
            A[:n-i] = A[:n-i][::-1]
            ans.append(n-i)
        return ans
