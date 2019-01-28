import random
import bisect


# Find n smallest in tree
class Solution:
    def kthSmallest(self, root, k):
        self.k = k
        self.res = None
        self.helper(root)
        return self.res

    def helper(self, node):
        if not node:
            return
        self.helper(node.left)
        self.k -= 1
        if self.k == 0:
            self.res = node.val
            return
        self.helper(node.right)


def bst(arr, left, right, val):
    while left < right:
        q = (left + right)/2
        if val < arr[q]:
            right = q - 1
        else:
            left = q + 1
    return right


# find upper bound > k
def upperBound(nums, lo, hi, k):
    mid = None
    while lo <= hi:
        mid = (lo + hi)/2
        if nums[mid] > k and (mid == 0 or nums[mid-1] <= k):
            return mid
        elif nums[mid] > k:
            hi = mid - 1
        else:
            lo = mid + 1
    return mid

# print range(10)
# print "upper bound %d" % upperBound(range(10), 0, 9, 5)


# find lower bound >= k
def lowerBound(nums, lo, hi, k):
    mid = None
    while lo <= hi:
        mid = (lo + hi)/2
        if nums[mid] >= k and (mid == 0 or nums[mid-1] < k):
            return mid
        elif nums[mid] >= k:
            hi = mid - 1
        else:
            lo = mid + 1
    return mid

# print range(0, 20, 2)
# print "lower bound %d" % lowerBound(range(0, 20, 2), 0, 10, 5)


def longestSeqBST(arr):
    d = len(arr)
    m = [0 for x in xrange(d)]
    m[0] = arr[0]
    current = 1
    for i in xrange(1, d):
        pos = bst(arr, 0, current - 1, arr[i])
        if pos == current - 1:
            current += 1
        pos += 1
        m[pos] = arr[i]
    return current


def longestSeqDP(arr):
    d = len(arr)
    m = [1 for x in range(d)]
    result = 1
    for i in range(1, d):
        for j in range(0, i):
            if arr[i] > arr[j] and m[i] < m[j] + 1:
                m[i] = m[j] + 1
                result = max(result, m[i])
    return result
print longestSeqDP([1, 5, 3, 4, 2, 7, 9, 10, 2, 11])


# Find median of two arrays
def findMedian(arr1, arr2):
    p, q = len(arr1), len(arr2)
    arr1, arr2 = sorted((arr1, arr2), key=len)
    idx = (p + q - 1)/2
    l, r = 0, min(q, p)
    while l < r:
        i = (l + r)/2
        if idx - i - 1 < 0 or arr1[i] >= arr2[idx - i - 1]:
            r = i
        else:
            l = i + 1
    i = l
    # get the last 4 numbers
    nextfew = sorted(arr1[i:i+2] + arr2[idx-i:idx-i+2])
    # if total element is even, take average of first 2
    # otherwise just get first number
    return (nextfew[0] + nextfew[1 - (p+q) % 2]) / 2.0

# print findMedian([1, 3, 5, 7, 9], [6, 10, 14, 16, 19])


# Find minimal sub array sum >= s
def bstNum(nums, l, r, val):
    while l < r:
        m = l + (r - l)/2
        if nums[m] > val:
            r = m - 1
        else:
            l = m + 1
    return l


def subArraySum(nums, val):
    if sum(nums) < val:
        return 0

    for i in xrange(1, len(nums)):
        nums[i] += nums[i-1]

    res = len(nums) + 1
    for i in xrange(len(nums) - 1, -1, -1):
        if nums[i] < val:
            break
        v = nums[i] - val
        idx = bstNum(nums, 0, i-1, v)
        if nums[idx] == v and i - idx < res:
            res = i - idx
        elif i - idx + 1 < res:
            res = i - idx + 1
    return res if res <= len(nums) else 0

# print subArraySum([1, 4, 4], 4)


# space complexity O(1), readonly array contains from 1, n
# only 1 duplicate
def findDuplicate(nums):
    if not nums:
        return 0

    l, r = 1, len(nums) - 1
    while l < r:
        mid = l + (r - l)/2
        count = 0
        for x in nums:
            if x <= nums[mid]:
                count += 1
        if count > mid + 1:
            r = mid
        else:
            l = mid + 1

    return l

# print findDuplicate([1, 1, 2])


# Search matrix mxn
def searchMatrix(matrix, target):
    """
    :type matrix: List[List[int]]
    :type target: int
    :rtype: bool
    """
    row = len(matrix)
    col = len(matrix[0])
    l = row * col
    lo, hi = 0, l - 1
    while lo <= hi:
        m = lo + (hi-lo)/2
        i, j = m/col, m % col
        if matrix[i][j] == target:
            return True
        elif matrix[i][j] > target:
            hi = m - 1
        else:
            lo = m + 1
    return False


# sum modulo M
def sumModulo(nums, val):
    d = len(nums)
    result = 0
    m = [nums[0] % val]
    for i in xrange(1, d):
        nums[i] = (nums[i] + nums[i-1]) % val
        idx = bisect.bisect(m, nums[i])
        if idx == i:
            result = max(result, nums[i])
        else:
            result = max(result, (nums[i] - m[idx]) % val)
        bisect.insort(m, nums[i])

    return result

# print sumModulo([3, 1, 21, 9, 9, 5], 7)
