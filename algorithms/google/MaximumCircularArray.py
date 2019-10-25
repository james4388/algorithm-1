

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
