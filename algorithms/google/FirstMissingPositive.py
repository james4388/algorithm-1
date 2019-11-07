# First missing positive
# https://leetcode.com/problems/first-missing-positive/description/
# use hashmap => space O(n)
# Optimize: put number to its correct position in place
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
