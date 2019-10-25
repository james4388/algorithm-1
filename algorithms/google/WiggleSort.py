

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


# Wiggle sort 2
# https://leetcode.com/problems/wiggle-sort-ii/description/
def wiggleSort2(nums):
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
