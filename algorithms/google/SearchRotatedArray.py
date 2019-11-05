
# https://leetcode.com/problems/search-in-rotated-sorted-array-ii/
# Handle duplicate case: {2, 2, 2, 2, 2, 2, 2, 2, 0, 2}
# if mid == right, cannot decide which part to recur, decrease right index
def searchRotate(nums, target):
    if not nums:
        return -1

    l, r = 0, len(nums) - 1

    while l < r:
        mid = l + (r - l)//2

        if nums[mid] == target:
            return mid

        if nums[mid] > nums[r]:
            if nums[l] <= target < nums[mid]:
                r = mid - 1
            else:
                l = mid + 1
        elif nums[mid] < nums[r]:
            if nums[mid] < target <= nums[r]:
                l = mid + 1
            else:
                r = mid - 1
        else:
            r -= 1
    return l if nums[l] == target else -1
