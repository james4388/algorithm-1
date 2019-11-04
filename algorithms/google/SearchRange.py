

# https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
# Find first and last element in array sorted
# Solution: use 2 binary search
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if not nums:
            return [-1, -1]
        ans = [-1, -1]
        n = len(nums)
        l, r = 0, n - 1

        while l < r:
            mid = l + (r - l)//2
            if nums[mid] < target:
                l = mid + 1
            else:
                r = mid
        if l >= n or nums[l] != target:
            return ans
        ans[0] = l
        l, r = 0, n - 1
        while l < r:
            mid = l + (r - l + 1)//2
            if nums[mid] > target:
                r = mid - 1
            else:
                l = mid
        ans[1] = r
        return ans
