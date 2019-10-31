

# https://leetcode.com/problems/find-peak-element/
# Find peak element where it's bigger than neighbors
# Solution: sequential scan: nums[i] > nums[i+1]
# Binary search: if mid > mid + 1 => in decrease slope, find in left, mid
# otherwise find in mid + 1, right
class Solution(object):
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        left, right = 0, len(nums) - 1

        while left < right:
            mid = (left + right) / 2
            if nums[mid] > nums[mid+1]:
                right = mid
            else:
                left = mid + 1
        return left
