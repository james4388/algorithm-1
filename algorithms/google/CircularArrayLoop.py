

# https://leetcode.com/problems/circular-array-loop/
# Give circular array, at each index jump = index + value
# Find if array has a loop
# Solution: at each index make a jump and mark it along the way
class Solution(object):
    def circularArrayLoop(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        for i, num in enumerate(nums):
            # use a distinct marker for each starting point
            mark = str(i)
            while (type(nums[i]) == int) and (num * nums[i] > 0) and (nums[i] % len(nums) != 0):
                jump = nums[i]
                nums[i] = mark
                i = (i + jump) % len(nums)
            if nums[i] == mark:
                return True

        return False
