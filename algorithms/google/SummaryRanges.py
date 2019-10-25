

# https://leetcode.com/problems/summary-ranges/
# Given sorted arry, return summary of its ranges
# Use len var to track length of range
def summaryRanges(nums):
    ans = []
    if not nums:
        return ans

    l = 1

    def summary(i):
        return (str(nums[i-1]) if l == 1
                else "{}->{}".format(nums[i-l], nums[i-1]))

    for idx in range(1, len(nums)):
        if nums[idx] == nums[idx-1] + 1:
            l += 1
        else:
            ans.append(summary(idx))
            l = 1

    ans.append(summary(len(nums)))
    return ans
