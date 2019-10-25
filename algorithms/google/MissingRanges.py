

# - https://leetcode.com/problems/missing-ranges
# - Given a range lower and upper (inclusive), and a list of number in range,
# - return missing range
# Solution: use `next` var to track of next continuous value, increase next
# if nums[i] equal, if nums[i] higher, has a missing range from
# `next` to min(nums[i] - 1, upper), set `next` to nums[i] + 1
def findMissingRanges(nums, lower, upper):
    ans = []

    def summary(low, high):
        high = min(upper, high)
        return str(low) if low == high else "{}->{}".format(low, high)

    val = lower
    for idx in range(len(nums)):
        if nums[idx] == val:
            val += 1
        elif nums[idx] > val:
            ans.append(summary(val, nums[idx] - 1))
            val = nums[idx] + 1

    if val < upper:
        ans.append(summary(val, upper))
    return ans

print("missing range...", findMissingRanges([1, 3, 6, 9, 10, 15], 0, 13))
