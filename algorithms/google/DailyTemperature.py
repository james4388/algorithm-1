

# https://leetcode.com/problems/daily-temperatures/
# Give a list of temperature for each day, return number of day to wait for warmer
# Solution: use stack to store previous temperature, compare and pop from stack if larger
class Solution(object):
    def dailyTemperatures(self, T):
        """
        :type T: List[int]
        :rtype: List[int]
        """
        ans = [0 for i in range(len(T))]
        stack = []

        for idx, temp in enumerate(T):
            while stack and temp > T[stack[-1]]:
                prev = stack.pop()
                ans[prev] = idx - prev
            stack.append(idx)
        return ans
