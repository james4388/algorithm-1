

# Longest valid parenthesis
# https://leetcode.com/problems/longest-valid-parentheses/
# - Solution 1: consider simple case )()(), if char ( length = 0, if char ) check previous
# char, if it is (, then there's match, dp[i] = 2 + dp[i-2]
# otherwise if there's valid sequence dp[i-1] > 0, check char before valid
# sequence i - dp[i-1] - 1, if match (
# then dp[i] = 2 + dp[i-1] + dp[i - dp[i-1] - 2]
# - Solution 2: use stack, push -1 to stack to remember last valid point,
# if ( push index into stack, if ) pop from stack, if stack not empty, length = idx - check previous
# valid point (if stack is empty, push index to stack)
# - Solution 3: count open and close from left to right, if equal max = 2 * num,
# if close > open => invalid point, reset 0; count from right to left if open > close reset
# 2 cases: ()()(()))))()) (close > open) and ((((((()()() (open > close)
class Solution(object):
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        n = len(s)
        dp = [0 for i in range(n)]
        res = 0

        for i in range(n):
            if s[i] == '(':
                dp[i] = 0
            if s[i] == ')' and i > 0:
                if s[i-1] == '(':
                    dp[i] = 2 + dp[i-2]
                elif dp[i-1] > 0:
                    prev = i - dp[i-1] - 1
                    if prev >= 0 and s[prev] == '(':
                        dp[i] = 2 + dp[i-1] + dp[prev - 1]

            res = max(res, dp[i])
        return res

    def longestValidParentheses2(self, s):
        stack = [-1]
        ans = 0
        for idx, char in enumerate(s):
            if char == '(':
                stack.append(idx)
            else:
                stack.pop()
                if stack:
                    ans = max(ans, idx - stack[-1])
                else:
                    stack.append(idx)
        return ans

    def longestValidParentheses3(self, s):
        left, right = 0, 0
        ans = 0
        n = len(s)
        for i in range(n):
            if s[i] == '(':
                left += 1
            else:
                right += 1

            if left == right:
                ans = max(ans, 2 * left)
            elif right > left:
                left, right = 0, 0

        for i in range(n - 1, -1, -1):
            if s[i] == '(':
                left += 1
            else:
                right += 1
            if left == right:
                ans = max(ans, 2 * right)
            elif left > right:
                left, right = 0, 0
        return ans


if __name__ == '__main__':
    sol = Solution()
    print(sol.longestValidParentheses3('((((((()()()'))
