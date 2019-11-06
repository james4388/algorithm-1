

# https://leetcode.com/problems/brace-expansion/submissions/
# Give expression: {a, b}c => [ac, bc]
# Solution: if encounter {, find next } and split by , for each expression
#
class Solution(object):
    def expand(self, S):
        """
        :type S: str
        :rtype: List[str]
        """
        idx = 0
        n = len(S)
        brackets = {}
        curr = ['']
        for idx, char in enumerate(S):
            if char == '{':
                prev = idx
            if char == '}':
                brackets[prev] = idx
        idx = 0
        while idx < n:
            if S[idx] == '{':
                tokens = S[idx+1: brackets[idx]].split(",")
                curr = [x + y for x in curr for y in tokens]
                idx = brackets[idx]
            else:
                char = S[idx]
                curr = [x + char for x in curr]
            idx += 1
        return sorted(curr)
