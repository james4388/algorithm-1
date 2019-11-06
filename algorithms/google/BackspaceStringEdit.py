

# Backspace string edit
# https://leetcode.com/problems/backspace-string-compare/
# Solution: Use stack to store char and pop
# Solution 2: use 2 pointer i, j for each string, find first non-backspace characters
# in 2 string and compare them
class BPSolution:
    def _edit(self, t):
        stack = []
        for char in t:
            if char != '#':
                stack.append(char)
            elif stack:
                stack.pop()
        return ''.join(stack)

    def backspaceCompare(self, S, T):
        return self._edit(S) == self._edit(T)
