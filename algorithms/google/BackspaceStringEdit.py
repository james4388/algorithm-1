

# Backspace string edit
# https://leetcode.com/problems/backspace-string-compare/
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
