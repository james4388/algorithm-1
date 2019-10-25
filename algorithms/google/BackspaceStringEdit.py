

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


# Repeated string match
# https://leetcode.com/problems/repeated-string-match/
# calculate time len(B)/len(A) + 1, check if B is substring A*q or A*(q+1)
class StringSolution:
    def repeatedStringMatch(self, A, B):
        if not A and not B:
            return 1
        q = (len(B) - 1) // len(A) + 1
        for i in range(2):
            if B in A*(q+i):
                return q+i
        return -1
