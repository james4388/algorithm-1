

# https://leetcode.com/problems/evaluate-reverse-polish-notation/
class Solution(object):
    def doCal(self, v1, v2, op):
        if op == '+':
            return v1 + v2
        if op == '-':
            return v1 - v2
        if op == '/':
            return int(float(v1)/v2)
        if op == '*':
            return v1*v2

    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """
        if not tokens:
            return 0
        ops = []
        for token in tokens:
            if token in ('+', '-', '*', '/'):
                v2 = ops.pop()
                v1 = ops.pop()
                ops.append(self.doCal(v1, v2, token))
            else:
                ops.append(int(token))
        return ops[0]
