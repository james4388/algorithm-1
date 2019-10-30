

# https://leetcode.com/problems/parsing-a-boolean-expression/
# Bool expression: !, &, | with true or false as t, f
# Solution: same as parse lisp expression
class Solution(object):
    def parseBoolExpr(self, expr):
        """
        :type expression: str
        :rtype: bool
        """

        def parseExpr(expr):
            countBracket = 0
            tokens = []
            buff = ""
            for char in expr:
                if char == '(':
                    countBracket += 1
                if char == ')':
                    countBracket -= 1
                if char == ',' and countBracket == 0:
                    tokens.append(buff)
                    buff = ""
                else:
                    buff += char
            if buff:
                tokens.append(buff)
            return tokens

        if not expr:
            return False

        if expr[0] in ('f', 't'):
            return expr[0] == 't'

        tokens = parseExpr(expr[2:-1])
        if expr[0] == '!':
            return not self.parseBoolExpr(tokens[0])
        if expr[0] == '&':
            for token in tokens:
                if not self.parseBoolExpr(token):
                    return False
            return True
        if expr[0] == '|':
            for token in tokens:
                if self.parseBoolExpr(token):
                    return True
            return False
        return False
