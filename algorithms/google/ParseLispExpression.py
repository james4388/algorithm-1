

# HARD: https://leetcode.com/problems/parse-lisp-expression/
# Evaluate Lisp expression: let x 1 y 2 z (add x y) m (let x 2 (mult x 5))
# Solution: parse list of tokens for each operations, recursively evaluate
# each token to add or mutiply, for "let" operation, evaluate every second
# expression and push it into var scope to evaluate last expression
class Solution:
    def evaluate(self, expression):
        return self.eval(expression, {})

    def eval(self, exp, prev_scope):
        if exp[0] != '(':
            # Expression is number, or defined variable
            if exp[0] == '-' or exp[0].isdigit():
                return int(exp)
            return prev_scope.get(exp)

        scope = {}
        scope.update(prev_scope)
        tokens = None

        if exp.startswith("(mult"):
            tokens = self.parse(exp[6:-1])
            return self.eval(tokens[0], scope) * self.eval(tokens[1], scope)

        elif exp.startswith("(add"):
            tokens = self.parse(exp[5:-1])
            return self.eval(tokens[0], scope) + self.eval(tokens[1], scope)
        else:
            # e.g let x 1 y 2 z (add x y), evaluate each second token, push
            # to scope for last expression
            for i in range(0, len(tokens) - 2, 2):
                scope[tokens[i]] = self.eval(tokens[i + 1], scope)

            return self.eval(tokens[-1], scope)

    # For example, parse "3 (add 2 3))" into "3" and "(add 2 3)"
    def parse(self, exp):
        res, builder = [], ""
        balance = 0
        for c in exp:
            if c == '(':
                balance += 1
            if c == ')' :
                balance -= 1
            if balance == 0 and c == ' ':
                res.append(builder)
                builder = ""
            else:
                builder += c

        if len(builder) > 0:
            res.append(builder)
        return res
