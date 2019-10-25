

# Strobogrammatic number
# https://leetcode.com/problems/strobogrammatic-number-ii/
# recursive for pair 00, 11, 69, 88, 96
class StrobogrammaticSolution(object):
    def find(self, left, right, n, res):
        count = len(left) + len(right)
        if count == n:
            res.append(left + right)
            return

        if count == n - 1:
            for num in ('0', '1', '8'):
                res.append(left + num + right)
            return

        for num in ('0', '1', '8'):
            if not left and num == '0':
                continue
            self.find(left + num, num + right, n, res)

        self.find(left + '6', '9' + right, n, res)
        self.find(left + '9', '6' + right, n, res)
        return

    def findStrobogrammatic(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        res = []
        self.find('', '', n, res)
        return res
