

# Fruit into basket
# https://leetcode.com/problems/fruit-into-baskets/
# Same idea as longest substring with most 2 distinct chars
# use 2 variables to store 2 type of fruits, when encounter 3rd types update
# length, move start pointer to minimum index, otherwise keep updating
# index of 2 types
class FruitSolution:
    def totalFruit(self, tree):
        """
        :type tree: List[int]
        :rtype: int
        """
        if not tree:
            return 0

        n = len(tree)
        if n <= 2:
            return n

        p = q = -1
        res = 0
        start = 0
        for i in range(n):
            if p == -1 or tree[i] == tree[p]:
                p = i
            elif q == -1 or tree[i] == tree[q]:
                q = i
            else:
                res = max(i - start, res)
                if p < q:
                    start = p + 1
                    p = i
                else:
                    start = q + 1
                    q = i
        return max(res, i - start + 1)
