import sys


class Node(object):
    def __init__(self, val, idx, left=None, right=None):
        self.val = val
        self.idx = idx
        self.left = left
        self.right = right

    def __str__(self):
        return "{} at {}".format(self.val, self.idx)


# Tree with root is mimum or maximum of two children
class TournamentTree:
    def construct(self, arr):
        nodes = []

        for i, num in enumerate(arr):
            nodes.append(Node(num, i))

        while len(nodes) > 1:
            tmp = []

            for i in range(0, len(nodes), 2):
                if i == len(nodes) - 1:
                    root = Node(nodes[i].val, nodes[i].idx, nodes[i])
                else:
                    if nodes[i].val < nodes[i+1].val:
                        root = Node(nodes[i].val, nodes[i].idx, nodes[i], nodes[i+1])
                    else:
                        root = Node(nodes[i+1].val, nodes[i+1].idx, nodes[i], nodes[i+1])
                tmp.append(root)
            nodes = tmp
        return nodes[0]


class Solution(object):
    def _second(self, root):
        if not root or (not root.left and not root.right):
            return

        if (root.left and root.left.val < self.res and
            root.idx != root.left.idx):
            self.res = root.left.val
            self._second(root.right)
        elif (root.right and root.right.val < self.res and
              root.idx != root.right.idx):
            self.res = root.right.val
            self._second(root.left)

    def findSecondSmallest(self, root):
        self.res = sys.maxint
        self._second(root)
        return self.res

root = TournamentTree().construct([4, 3, 6, 2, 5, 1, 1])
print "second min...", Solution().findSecondSmallest(root)
