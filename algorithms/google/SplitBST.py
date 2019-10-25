

# Split BST
# https://leetcode.com/problems/split-bst/
# Recursive call split, if node < v, call to right,
# connect right to less subtree
# if node > v, call to left, connect left to greater subtree
class SplitSolution:
    def splitBST(self, root, v):
        if not root:
            return None, None

        if root.val == v:
            right = root.right
            root.right = None
            return root, right

        elif root.val < v:
            lte, gt = self.splitBST(root.right, v)
            root.right = lte
            return root, gt
        else:
            lte, gt = self.splitBST(root.left, v)
            root.left = gt
            return lte, root
