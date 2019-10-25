

# Binary tree longest consecutive subsequence, it could be increasing or
# decreasing
# https://leetcode.com/problems/binary-tree-longest-consecutive-sequence-ii
# Recursive call to left and right child, return the length of inc and dec
# if node is null return (0, 0)
# check if node = left +/- 1, = right +/- 1 and update range
class BSTLongestSol:
    ans = 1

    def longestConsecutiveSeq(self, root):
        if not root:
            return (0, 0)

        dec, inc = 1, 1

        if root.left:
            lr = self.longestConsecutiveSeq(root.left)
            if root.left.val == root.val + 1:
                dec = lr[0] + 1

            if root.left.val == root.val - 1:
                inc = lr[1] + 1

        if root.right:
            rr = self.longestConsecutiveSeq(root.right)
            if root.right.val == root.val + 1:
                inc = max(inc, rr[1] + 1)

            if root.right.val == root.val - 1:
                dec = max(dec, rr[0] + 1)

        self.ans = max(self.ans, dec + inc - 1)
        return (dec, inc)
