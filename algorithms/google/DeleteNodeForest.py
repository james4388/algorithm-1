

# https://leetcode.com/problems/delete-nodes-and-return-forest/submissions/
# Give a list of values to delete from tree, return forests after deleting
# Solution: recursively delete in left and right, return node if not delete, else
# return None, handle case where root is deleted
class Solution:
    def delNodes(self, root: TreeNode, to_delete: List[int]) -> List[TreeNode]:
        res = []
        def bs(node, curr):
            if not node:
                return None
            node.left = bs(node.left, curr)
            node.right = bs(node.right, curr)

            if node.val in curr:
                if node.left:
                    res.append(node.left)
                if node.right:
                    res.append(node.right)
                return None
            return node

        todel = set(to_delete)
        if root.val not in todel:
            res.append(root)
        bs(root, todel)
        return res
