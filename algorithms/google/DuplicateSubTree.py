

# Find duplicate subtree in tree
# Use serialize tree inorder, use hashmap to store subtree
class TreeSolution(object):
    def serialize(self, root):
        if not root:
            return ''

        left = self.serialize(root.left)
        right = self.serialize(root.right)
        subtree = left + str(root.val) + right
        if len(subtree) > 1 and subtree in self.table:
            self.result = True
        return subtree

    def is_duplicate(self, root):
        self.table = {}
        self.result = False
        self.serialize(root)
        return self.result
