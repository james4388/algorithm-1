

# Serialize and deserialize tree
# https://leetcode.com/problems/serialize-and-deserialize-binary-tree/description/
# use queue to serialize pre-order, None -> null, create full tree string
# split string by comma "," and use queue to deserialize tree
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        out = []
        q = deque([root])
        while q:
            item = q.popleft()
            if not item:
                out.append('null')
                continue
            out.append(str(item.val))
            q.append(item.left)
            q.append(item.right)
        return ','.join(out)

    def makeNode(self, val):
        return Node(int(val)) if val != 'null' else None

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        arr = data.split(',')
        root = self.makeNode(arr[0])
        q = deque([root])
        i = 1
        while q and i < len(arr):
            item = q.popleft()
            if not item:
                continue
            item.left = self.makeNode(arr[i])
            item.right = self.makeNode(arr[i+1])
            q.append(item.left)
            q.append(item.right)
            i += 2
        return root
