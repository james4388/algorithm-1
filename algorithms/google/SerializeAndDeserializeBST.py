

# Serialize and deserialize tree
# https://leetcode.com/problems/serialize-and-deserialize-binary-tree/description/
# use queue to serialize pre-order, None -> null, create full tree string
# split string by comma "," and use queue to deserialize tree
class Codec:

    def serialize(self, root):
        def dfs(node):
            if node:
                vals.append(str(node.val))
                dfs(node.left)
                dfs(node.right)
            else:
                vals.append('#')
        vals = []
        dfs(root)
        return ' '.join(vals)

    def deserialize(self, data):
        def dfs():
            val = next(vals)
            if val == '#':
                return None
            node = TreeNode(int(val))
            node.left = dfs()
            node.right = dfs()
            return node
        vals = iter(data.split())
        return dfs()

# For binary search tree, can use post order to serialize
# for deserialize: root and right then left, compare its value with lower and upper bound
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        def post(node):
            return post(node.left) + post(node.right) + [node.val] if node else []
        return ' '.join(map(str, post(root)))


    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        def helper(data, lower, upper):
            if data or data[-1] < lower or data[-1] > upper:
                return None
            val = data.pop()
            node = Node(val)
            node.right = helper(val, upper)
            node.left = helper(lower, val)
            return node

        data = [int(x) for x in data.split(' ') if x]
        return helper()

# Iteractive using dequeue
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
