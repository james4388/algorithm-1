# Each node has start, end and total = sum of its children

class Node(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.total = 0
        self.left = None
        self.right = None


class SegmentTree(object):
    def __init__(self, nums):
        self.root = self._init_tree(nums, 0, len(nums) - 1)

    def _init_tree(self, nums, l, r):
        if l > r:
            return None

        if l == r:
            node = Node(l, r)
            node.total = nums[l]
            return node

        mid = (l + r) / 2

        node = Node(l, r)
        node.left = self._init_tree(nums, l, mid)
        node.right = self._init_tree(nums, mid+1, r)

        node.total = node.left.total + node.right.total
        return node

    def _update(self, node, index, value):
        if not node:
            return

        if node.start == node.end and node.start == index:
            node.total = value
            return value

        mid = (node.start + node.end) / 2
        if index <= mid:
            self._update(node.left, index, value)
        else:
            self._update(node.right, index, value)

        node.total = node.left.total + node.right.total
        return node.total

    def update(self, index, value):
        return self._update(self.root, index, value)

    def _sum(self, node, i, j):
        if node.start == i and node.end == j:
            return node.total

        mid = (node.start + node.end) / 2
        if j <= mid:
            return self._sum(node.left, i, j)
        elif i >= mid + 1:
            return self._sum(node.right, i, j)
        else:
            return (self._sum(node.left, i, mid) +
                    self._sum(node.right, mid + 1, j))

    def sumRange(self, i, j):
        return self._sum(self.root, i, j)


if __name__ == '__main__':
    arr = [1, 3, 5, 7, 9, 11, 13, 15]
    st = SegmentTree(arr)
    print "sum 2-5...", arr, st.sumRange(2, 5)
    arr[4] = 25
    st.update(4, 25)
    print "sum 2-5...", arr, st.sumRange(2, 5)
