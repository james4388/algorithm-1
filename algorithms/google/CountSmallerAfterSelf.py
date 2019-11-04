

# https://leetcode.com/submissions/detail/275835647/
# Give list of number, return count number of smaller after index
# Solution: brute force 2 loop n^2
# Use bst: insert node into tree, at every node track number of its left child,
# insert new node into tree, insert left => increase number of left node
# insert right, add number of left nodes and count of duplicate into sum
# return sum + number of its left nodes
class Node:
    def __init__(self, val):
        self.left = None
        self.right = None
        self.val = val
        self.count = 0
        self.num_left = 0

def insert(node, num):
    count = 0

    while node.val != num:
        if node.val > num:
            node.num_left += 1
            if not node.left:
                node.left = Node(num)
            node = node.left
        else:
            count += node.num_left + node.count
            if not node.right:
                node.right = Node(num)
            node = node.right
    node.count += 1
    return count + node.num_left

class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        if not nums:
            return []
        n = len(nums)
        ans = [0 for i in range(n)]
        root = Node(nums[-1])
        for i in range(n-1, -1, -1):
            ans[i] = insert(root, nums[i])
        return ans
