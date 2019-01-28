#!/bin/python
import sys

MIN = -sys.maxint
MAX = sys.maxint


class TreeNode(object):
    def __init__(self, val, parent=None, left=None, right=None):
        self.val = val
        self.parent = parent
        self.left = left
        self.right = right

    def __str__(self):
        return str(self.val)


def _build_tree(arr, l, r):
    if l > r:
        return None

    mid = l + (r - l)/2
    n = TreeNode(arr[mid])
    n.left = _build_tree(arr, l, mid - 1)
    n.right = _build_tree(arr, mid + 1, r)
    return n


def build_tree(arr):
    return _build_tree(arr, 0, len(arr)-1)


def process_node(val):
    print(val)


def inorder_tree(node):
    if node:
        inorder_tree(node.left)
        process_node(node.val)
        inorder_tree(node.right)


# Iteractive in order tree traversal
# Use stack, push node and set node to left
# if null, pop from stack, set node to right
def inOrderTranversal(root):
    stack = []
    node = root
    while True:
        if node:
            stack.append(node)
            node = node.left
        else:
            if not stack:
                break
            node = stack.pop()
            print node.val
            node = node.right

tree = build_tree(range(0, 30, 2))


# Without stack, use right link to previous parent node
def morrisInorderTranversal(root):
    node = root
    while node:
        if node.left is None:
            print node.val
            node = node.right
        else:
            pre = node.left
            # Go to deepest right node of current node
            while pre.right and pre.right != node:
                pre = pre.right

            # Link right most child to current node
            # and move to left
            if pre.right is None:
                pre.right = node
                node = node.left
            else:
                # After process all left, node is now parent, unlink and move
                # to right part
                pre.right = None
                print node.val
                node = node.right


def postorder_tree(node):
    if node:
        postorder_tree(node.left)
        postorder_tree(node.right)
        process_node(node.val)


# Use stack: push right node and node to stack
# set node to left
# if node null, pop node from stack, if right equals to current stack top
# remove right, and push back node, otherwise print node and set to null
# 4 5 2 6 7 3 1 => stack
def postOrderTranversal(root):
    stack = []
    node = root
    while True:
        if node:
            if node.right:
                stack.append(node.right)
            stack.append(node)
            node = node.left
        else:
            if not stack:
                break
            node = stack.pop()
            if stack and stack[-1] == node.right:
                right = stack.pop()
                stack.append(node)
                node = right
            else:
                print node.val
                node = None

# print "post order tranversal..."
# postOrderTranversal(tree)


def preorder_tree(node):
    if node:
        process_node(node.val)
        preorder_tree(node.left)
        preorder_tree(node.right)


# Use stack, print root value, push right to stack
# set node to left, repeat
# if null, pop node from stack
def preOrderTranversal(root):
    stack = []
    node = root
    while True:
        if node:
            print node.val
            if node.right:
                stack.append(node.right)
            node = node.left
        else:
            if not stack:
                break
            node = stack.pop()

preorder_tree(tree)
print "preorder tree...\n"
preOrderTranversal(tree)


# Predecessor and successor in BST
# if left subtree is not null find right most, if right subtree is not null
# find the left most, if root > key, set successor as root.val, move to left
# if root < key, set predecessor to root.val, move to right
class TreePreSuccess:
    predecessor, successor = None, None

    def _findPreSuc(self, root, target):
        if not root:
            return

        if root.val == target:
            if root.left:
                node = root.left
                while node.left:
                    node = node.left
                self.predecessor = node.val
            if root.right:
                node = root.right
                while node.right:
                    node = node.right
                self.successor = node.val
        elif root.val > target:
            self.successor = root.val
            self._findPreSuc(root.left, target)
        else:
            self.predecessor = root.val
            self._findPreSuc(root.right, target)

    def findPredecessorSuccessor(self, root, target):
        self._findPreSuc(root, target)
        return self.predecessor, self.successor


def insert_tree(node, val):
    if node is None:
        p = TreeNode(val)
        node = p
        return node

    if val < node.val:
        if node.left is None:
            node.left = TreeNode(val)
        else:
            insert_tree(node.left, val)
    else:
        if node.right is None:
            node.right = TreeNode(val)
        else:
            insert_tree(node.right, val)


# Find k smallest number in
def _ksmall(node, k, stack):
    while node:
        stack.append(node)
        node = node.left

    while len(stack) > 0 and k > 0:
        n = stack.pop()
        k -= 1
        if k == 0:
            return n.val

        if n.right:
            return _ksmall(n.right, k, stack)
    return None


def ksmallest(tree, k):
    stack = []
    node = tree
    return _ksmall(node, k, stack)


class TreeSolution:
    def ksmallest2(self, node, k):
        self.res = None
        self.k = k
        self.ksmallest_helper(node)
        return self.res

    def ksmallest_helper(self, node):
        if not node:
            return
        self.ksmallest_helper(node.left)
        self.k -= 1
        if self.k == 0:
            self.res = node.val
            return
        self.ksmallest_helper(node.right)

    # Search value in tree that less than or equal value
    def search_lte(self, node, val):
        self.res = None
        self.search_helper(node, val)
        return self.res

    def search_helper(self, node, val):
        if not node:
            return

        if node.val > val:
            self.search_helper(node.left, val)
        else:
            self.res = node.val
            self.search_helper(node.right, val)

    # Check if sequence exists in tree
    def find_sequence(self, node, seq):
        if not seq:
            return True

        self.idx = 0
        self.find_seq_helper(node, seq)
        return self.idx == len(seq)

    def find_seq_helper(self, node, seq):
        if not node:
            return

        self.find_seq_helper(node.left, seq)
        if self.idx < len(seq) and node.val == seq[self.idx]:
            self.idx += 1
        self.find_seq_helper(node.right, seq)

    # Sum node = node.val + left + right
    def _sum_node(self, node):
        if not node:
            return
        self._sum_node(node.left)
        self._sum_node(node.right)
        self._sum = self._sum + node.val
        node.val = self._sum

    def sum_node(self, node):
        self._sum = 0
        self._sum_node(node)

    def _delete(self, node):
        if not node:
            return None

        if not node.left:
            return node.right

        if not node.right:
            return node.left

        succ = node.right
        pre = None
        while succ.left:
            pre = succ
            succ = succ.left
        # Delete successor node
        if pre:
            pre.left = succ.right
        else:
            node.right = succ.right
        node.val = succ.val
        return node

    def deleteKey(self, root, key):
        pre = None
        cur = root

        while cur and cur.val != key:
            pre = cur
            if cur.val < key:
                cur = cur.right
            else:
                cur = cur.left

        # Delete root
        if not pre:
            return self._delete(cur)

        if cur == pre.left:
            pre.left = self._delete(cur)
        else:
            pre.right = self._delete(cur)
        return root


# Check if tree is BST
def _binary_tree(node, _min, _max):
    if not node:
        return True

    if node.val < _min or node.val > _max:
        return False

    return (_binary_tree(node.left, _min, min(node.val, _max)) and
            _binary_tree(node.right, max(node.val, _min), _max))


def is_binary_tree(node):
    return _binary_tree(node, MIN, MAX)


# tree = build_tree(range(0, 20, 2))
# invalid_tree = build_tree([2, 6, 4, 5, 10])
# insert_tree(tree, 25)
# insert_tree(tree, 13)
# tranverse_tree(tree)
# print "4th smallest....", ksmallest(tree, 8)
# print "is binary tree...", is_binary_tree(invalid_tree)
# print "search less than equal...", TreeSolution().search_lte(tree, 14)
# print "check sequence....", TreeSolution().find_sequence(tree, [2, 4, 8, 10, 19])
# print "sum node...."
# TreeSolution().sum_node(tree)
# tranverse_tree(tree)
