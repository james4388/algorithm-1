

# https://leetcode.com/problems/copy-list-with-random-pointer/
# A linkedlist with random pointer to a node in list
# Copy linked list, use a copy to store new copy of node,
# Use second loop to set random pointer for new copied node
class RandomListNode(object):
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None


class Solution(object):
    # Runtime O(n), space O(n)
    def copyRandomList(self, head):
        if not head:
            return None

        cache = {}
        curr = head

        newHead = RandomListNode(0)
        prev = newHead

        while curr:
            node = RandomListNode(curr.label)
            cache[curr] = node

            prev.next = node
            prev = node
            curr = curr.next

        curr = head
        while curr:
            node = cache[curr]
            random = cache.get(curr.random)
            node.random = random
            curr = curr.next
        return newHead.next

    # 3 steps: - 1: link node.next to copy, and copy.next to next
    # - 2: link copy node to its random, node.next.random = node.random.next
    # - 3: restore previous list: node.next = node.next.next
    def copyRandomList2(self, head):
        curr = head

        # Step 1
        while curr:
            nextNode = curr.next
            curr.next = RandomListNode(curr.label)
            curr.next.next = nextNode
            curr = nextNode

        # Step 2
        curr = head
        while curr:
            curr.next.random = curr.random
            curr = curr.next.next

        # Step 3
        curr = head
        newHead = RandomListNode(0)
        prev = newHead

        while curr:
            copy = curr.next
            nextNode = copy.next

            curr.next = nextNode
            prev.next = copy

            prev = prev.next
            curr = curr.next
        return newHead.next
