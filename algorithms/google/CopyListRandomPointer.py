

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
    def copyRandomList(self, head):
        """
        :type head: RandomListNode
        :rtype: RandomListNode
        """
        if not head:
            return None
        
        tmp = head.next
        prev = nhead = RandomListNode(head.label)
        head.copy = nhead
        while tmp:
            prev.next = RandomListNode(tmp.label)
            prev = prev.next
            # Linked new node to original node
            tmp.copy = prev
            tmp = tmp.next
        
        tmp = nhead
        h = head
        
        # Copy random pointer
        while tmp:
            if h.random:
                tmp.random = h.random.copy
            tmp = tmp.next
            h = h.next
        return nhead
