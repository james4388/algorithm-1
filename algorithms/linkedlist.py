#!/bin/python
import unittest


class ListNode(object):
    def __init__(self, data, next=None):
        self.val = data
        self.next = next

    def __str__(self):
        return str(self.val)


# Single LinkedList
class LinkedList(object):
    def __init__(self):
        self.head = None

    def __len__(self):
        temp = self.head
        count = 0
        while temp:
            count += 1
            temp = temp.next

        return count

    def __iter__(self):
        self.current = self.head
        return self

    @classmethod
    def from_array(cls, arr):
        l = cls()
        for x in reversed(arr):
            l.push(x)
        return l

    def next(self):
        if not self.current:
            raise StopIteration

        n = self.current
        self.current = self.current.next
        return n

    def push(self, data):
        n = ListNode(data)
        if not self.head:
            self.head = n
        else:
            n.next = self.head
            self.head = n


class LinkedListTest(unittest.TestCase):
    def setUp(self):
        self.linkedlist = LinkedList()
        self.linkedlist.push(10)
        self.linkedlist.push(20)

    def test_push(self):
        self.linkedlist.push(30)
        self.assertEqual(len(self.linkedlist), 3)
        self.assertEqual(self.linkedlist.head.data, 30)

    def test_iterator(self):
        vals = [x.data for x in self.linkedlist]
        self.assertEqual(vals, [20, 10])
        self.assertNotEqual(self.linkedlist.head, None)

    def tearDown(self):
        self.linkedlist = None


def reverse_linkedlist(head):
    prev = None
    curr = head

    while curr is not None:
        tmp = curr.next
        curr.next = prev

        prev = curr
        curr = tmp
    return prev


# Check if linkedlist has cycle
def hasCycle(head):
    """
    :type head: ListNode
    :rtype: bool
    """
    if not head or not head.next:
        return False

    slow = head
    fast = head.next

    while slow != fast:
        if not fast or not fast.next:
            return False
        slow = slow.next
        fast = fast.next.next
    return True


# Detect cycle node in linkedlist
def detectCycle(head):
    if not head or not head.next:
        return None

    slow = fast = entry = head

    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:
            while slow != entry:
                entry = entry.next
                slow = slow.next
            return entry
    return None


def print_list(head):
    s = head
    while s:
        print s.val
        s = s.next


# Sort linkedlist using merge sort
def merge(p, q):
    dummy = ListNode(0)
    cur = dummy
    while p and q:
        if p.val < q.val:
            cur.next = p
            p = p.next
        else:
            cur.next = q
            q = q.next
        cur = cur.next
    cur.next = p or q
    return dummy.next


def sort(head):
    if not head or not head.next:
        return head

    pre = slow = fast = head
    while fast and fast.next:
        pre = slow
        slow = slow.next
        fast = fast.next.next

    pre.next = None
    p = sort(head)
    q = sort(slow)
    return merge(p, q)


# Check if 2 linkedlist intersect
def getIntersectionNode(headA, headB):
    if not headA or not headB:
        return None

    a, b = headA, headB
    while a != b:
        a = a.next if a else headB
        b = b.next if b else headA
    return a


# second method find difference in length of both linkedlist
def getIntersectionNode2(headA, headB):
    if not headA or not headB:
        return None

    lenA, lenB = 0, 0
    a, b = headA, headB
    while a:
        lenA += 1
        a = a.next

    while b:
        lenB += 1
        b = b.next

    a, b = headA, headB
    while lenA < lenB:
        b = b.next

    while lenB < lenA:
        a = a.next

    while a != b:
        a = a.next
        b = b.next
    return a

# l = LinkedList.from_array([2, 1, 4, 3, 6, 0])
# head = l.head
#
# head = sort(head)
# print_list(head)


# Copy linkedlist with random pointer
# https://leetcode.com/problems/copy-list-with-random-pointer/description/
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
            tmp.copy = prev
            tmp = tmp.next

        tmp = nhead
        h = head
        while tmp:
            if h.random:
                tmp.random = h.random.copy
            tmp = tmp.next
            h = h.next
        return nhead
