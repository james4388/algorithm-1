

class Node():
    def __init__(self, val, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next

    def __str__(self):
        return str(self.val)


class DoublyLinkedlist():
    def __init__(self):
        self.head = None

    def push(self, val):
        n = Node(val)
        if not self.head:
            self.head = n
        else:
            n.next = self.head
            self.head.prev = n
            self.head = n

    def from_array(self, arr):
        for x in reversed(arr):
            self.push(x)

    def __str__(self):
        vals = []
        tmp = self.head
        while tmp is not None:
            vals.append(str(tmp.val))
            tmp = tmp.next
        return ' = '.join(vals)
