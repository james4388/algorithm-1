

class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = s


class INode(object):
    def __init__(self, interval):
        self.i = interval
        self.max = interval.end
        self.left = self.right = None


def insert(node, interval):
    if not node:
        return INode(interval)

    item = node.i
    if item.start < interval.start:
        node.left = insert(node.left, interval)
    else:
        node.right = insert(node.right, interval)

    if item.max < interval.end:
        item.max = interval.end
    return node
