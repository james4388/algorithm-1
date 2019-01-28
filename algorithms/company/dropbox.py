import Queue
import threading
import requests
import time

from threading import Lock, RLock, Semaphore, Condition


# Check if a cell is illuminated in array, there're lamps (or queen in chess)
# solution: store x, y of lamps, diagonal y - x and x + y in dictionary
class LampSolution(object):
    def storeLamps(self, lamps):
        table = {'x': set(), 'y': set(), 'd1': set(), 'd2': set()}
        for lamp in lamps:
            table['x'].add(lamp[0])
            table['y'].add(lamp[1])
            table['d1'].add(lamp[1] - lamp[0])
            table['d2'].add(lamp[0] + lamp[1])

        return table

    def isIlluminated(self, N, lamps, cell):
        table = self.storeLamps(lamps)
        return (cell[0] in table['x'] or cell[1] in table['y'] or
                cell[1] - cell[0] in table['d1'] or
                cell[0] + cell[1] in table['d2'])


# Easy: Closest binary search tree value
# https://leetcode.com/problems/closest-binary-search-tree-value/
class ClosestValueSolution:
    def _closest(self, node, target):
        if not node:
            return

        if abs(node.val - target) < abs(self.value - target):
            self.value = node.val
        if target < node.val:
            self._closest(node.left, target)
        else:
            self._closest(node.right, target)

    def closestValue(self, root, target):
        if not root:
            return None

        self.value = float('inf')
        self._closest(root, target)
        return self.value


# Design file system support ls, mkdir, pwd, rm, cat, mv, cd
class iNode:
    def __init__(self, name='', parent=None, file=False, content=''):
        self.name = name
        self.isFile = file
        self.content = content
        self.parent = parent
        self.children = {'..': self.parent, '.': self}

    @property
    def fullPath(self):
        if self.parent:
            return self.parent.fullPath + self.name + '/'
        return '/'

    def __repr__(self):
        return self.name


class FileSystem:
    def __init__(self):
        self.root = iNode()
        self.curr = self.root

    def ls(self):
        return (name for name in self.curr.children)

    def _transverse(self, path):
        node = self.root if path.startswith('/') else self.curr
        for name in path.split('/'):
            # Handle edge case where path ending with '/'
            if name == '':
                continue

            if name not in node.children:
                return None
            node = node.children[name]
        return node

    def cd(self, path):
        node = self._transverse(path)
        self.curr = node

    def mkdir(self, path):
        node = self.root if path.startswith('/') else self.curr
        for name in path.split('/'):
            if not name:
                continue

            if name not in node.children:
                n = iNode(name, parent=node)
                node.children[name] = n
                node = n

    def mv(self, currPath, newPath):
        currNode = self._transverse(currPath)
        currParent = currNode.parent
        currParent.children.pop(currNode.name, None)

        newName = ''
        j = len(newPath) - 2 if newPath.endswith('/') else len(newPath) - 1
        while newPath[j] != '/':
            newName = newPath[j] + newName
            j -= 1
        newParent = self._transverse(newPath[0:j+1])

        currNode.name = newName
        currNode.parent = newParent
        newParent.children[newName] = currNode

    def rm(self, path):
        node = self._transverse(path)
        parent = node.parent
        parent.children.pop(node.name)

    def pwd(self):
        return self.curr.fullPath


# Webcrawler multithreading
# Use another queue to parse html text using BeautifulSoup
class Worker(threading.Thread):
    def __init__(self, queue, output):
        threading.Thread.__init__(self)
        self.queue = queue
        self.output = output
        self.daemon = True

    def run(self):
        while True:
            url = self.queue.get()
            resp = requests.get(url)
            print("Crawling url...", url, resp.status_code)
            self.output.put(resp.text)
            self.queue.task_done()


def webcrawler(hosts):
    queue = Queue()
    out = Queue()

    for host in hosts:
        queue.put(host)

    for i in range(5):
        worker = Worker(queue, out)
        worker.daemon = True
        worker.start()

    # Wait for all tasks finished
    queue.join()


def downloadFile(url, outfile):
    r = requests.get(url, stream=True)
    with open(outfile, 'w') as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)


'''
- Token Bucket algorithm: use linklist to store consumed token (num_token, time),
delete from head for expired tokens, add to available when there's request to
consume
- Optimize use background thread scheduled at time/tokens to refill 1 token
- Optimize: as original algorithm but compare request time to last refill time,
calculate elapsed time and number of token to refill, add to available tokens
'''


class Bucket(object):
    def __init__(self, max_amount, refill_time, refill_amount):
        self.max_amount = max_amount
        self.refill_time = refill_time
        self.refill_amount = refill_amount
        self.reset()

    def _refill_count(self):
        return int(((time.time() - self.last_update) / self.refill_time))

    def reset(self):
        self.value = self.max_amount
        self.last_update = time.time()

    def get(self):
        return min(
            self.max_amount,
            self.value + self._refill_count() * self.refill_amount
        )

    def reduce(self, tokens):
        refill_count = self._refill_count()
        self.value += refill_count * self.refill_amount
        self.last_update += refill_count * self.refill_time

        if self.value >= self.max_amount:
            self.reset()
        if tokens > self.value:
            return False

        self.value -= tokens
        return True

'''
    lock.acquire()
    try:
        do_something()
    finally:
        lock.release()

    context management protocol
    with lock:
        do_something()

    use acquire(False) to not block, return True or False instead
- RLock only blocked if hold by other thread, acquire can be nested without
blocking
- Event: event = threading.Event()
    - client wait for event: event.wait()
    - sever set en event or clear it: event.set(), event.clear()
- Condition object: cv = threading.Condition()
    - producer produces item and notify consumer:
        with cv:
            ..produce item
            cv.notify()
    - consumer wait for item to consume
        with cv:
            while not item:
                cv.wait()
            get_item
'''
