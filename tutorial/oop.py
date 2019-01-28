from collections import deque


class Parent:
    # this variable is not visible to outside
    __count = 0

    def __init__(self):
        print "calling parent constructor"

    def __del__(self):
        print "calling destructor"

    def __repr__(self):
        return Parent.getAttr(self)

    def __str__(self):
        return "this is string presented for parent"

    def __cmp__(self, obj):
        print "calling compare"
        return True

    def parentMethod(self):
        print "calling parent method"

    def setAttr(self, attr):
        Parent.parentAttr = attr

    def getAttr(self):
        print "parent attribute:", Parent.parentAttr


class Child(Parent):
    def __init__(self):
        print "calling child constructor"

    def childMethod(self):
        print "calling child method"


# c = Child()
# c.childMethod()
# c.setAttr(200)
# c.getAttr()
# print str(c)

# Log file OOP
class LogEntry(object):

    def __init__(self, date, username, url):
        self.date = date
        self.username = username
        self.url = url


class UrlStat(object):

    def __init__(self, url, user):
        self.url = url
        self.userview = {}
        self.totalview = 0

    def update(self, user):
        if user in self.userview:
            self.userview[user] += 1
        else:
            self.userview = 1

        self.totalview += 1


class Log(object):
    '''
        Log file format: date - user - url
    '''

    def __init__(self, file):
        self.file = file
        self.entries = []
        self.urlstat = {}

    def parse_file(self):
        with open(self.file, 'r') as f:
            for line in f:
                date, user, url = line.split(" - ")
                self.entries.append(LogEntry(date, user, url))
                if url not in self.urlstat:
                    self.urlstat[url] = UrlStat(url)
                self.urlstat[url].update(user)

    def most_view(self, n=1):
        views = self.urlstat.items().sort(key=lambda x: x[1].totalview,
                                          reverse=True)
        return [x[0] for x in views[:n]]

    def most_user(self, n=1):
        users = self.urlstat.items().sort(key=lambda x: len(x[1].userview),
                                          reverse=True)
        return [x[0] for x in users[:n]]


class GroupLog(Log):

    def __init__(self, file):
        super(Log, self).__init__(file)

    @staticmethod
    def group_view(group, stat):
        count = 0
        for k, v in stat.userview.items():
            if k in group:
                count += v
        return count

    @staticmethod
    def group_user(group, stat):
        count = 0
        for k, v in stat.userview.items():
            if k in group:
                count += 1
        return count

    def most_view_group(self, group, n=1):
        views = self.urlstat.items().sort(key=lambda x: GroupLog.group_view(group, x[1]), reverse=True)
        return [x[0] for x in views[:n]]

    def most_user_group(self, group, n=1):
        users = self.urlstat.items().sort(key=lambda x: GroupLog.group_user(group, x[1]), reverse=True)
        return [x[0] for x in users[:n]]


# Design LRU cache (Least Recently Used)
class LRU(object):
    """
        LRU cache initialization
        :data hash table of key, value
        :capacity size of LRU
        :double linked list to keep track of recently used items
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.records = {}
        self.item_used = deque([])

    def __getitem__(self, key):
        if key not in self.records:
            raise KeyError("%s does not exist." % key)
            return None

        self.item_used.remove(key)
        self.item_used.appendleft(key)
        return self.records[key]

    def __setitem__(self, key, item):
        if key in self.records:
            self.item_used.remove(key)
            self.item_used.appendleft(key)
            self.records[key] = item
        else:
            if len(self.records) == self.capacity:
                k = self.item_used.pop()
                del self.records[k]
            self.records[key] = item
            self.item_used.appendleft(key)

    def __str__(self):
        return str(self.records.items())

# lru = LRU(3)
# lru["a"] = 1
# lru["b"] = 2
# lru["c"] = 3
# lru["a"] = 4
# lru["d"] = 5
# lru["e"] = 6
# print lru
# print lru["f"]
