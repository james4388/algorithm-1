from collections import Counter, deque, defaultdict, namedtuple

# Counter
c = Counter(a=4, b=2, c=3, d=-2)
print list(c.elements())

c = Counter('abradacafadd')
c.most_common(3)  # return most common elements > 3

# operations on counter
c.items()
c.values()
c.clear()

d = Counter(a=1, b=2, d=1)
print c - d
print c + d
print c & d  # intersection
print c | d  # union

# deque: double-ended queue
dq = deque('abcd')
# append to both sides
dq.append('k')
dq.appendleft('t')

# extends to both sides
dq.extend(['e', 'f'])
dq.extendleft(['1', '2'])

# pop from both sides
dq.pop()
dq.popleft()

dq.remove('a')
dq.count('a')
dq.rotate(3)  # rotate n steps to the right
dq.reverse()
dq.maxlen

# default dict
dd = defaultdict(list)
for x in range(0, 10, 2):
    dd['a'].append(x)

for y in range(1, 10, 2):
    dd['b'].append(y)

print "default dict ...", dd

# namedtuple
Emp = namedtuple('Employee', 'id, name, age')
e1 = Emp._make([12, 'Harry', 25])
e1._replace(id=199)
print "named tuple employee ...", e1

# OrderedDict
od = e1._asdict()
item = od.popitem()
print "poped item...", item
print "ordered dict....", od
