__author__ = 'rosary'

# list
l = range(10)
l.append(12)
l.insert(2, 5)
l.extend([23, 24])
print l

l.remove(12)
l.pop()
print l.index(5)
print l.count(5)

l.sort(reverse=False)
l.reverse()
print l

print "len %d min %d max %d" % (len(l), min(l), max(l),)

# string
st = "hello world, this this is is sample sample text text"
print st.count("is", 20, 50)

print st.find("this", 10, 50)

st.capitalize()
st.lower()

print ", ".join(["a", "b", "c"])

st.lstrip()
st.strip()

print st.split(" ")
print st.splitlines()
print st.partition(" ")

# set
s1 = set("hello world")
s2 = set("python world")

print s1.issubset(s2)
print s1.issuperset(s2)

s3 = s1.union(s2)
s4 = s1.intersection(s2)
s5 = s1.difference(s2)

# update set
s1.update(s2)
s1.intersection_update(s2)
s1.difference_update(s2)
s1.symmetric_difference_update(s2)

s1.add("a")
# raise KeyError if not contain
s1.remove("h")
s1.discard
# raise error if empty
s1.pop()
s2.clear()

# dictionary
d1 = dict(one=1, two=2, three=3, four=4)
d2 = dict([('a', 1), ('b', 2), ('c', 3), ('d', 4)])

del d2['d']

itr = iter(d1)

d3 = d2.copy()
d3.clear()

print d1.get('one', 0)
print d1.has_key('two')

d1.items()
d1.iteritems()
d1.iterkeys()
d1.itervalues()

d1.keys()
d1.values()

# return value of item
d1.pop('four', None)
d1.popitem()
d1.setdefault('five', 5)

d2.update(b=3, c=4)

# return dict view reflect dict changes
v1 = d2.viewitems()
v2 = d2.viewkeys()
v3 = d2.viewvalues()

# file operations
with open("./test.py") as f:
    for line in f:
        print line

f.fileno()
f.next()  # return next input file
f.read(1024)  # read size byte
f.readline()
f.readlines()
f.seek(1024)
f.tell()  # return current position
f.truncate(1024)
f.write("hello")
f.writelines("hello\n world")
f.flush()  # flush internal buffer
f.close()




