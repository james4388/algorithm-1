'''
https://www.glassdoor.com/Interview/Amazon-Software-Development-Engineer-II-Interview-Questions-EI_IE6036.0,6_KO7,39.htm

Question 1: Design single system, single user table booking, 16 tables capacity
of 4 and 16 tables capacity of 8. What if a party of 16 requests a table. You
can join table next each other.

Question 2: Design a deck cards, assume 10 millions users using this deck

Question 3: Given array with only 0s and 1s, find index of first 1, if array
is infinite how would you find in O(logN)

Question 4: Design speed camera system, to upload snapshot for vehicles, query
find speeder for location and date range.

Question 5: Find out top trending products of last hour/day/month. Given we
have access to real-time stream of sold product ids.
A: How stream of data look likes: e.g [10, 5, 5, 5, 4, 3, 3, 2, 10, 10, 6]

Question 6: Design parking lots
* What kind of parking lots: open space, or building
* How many parking lots: hundreds, thousand
* Type of parking lots: busy, available, reserved
* Vehicles parking lots: compact, medium, large, ex large
* Prices strategy for each type of parking lots, vehicles
* Design abstract classes:
- Vehicles -> concrete classes (Car, motorbike, truck, bus), size, canfitinspot
- ParkingLot -> List level, methods: placeVehicles
- Level -> list of Parking spot

Question 7: Design order tracking system: once order received, assigned to
delivery boy, update status for every state, received with expected time,
delivery boy assigned, order pick up, order delivered. Hints: use observer
pattern to register users, Subject -> register Observer

Question 8: Design building access system for employee

Question 9: Design Unix file system
- Class Entry: name, parent, created, delete, getPath
- Subclass: File -> content, size
    - Directory: contents List <Entry>, size

Question 10: Design stock client facing service that provides stock open, high,
low, closed
- FTP file server: easy view, parse and back up, drawback: add additional data
will break client parsing
- SQL database: benefits: easy query processing, rolling and backup data,
integrate to existing application; drawbacks: heavier data, difficult for human
to read data -> implement extra layer to view and maintain data, security issue
- XML file: easy share, parsing, backup data; add new data does not break the
parser, drawbacks: send all data, query need to parse all data

Question 11: Design data structure for large social network like Linkedin,
facebook, design algorithm to find shortest path between 2 peoples
- Use BFS to search for connection from source to destination, can improve by
bi-directional BFS, keep list of visited nodes from source and dest, if
collision happen merge 2 pathnodes.

Question 12: Detect duplicate from 10 billion urls
- Average 100 chars per url, 4 bytes per char => 4TB storage need => do not fit
in memory of 1 machine.
- Solution 1: Disk storage: split urls into 4000 chunks, each chunk is 1GB
store urls in file, with filename <hash % 4000>.txt, so urls with same hash
will be in same file => load each file into memory, create hash table, to
detect duplicate.
- Solution 2: Multiple machines: send each chunks to 4000 machines, pros: can
run in parallel => run faster, need to handle failure if one machines failed

Question 13: Search server with 100 machine, queryProcess is expensive
- Requirement: easy look up query, expire old data
- A linkedlist easy to purge data but not look up, a hash table easy to look up
- Combine both data structure => LRU cache
- Single cache: quick return, not effective as query sent to another machine
become fresh query.
- Cache copy: each machine has a copy of cache, size larger for N machine,
update cache means firing off N machine.
- Segment cache: each machine store a segment of cache, use formula hash(query)
% N, then machine would know which hold the cache.
- Update cache: automatic time out, may use different time out based on how
frequently content update or url update.

Question 14: Sale rank for category
- Scope: Total sale over time, product can be in multiple categories
- Assumption: Do not need to have 100% up-to-date.
- Data should be updated every hours.
- Component: sale data store in database, every hour query database by category,
compute sale rank and store in some data table in caches, frontend will pull
sale rank from this table.
- Analytics: hitting database would be expensive => track total sale last week
=> update sale count for each day of week, need separate table for category:
product -> category, then we just need to join 2 tables.
    - Immediately commit into database => more write, we can cache purchase
data into cache or log file and periodically process cache.
- Join is expensive => list each product one per category
- Database query is expensive => store in file with product id and timestamp,
use map-reduce, each category has its own folder, product written in each of
category => overall ranking can do a merge

Question 15: Design online chat system:
- Features: add user, update, remove user; add friend request, approved, cancel
friend request; create group chat, invite to private chat; message
- Design classes: UserService, User, Chat -> PrivateChat, GroupChat,
FriendRequest, Message

Question 16: Design pastebin services
- Features: user enters block of text and get generate random url, user enters
url to view content (Does the url expire after period of time?), track
analytics (monthly visitors?), user is anonymous or logged in?
- Question: number of urls generate per day?, how long is the text content?,
handle duplicate content?, delete url after long time not visisted?, allow edit
content?
- Note: generate url using hash function MD5 (ip + timestamp) -> convert to
base 62 [a,b,...zA,B,..,Z,0...9]
 - Analytics: store analytics in log file each line with format: url datetime,
 use mapreduce -> map: yield (url, year_month), 1 -> reduce (key, values):
 yield key, sum(values)
 - Bottlenecks: webserver api -> load balancer, heavy read -> use cache

Question 17: Design Amazon warehouse
- Features: place package and able to deliver package.
- Question: how many packages can put in warehouse? millions
 - how package are organized in warehouse? by category? by class? by size?
 - Suppose warehouse is very big, how it is divided? by zone, region?
 - How warehouse handle an order with multiple package? progressive assemble
 - Shelves slot and package size? small, medium, large
- Design classes: Warehouse -> Regions -> Shelf [slots], Package, Order, Picker
- Each regions has a list of picker
- Warehouse: dispatch_order(order), place_package
- Shelf: place package -> find fit size empty slots.
- Picker: package list to pick, go to their region and fetch the package.

Question 18: Design Amazon buy together recommandation system
- Features: recommend top products buying together for users
- Questions: what criteria to find list of recommendation products? user buying
history, other users buying history.
 - How many products for recommendation? top 5 products
 - How do I rank for list of 5 product? by total sales over a year, a month
 or a week.
 - What informations that we have for finding total sales? transaction history
 order no - item1 - quantity1 - item2 - quantity2 - timestamp
 - How frequently update the recommend items? is it real time?
 - Is it personalize by user's interest?
- Component: client -> load balancer -> web server, sale data -> SQL write,
Recommendation system -> read data -> do calculation -> cache object store.
- Map reduce: map (item1 - item 2) => (item1, item2, year_month), 1
 - reduce: (key, values) => key, sum(values)
- Result: item1 - item2 - total => sort and take top 5 items

Question 19: Allocation 64-bit id system: generate roughly sorted id
- Use timestamp, worker id choose at startup by zookeeper, sequence numbers
are per thread
- id = '{timestamp}{data_center_id}{worker_id}{sequence_bit}'
- get milli seconds from epoch

```
import datetime
epoch = datetime.datetime.utcfromtimestamp(0)

def unix_time_millis(dt):
    return (dt - epoch).total_seconds() * 1000.0
```

Tips:
- How you would think about the problem space
- How you think about bottlenecks
- What you can do to remove these bottlenecks.

Ask questions:
- Helps you narrow the scope of what you’re supposed to do
- Helps clarify what the user expectation of the system is
- Gives you direction about where to proceed
- Informs you of possible bottlenecks/problem areas

'''
from abc import ABCMeta


# Design hash table
class HashItem(object):
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def __str__(self):
        return '%s:%s' % (self.key, self.value)


class HashTable(object):
    def __init__(self, size):
        self.size = size
        self.table = [[] for _ in range(size)]

    def _hash_(self, key):
        return key % self.size

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError('Invalid key.')

        idx = self._hash_(key)
        for item in self.table[idx]:
            if item.key == key:
                return item.value
        raise KeyError('Key not found.')

    def __setitem__(self, key, value):
        if not isinstance(key, int):
            raise TypeError('Invalid key.')

        idx = self._hash_(key)
        for item in self.table[idx]:
            if item.key == key:
                item.value = value
                return
        self.table[idx].append(HashItem(key, value))

    def __delitem__(self, key):
        if not isinstance(key, int):
            raise TypeError('Invalid key.')

        idx = self._hash_(key)
        for i, item in enumerate(self.table[idx]):
            if item.key == key:
                del self.table[idx][i]
                return
        raise KeyError('Key not found')

    def __str__(self):
        m = [[str(x) for x in y] for y in self.table if y]
        return str(m)


'''
Cracking the code interview p307
Design call handler center with 3 levels of employees: respondent,
manager and director. Incoming call will be allocated to first free
responsedent, escalate to manager and then director if no one free.
Implement dispatchCall()
- 3 classes for each employee types, sub-class from employee to keep common
infos: name, phone, address, class CallHander, Call, caller
'''
from enum import Enum


class Rank(Enum):
    RESPONDENT = 0
    MANAGER = 1
    DIRECTOR = 2


class Employee(object):
    def __init__(self):
        self.currentCall = None
        self.rank = Rank.RESPONDENT

    def receiveCall(self, call):
        pass

    def completedCall(self, call):
        pass

    def assignNewCall(self, call):
        pass

    def isFree(self):
        return self.currentCall is None

    def escalateCall(self):
        pass


class Respondent(Employee):
    rank = Rank.RESPONDENT


class Manager(Employee):
    rank = Rank.MANAGER


class Director(Employee):
    rank = Rank.DIRECTOR


class Call(object):
    def __init__(self, caller):
        self.caller = caller
        self.rank = Rank.RESPONDENT
        self.handler = None

    def reply(self, message):
        pass

    def setRank(self, rank):
        pass

    def setHandler(self, handler):
        pass


class CallHandler(object):
    def __init__(self, numRespondent, numManager, numDirector):
        self.employees = self.initEmployeeList()
        self.callQueue = [[], [], []]

    # Return list of list employee by rank List <List<Employee>>
    def initEmployeeList(self, r, m, d):
        pass

    def getHandlerForCall(self, call):
        pass

    def dispatchCall(self, call):
        emp = self.getHandlerForCall(call)
        if emp:
            emp.receiveCall(call)
            call.setHandler(emp)
        else:
            call.reply("Please wait for available respondent.")
            self.putCallIntoQueue(call)

    def putCallIntoQueue(self, call):
        self.callQueue[call.rank.value].append(call)


''' Design employee access building
Question:
- Which type of employees? engineer, supervisor, director?
- which criteria to allow access or deny access ? employee type, team?
- Does it restrict on access time ?from 9AM to 3PM, until date ?
- Is it allowed to access whole building or just some rooms of building?
- Can access be revoked? yes
- Do we need log the access for employee? yes
Design classes: Employee -> Engineer, Supervisor, Director; Building -> Room;
Team; AccessRule -> TypeAccess, TeamAccess; AccessLog; AccessControl

Drawbacks: has to create subclass AccessRule for new rule types, cannot create
complex rule => create CustomRule class
'''
class EmpType(Enum):
    ENGINEER = 0
    SUPERVISOR = 1
    DIRECTOR = 2


class EmployeeAC(object):
    def __init__(self, name, team):
        self.name = name
        self.team = team
        self._type = None

    def checkin(self, facility):
        return self.facility.can_access(self)

    def checkout(self, facility):
        return self.facility.remove(self)


class Engineer(EmployeeAC):
    _type = EmpType.ENGINEER


class Supervisor(EmployeeAC):
    _type = EmpType.SUPERVISOR


class Team(object):
    def __init__(self, name):
        self.name = name


class Facility(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, address):
        self.name = name
        self.address = address
        self.access_rules = []
        self.users = []

    def addRule(self, rule):
        pass

    def removeRule(self, rule):
        pass

    def can_access(self, employee):
        for rule in self.access_rules:
            if rule.allow_access(employee):
                return True
        return False

    def remove(self, employee):
        pass


class Building(Facility):
    pass


class Room(Facility):
    def __init__(self, name, building):
        super(Facility, self).__init__(name, building.address)
        self.building = building


class AccessRule(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.access_list = []
        self.start_time = None
        self.end_time = None

    def addAccess(self, obj):
        self.access_list.append(obj)

    def revokeAccess(self, obj):
        pass

    def allow_access(self, employee):
        pass


class EmpTypeAccess(AccessRule):
    def allow_access(self, employee):
        for _type in self.access_list:
            if employee._type == _type:
                return True
        return False


class TeamAccess(AccessRule):
    def allow_access(self, employee):
        for team in self.access_list:
            if employee.team == team:
                return True
        return False


class AccessLog(object):
    pass
