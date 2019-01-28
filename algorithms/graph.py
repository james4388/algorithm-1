from collections import defaultdict, deque
from queue_stack import Queue, Stack
import sys


UNDISCOVERED = -1
DISCOVERED = 0
PROCESSED = 1
MAX = sys.maxint


class EdgeNode(object):
    def __init__(self, label, weight):
        self.label = label
        self.weight = weight


# Graph data structure
# Dictionary for each node {'A': [(B, 5), (C, 10)], 'E': [(D, 4)]....}
class Graph(object):
    def __init__(self, directed=True):
        self.directed = directed
        self.nvertices = 0
        self.nedges = 0
        self.degrees = defaultdict(int)
        self.edges = defaultdict(list)

    @classmethod
    def read_graph(cls, directed):
        g = Graph(directed)
        nedges = int(raw_input())
        for i in xrange(nedges):
            x, y, w = raw_input().split()
            g.insert_edge(x, y, int(w), directed)
        return g

    @classmethod
    def load_from_file(cls, fn, directed=True):
        g = Graph(directed)
        with open(fn, 'r') as fd:
            for line in fd:
                x, y, w = line.split()
                g.insert_edge(x, y, int(w), directed)
        return g

    def insert_edge(self, x, y, weight, directed):
        n = EdgeNode(y, weight)
        if x not in self.edges:
            self.nvertices += 1
            self.edges[x] = []

        if y not in self.edges:
            self.nvertices += 1
            self.edges[y] = []

        self.edges[x].append(n)
        self.degrees[x] += 1

        if not directed:
            self.insert_edge(y, x, weight, True)
        else:
            self.nedges += 1

    def print_graph(self):
        for k, edges in self.edges.iteritems():
            for edge in edges:
                print "%s---->%s: %s" % (k, edge.label, edge.weight)

    def init_bfs(self):
        self.status = {}
        self.parent = {}
        self.queue = Queue()
        # -1: UNDISCOVERED 0: DISCOVERED, 1: PROCESSED
        for x in self.edges:
            self.status[x] = UNDISCOVERED

    def process_edge(self, x, y, back=False):
        if back:
            if self.directed:
                print "Cycle found in directed graph from...", x, y
            elif self.parent.get(x) != y:
                print "Cycle found from %s to %s" % (x, y)

        print "Process edge %s--%s" % (x, y)

    def bfs(self, start):
        self.init_bfs()
        self.queue.enqueue(start)
        self.status[start] = DISCOVERED

        while len(self.queue) > 0:
            v = self.queue.dequeue()
            adj = self.edges.get(v, [])
            self.status[v] = PROCESSED
            for edge in adj:
                y = edge.label
                # If undirected, this edge already processed by y
                if self.status[y] != PROCESSED or self.directed:
                    self.process_edge(v, y)
                if self.status[y] == UNDISCOVERED:
                    self.queue.append(y)
                    self.status[y] = DISCOVERED
                    self.parent[y] = v

    def connected_component(self):
        comp = 0
        nodes = self.edges.keys()
        self.init_bfs()
        for node in nodes:
            if self.status[node] != PROCESSED:
                comp += 1
                self.bfs(node)
                print "Component %s includes: %s" % (comp, self.parent)

    def init_dfs(self):
        self.status = {}
        self.parent = {}
        # Descendant = (exit - entry)/2
        self.entry_time = {}
        self.exit_time = {}
        self.time = 0
        self.topo = Stack()

        for x in self.edges:
            self.status[x] = UNDISCOVERED

    def dfs(self, start):
        self.init_dfs()
        self.run_dfs(start)

    def process_vertex_late(self, v):
        self.topo.push(v)

    def print_topo(self):
        print "Topology sort..."
        while not self.topo.is_empty():
            print self.topo.pop()

    def run_dfs(self, start):

        self.status[start] = DISCOVERED
        self.time += 1
        self.entry_time[start] = self.time

        adj = self.edges.get(start, [])
        for edge in adj:
            y = edge.label
            if self.status[y] == UNDISCOVERED:
                # Tree edge in undirected graph
                self.process_edge(start, y)
                self.parent[y] = start
                self.run_dfs(y)
            elif self.status[y] == DISCOVERED:
                self.process_edge(start, y, back=True)
            elif self.directed:
                self.process_edge(start, y, back=False)

        self.process_vertex_late(start)
        self.time += 1
        self.exit_time[start] = self.time
        self.status[start] = PROCESSED

    def topology_sort(self):
        self.init_dfs()
        for x in self.edges:
            if self.status[x] == UNDISCOVERED:
                self.run_dfs(x)

        self.print_topo()

    # Each round select minium weighted edge into tree
    # Add vertex to tree and update distance of other nontree vertex to tree
    def prim_mst(self, v):
        parent = {}
        intree = {}
        distance = {}
        for x in self.edges:
            distance[x] = MAX
            intree[x] = False

        distance[v] = 0

        for i in range(self.nvertices):
            intree[v] = True
            # Update distance of other vertex from v
            for edge in self.edges[v]:
                y = edge.label
                w = edge.weight
                if not intree[y] and distance[y] > w:
                    distance[y] = w
                    parent[y] = v

            # Select edge with minimum distance
            m_dist = MAX
            for m, n in distance.items():
                if intree[m]:
                    continue

                if m_dist > n:
                    m_dist = n
                    v = m

            print "Add edge...", parent.get(v), v

    # Djikstra algorithm, simillar to Prim's spaning tree algorithm
    # Difference is updating distance x = min(distance x, distance v + w(v, x))
    def shortest_path(self, v):
        parent = {}
        intree = {}
        distance = {}
        for x in self.edges:
            distance[x] = MAX
            intree[x] = False
            parent[x] = None

        distance[v] = 0

        for i in range(self.nvertices):
            intree[v] = True
            # Update distance of other vertex from v
            for edge in self.edges[v]:
                y = edge.label
                w = edge.weight
                # Update distance of adjection nodes
                if distance[y] > distance[v] + w:
                    distance[y] = distance[v] + w
                    parent[y] = v

            # Select edge with minimum distance
            m_dist = MAX
            for m, n in distance.items():
                if intree[m]:
                    continue

                if m_dist > n:
                    m_dist = n
                    v = m

        return parent, distance

    def find_shortest_path(self, s, t):
        parent, distance = self.shortest_path(s)
        print "Distance...", distance, parent
        m = t
        stack = Stack()
        stack.push(m)
        while m != s:
            k = parent[m]
            if not k:
                print "No path found..."
                break
            stack.push(k)
            m = k

        while not stack.is_empty():
            print stack.pop()

    # Floy Bellman-ford, matrix graph
    def floyd(self, n, graph):
        for k in range(1, n+1):
            for i in range(1, n+1):
                for j in range(1, n+1):
                    if graph[i][k] and graph[k][j]:
                        through_k = graph[i][k] + graph[k][j]
                        if not graph[i][j] or graph[i][j] > through_k:
                            graph[i][j] = through_k

    def isBipartie(self, n, graph):
        colors = [-1 for _ in range(n)]
        queue = deque([0])
        colors[0] = 1

        while queue:
            u = queue.popleft()
            for v in range(n):
                if graph[u][v]:
                    if colors[v] == -1:
                        colors[v] = 1 - colors[u]
                    else:
                        if colors[v] == colors[u]:
                            return False
        return True


# "Find" on parent dictionary, if parent[v] = v return v
# else parent[v] = find(parent[v])
# "Union" edge: u -> v, u father of v, find parent of u = parent_u,
# parent[v] = parent_of_u
# Path compression and union by rank
class UnionFind:
    # Compress path as finding
    def __init__(self, num):
        self.parent = list(range(num))
        self.rank = [0 for i in range(num)]

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    # Optimize rank on union, choose parent node with higher rank
    def union(self, x, y):
        xr = self.find(x)
        yr = self.find(y)

        # Cannot union, there's cycle
        if xr == yr:
            return False

        if self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.parent[yr] = xr
        else:
            self.parent[xr] = yr
            self.rank[yr] += 1


def detectCycle(edges):
    uf = UnionFind(len(edges))
    for edge in edges:
        if not uf.union(*edge):
            return edge
    return False

if __name__ == '__main__':
    g = Graph.load_from_file('data_graph.txt', directed=True)
    # g.print_graph()
    # g.bfs('A')
    # g.connected_component()
    # g.dfs('A')
    # g.topology_sort()
    # g.prim_mst('A')
    g.find_shortest_path('A', 'G')
