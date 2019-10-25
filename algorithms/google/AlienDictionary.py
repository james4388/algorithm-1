from collections import defaultdict


# Alien dictionary
# https://www.geeksforgeeks.org/given-sorted-dictionary-find-precedence-characters/
# Create graph with each  vertex is char, compare each word pair, if missmatch
# add it as edge
class Graph(object):
    def __init__(self):
        self.adjections = defaultdict(set)

    def addEdge(self, _from, to):
        self.adjections[_from].add(to)

        if to not in self.adjections:
            self.adjections[to] = set()

    def runDfs(self, vertex):
        print "run dfs..", vertex, self.status
        # Discover
        self.status[vertex] = 1

        for adj in self.adjections[vertex]:
            if self.status[adj] == 0:
                self.runDfs(adj)

        self.status[vertex] = 2
        self.stack.append(vertex)

    def topo_sort(self):
        self.stack = []
        self.status = {k: 0 for k in self.adjections}
        for vertex in self.adjections:
            if self.status[vertex] == 0:
                self.runDfs(vertex)
        return self.stack

    def __str__(self):
        m = ['{} -> {}'.format(k, v) for (k, v) in self.adjections.items()]
        return '\n'.join(m)


class AlienDictionary(object):
    def buildGraph(self, words):
        graph = Graph()

        for pair in zip(words, words[1:]):
            a, b = pair
            for i in range(min(len(a), len(b))):
                if a[i] != b[i]:
                    graph.addEdge(a[i], b[i])
                    break
        return graph

    def findOrder(self, words):
        graph = self.buildGraph(words)
        print "graph...", graph
        topo = graph.topo_sort()
        return topo[::-1]
