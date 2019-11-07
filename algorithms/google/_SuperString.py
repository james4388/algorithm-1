

# https://leetcode.com/problems/find-the-shortest-superstring/
# Give a list of string, superstring is smallest string cover all strings in list
# Travelling saleman problem
# Solution: form directed graph, with weight is overlap between string i and j
# store in matrix NxN, do bfs: put (mask, node, distance, path) into queue
# cache (mask, node, distance), if current distance < cache distance, then continue
# add another node, update mask and distance if node is not cover
