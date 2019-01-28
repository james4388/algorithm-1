from collections import deque, OrderedDict, Counter
import requests


# Asteroid collions
# https://leetcode.com/problems/asteroid-collision/description/
# Use stack to store asteroid, if new asteroid < 0 and previous asteroid > 0
# collision happened, if new asteroid wins, continue to pop stack and collide
class Solution:
    def asteroidCollision(self, asteroids):
        """
        :type asteroids: List[int]
        :rtype: List[int]
        """
        if not asteroids:
            return []
        stack = []
        for asteroid in asteroids:
            while stack and asteroid < 0 < stack[-1]:
                if stack[-1] < -asteroid:
                    stack.pop()
                    continue
                elif stack[-1] == -asteroid:
                    stack.pop()
                break
            else:
                stack.append(asteroid)
        return stack


# Image cache
# if image does not exist, download and store image in cache
# else return image from cache, if exceed capacity, remove image from cache
# Cases: invalid url, file too big, empty file, file size equal capacity =>
# evict memory too frequent
class ImageCache(object):
    def __init__(self, capacity):
        self.cap = capacity
        self.cache = OrderedDict()
        self.size = 0

    def downloadImage(self, url):
        # print("Downloading image...", url)
        return requests.get(url).content

    def getImageLength(self, url):
        res = requests.get(url, stream=True)
        return int(res.headers.get('content-length', 0))

    def getImage(self, url):
        key = hash(url)
        if key in self.cache:
            length = len(self.cache[key])
            self.cache.move_to_end(key)
            print("{} {} {}".format(url, 'IN_CACHE', length))
        else:
            length = self.getImageLength(url)
            if length > self.cap:
                raise Exception('Cache overflowed.')

            if self.size + length > self.cap:
                self.removeCacheUntil(self.cap - length)

            self.size += length
            data = self.downloadImage(url)
            self.cache[key] = data
            print("{} {} {}".format(url, 'DOWNLOADED', length))

    def removeCacheUntil(self, size):
        while self.size > size:
            key, data = self.cache.popitem(last=False)
            self.size -= len(data)

    @classmethod
    def runTest(cls):
        cache = cls(524288)
        urls = ['http://i.imgur.com/xGmX4h3.jpg',
                'http://i.imgur.com/IUfsijF.jpg',
                'http://i.imgur.com/xGmX4h3.jpg']
        for url in urls:
            cache.getImage(url)


# ImageCache.runTest()


# Count words in file
class WordCount(object):
    def __init__(self):
        self.counter = Counter()

    def count(self, sentence):
        words = sentence.split()
        self.counter += Counter(words)

    def __repr__(self):
        return sorted(self.counter.items())


# Phonescreen
# Question 1: Given integer array, find product without number at each index
# answer: calculate total product, at each index divide product to number
# Handle edge case: there's zero value in array
# Follow up: what if cannot use division
# create right product array (multiple from right most position)
# loop from left and multiple with right size index
# https://leetcode.com/problems/product-of-array-except-self/description/
def productExceptSelf(nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    if not nums:
        return []
    n = len(nums)
    right = nums[:]
    for j in range(n-2, -1, -1):
        right[j] = right[j]*right[j+1]

    res = nums[:]
    mul = 1
    for i in range(n):
        if i < n-1:
            res[i] = mul * right[i+1]
        else:
            res[i] = mul
        mul = mul*nums[i]
    return res


# King escape: Queen at position x, y; check if King can move from position
# a, b to c, d without getting checked.
# https://codeforces.com/contest/1033/problem/A
# Fill board with queen move, and use DFS
class KingEscapeSolution(object):
    def fillBoard(self, board, queen):
        row, col = len(board), len(board[0])
        # fill row
        for j in range(col):
            board[queen[0]][j] = 1

        # fill column
        for i in range(row):
            board[i][queen[1]] = 1

        # fill diagonal
        m = min(queen[0], queen[1])
        i, j = queen[0] - m, queen[1] - m
        while i < row and j < col:
            board[i][j] = 1
            i += 1
            j += 1

        # find top right position that queen can reach, then move down left
        m = col - 1 - queen[1]
        i = queen[0] - m
        j = col - 1
        while i < row and j >= 0:
            board[i][j] = 1
            i += 1
            j -= 1

    def move(self, board, position, target):
        p, q = position
        row, col = len(board), len(board[0])
        if not (0 <= p < row) or not (0 <= q < col) or board[p][q] == 1:
            return False

        board[p][q] = 1

        for x, y in self.directions:
            nx, ny = p + x, q + y
            if (nx, ny) == target:
                return True

            if self.move(board, (nx, ny), target):
                return True
        return False

    def escape(self, n, kingPosition, target, queenPosition):
        board = [[0 for i in range(n)] for j in range(n)]
        self.fillBoard(board, queenPosition)
        self.directions = ((-1, -1), (-1, 0), (-1, 1), (0, 1),
                           (1, 1), (1, 0), (1, -1), (0, -1))
        return self.move(board, kingPosition, target)

ke = KingEscapeSolution()
print "king escape....", ke.escape(8, (1, 2), (6, 1), (3, 5))


# Square difference, check if a^2 - b^2 is prime number
# https://codeforces.com/contest/1033/problem/B
# if a - b != 1 return False, check a + b is prime

# Design data structure to store, delete and return min difference
# Use sorted linkedlist to store data, for easy add and delete
# Everytime delete or add new data, update the min diff
# 1-> 10 -> 11 min diff = 1, delete 10 -> min diff = 11 - 1 = 10
class MinDiffStore(object):
    pass

# Smallest range
# HARD: https://leetcode.com/problems/smallest-range/
# Given k sorted list, find smallest range that contains each element of list
# Merge k sorted list into one sorted list, run with window k and check for
# smallest window, merge: O(nlog(k)), check min windows: O(nk^2)
