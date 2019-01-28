from collections import deque, defaultdict, Counter, OrderedDict


# Game of life
# https://leetcode.com/problems/game-of-life/
# Store cell will turn to die and will survive and do post update
# Optimize use 2 bits to store current state and next state: 10 -> change from
# die to live, 01 -> live to die, [next_state, curr_state]
class GameOfLifeSolution(object):
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def countLive(self, board, i, j):
        count = 0
        m, n = len(board), len(board[0])
        for d in self.dirs:
            x, y = i + d[0], j + d[1]
            if 0 <= x < m and 0 <= y < n and board[x][y] == 1:
                count += 1
        return count

    def willSurvive(self, board, i, j):
        count = self.countLive(board, i, j)
        return count == 3

    def willDie(self, board, i, j):
        count = self.countLive(board, i, j)
        return count < 2 or count > 3

    def gameOfLife(self, board):
        """
        :type board: List[List[int]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        if not board:
            return
        m, n = len(board), len(board[0])
        changeDie = []
        changeLive = []

        for i in range(m):
            for j in range(n):
                if board[i][j] == 1:
                    if self.willDie(board, i, j):
                        changeDie.append((i, j))
                else:
                    if self.willSurvive(board, i, j):
                        changeLive.append((i, j))

        for (x, y) in changeDie:
            board[x][y] = 0

        for (x, y) in changeLive:
            board[x][y] = 1


# Phone directory
# https://leetcode.com/problems/design-phone-directory/
class PhoneDirectory:

    def __init__(self, maxNumbers):
        """
        Initialize your data structure here
        @param maxNumbers - The maximum numbers that can be stored in the phone directory.
        :type maxNumbers: int
        """
        self.numbers = set(range(maxNumbers))

    def get(self):
        """
        Provide a number which is not assigned to anyone.
        @return - Return an available number. Return -1 if none is available.
        :rtype: int
        """
        return self.numbers.pop() if self.numbers else -1

    def check(self, number):
        """
        Check if a number is available or not.
        :type number: int
        :rtype: bool
        """
        return number in self.numbers

    def release(self, number):
        """
        Recycle or release a number.
        :type number: int
        :rtype: void
        """
        self.numbers.add(number)


# Hit counter
# https://leetcode.com/problems/design-hit-counter/
# Use array of size 300 to count number of hit, if timestamp goes out of range
# override it
# Concurrency issue: add a write lock => slow down, move to distributed
# sum all counters across all machine, add cache for read
class HitCounter:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.counter = [[0, x + 1] for x in range(300)]

    def hit(self, timestamp):
        """
        Record a hit.
        @param timestamp - The current timestamp (in seconds granularity).
        :type timestamp: int
        :rtype: void
        """
        idx = timestamp % 300
        if self.counter[idx][1] == timestamp:
            self.counter[idx][0] += 1
        else:
            self.counter[idx] = [1, timestamp]

    def getHits(self, timestamp):
        """
        Return the number of hits in the past 5 minutes.
        @param timestamp - The current timestamp (in seconds granularity).
        :type timestamp: int
        :rtype: int
        """
        count = 0
        for (hit, ts) in self.counter:
            if timestamp - ts < 300:
                count += hit
        return count


# Add and search word data structure Trie
# https://leetcode.com/problems/add-and-search-word-data-structure-design/description/
class Node(object):
    def __init__(self, val=None):
        self.val = val
        self.children = {}


class WordDictionary(object):
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = Node()

    def addWord(self, word):
        """
        Adds a word into the data structure.
        :type word: str
        :rtype: void
        """
        node = self.root
        for i in range(len(word)):
            if word[i] in node.children:
                node = node.children[word[i]]
            else:
                break

        for j in range(i, len(word)):
            node.children[word[j]] = Node()
            node = node.children[word[j]]
        node.val = word

    def search(self, word):
        """
        Returns if the word is in the data structure. A word could contain the
        dot character '.' to represent any one letter.
        :type word: str
        :rtype: bool
        """
        nodes = [self.root]
        print self.root.children
        i = 0
        while i < len(word):
            tmp = []
            for node in nodes:
                if word[i] in node.children:
                    tmp.append(node.children[word[i]])
                elif word[i] == '.':
                    tmp.extend(node.children.values())
            if not tmp:
                return False
            nodes = tmp
            i += 1
        return True


# Design tic toc toe game
# https://leetcode.com/problems/design-tic-tac-toe
# Assign value for each player move, win score = n*playerScore
# Store total value for row, column and 2 diagonals (x+y = n-1 and y-x = 0)
# Runtime O(1), space O(n)
class TicTacToe:

    def __init__(self, n):
        """
        Initialize your data structure here.
        :type n: int
        """
        self.rows = [0 for _ in range(n)]
        self.columns = [0 for _ in range(n)]
        # First diagonal x+y, second y-x
        self.diagonal = [0, 0]
        self.score = {1: 1, 2: n+1}
        self.win = {1: n, 2: (n+1)*n}
        self.size = n

    def move(self, row, col, player):
        """
        Player {player} makes a move at ({row}, {col}).
        @param row The row of the board.
        @param col The column of the board.
        @param player The player, can be either 1 or 2.
        @return The current winning condition, can be either:
                0: No one wins.
                1: Player 1 wins.
                2: Player 2 wins.
        :type row: int
        :type col: int
        :type player: int
        :rtype: int
        """
        score = self.score[player]
        win_score = self.win[player]
        self.rows[row] += score
        if self.rows[row] == win_score:
            return player

        self.columns[col] += score
        if self.columns[col] == win_score:
            return player
        if col - row == 0:
            self.diagonal[1] += score
            if self.diagonal[1] == win_score:
                return player
        if col + row == self.size - 1:
            self.diagonal[0] += score
            if self.diagonal[0] == win_score:
                return player
        return 0


# Snakegame
# https://leetcode.com/problems/design-snake-game
# Use deque to store snake, pop food from queue
class SnakeGame:

    def __init__(self, width, height, food):
        """
        Initialize your data structure here.
        @param width - screen width
        @param height - screen height
        @param food - A list of food positions
        E.g food = [[1,1], [1,0]] means the first food is positioned at [1,1], the second is at [1,0].
        :type width: int
        :type height: int
        :type food: List[List[int]]
        """
        self.width = width
        self.height = height
        self.food = deque(food)
        self.snake = deque([[0, 0]])
        # For checking if snake eats its body
        self.dirs = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

    def isValidMove(self, move):
        return (0 <= move[0] < self.height and 0 <= move[1] < self.width and
                (move not in self.snake or move == self.snake[-1]))

    def move(self, direction):
        """
        Moves the snake.
        @param direction - 'U' = Up, 'L' = Left, 'R' = Right, 'D' = Down
        @return The game's score after the move. Return -1 if game over.
        Game over when snake crosses the screen boundary or bites its body.
        :type direction: str
        :rtype: int
        """
        head = self.snake[0]
        delta = self.dirs[direction]
        nextMove = [head[0] + delta[0], head[1] + delta[1]]
        if not self.isValidMove(nextMove):
            return -1

        if self.food and nextMove == self.food[0]:
            self.food.popleft()
        else:
            self.snake.pop()

        self.snake.appendleft(nextMove)

        return len(self.snake) - 1


# Design autocomplete system
# https://leetcode.com/problems/design-search-autocomplete-system/
# Use trie to store words for searching, each level store a list of sentences
class TrieNode:
    def __init__(self):
        self.branches = {}
        self.s = Counter()


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, freq=1):
        node = self.root

        for char in word:
            if char not in node.branches:
                node.branches[char] = TrieNode()
            node = node.branches[char]
            node.s[word] += freq


class AutocompleteSystem:

    def __init__(self, sentences, times):
        """
        :type sentences: List[str]
        :type times: List[int]
        """
        trie = Trie()
        for word, freq in zip(sentences, times):
            trie.insert(word, freq)
        self.trie = trie
        self.currSearch = ''
        self.node = trie.root

    def input(self, c):
        """
        :type c: str
        :rtype: List[str]
        """
        if c == '#':
            self.trie.insert(self.currSearch, 1)
            self.currSearch = ''
            self.node = self.trie.root
            return []
        else:
            self.currSearch += c
            if self.node and c in self.node.branches:
                self.node = self.node.branches[c]
                items = sorted((-count, word) for word, count in self.node.s.items())
                return [x[1] for x in items[:3]]
            else:
                self.node = None
                return []


# LRU Cache
# Solution1: use doubly linkedlist, solution2: use python built-in ordereddict
# https://leetcode.com/problems/lru-cache/
class LRUCache:
    def __init__(self, Capacity):
        self.size = Capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        val = self.cache[key]
        self.cache.move_to_end(key)
        return val

    def put(self, key, val):
        if key in self.cache:
            del self.cache[key]
        self.cache[key] = val
        if len(self.cache) > self.size:
            # First in first out
            self.cache.popitem(last=False)


class MazeSolution():
    UP = 0
    LEFT = 1
    RIGHT = 2
    DOWN = 3
    EXIT = 2
    PATH = 0
    OBS = 1

    def move(self, puzzle, i, j, d):
        # print "move...", i, j, d
        dir_row = [-1, 0, 0, 1]
        dir_col = [0, -1, 1, 0]
        step = 0
        while (0 <= i < len(puzzle) and 0 <= j < len(puzzle[0]) and
               puzzle[i][j] != self.OBS):

            if puzzle[i][j] == self.EXIT:
                return i, j, step
            i += dir_row[d]
            j += dir_col[d]
            step += 1

        return i - dir_row[d], j - dir_col[d], step - 1

    def iceCavePuzzle(self, puzzle, x, y):
        # obstacle = 1, path = 0, exit = 2
        row = len(puzzle)
        col = len(puzzle[0])

        _min = row*col + 1
        queue = deque([])
        num_dirs = 4

        # queue item, x, y, step, dir
        queue.append((x, y, 0, -1))
        while queue:
            item = queue.popleft()
            i, j, step, d = item
            if puzzle[i][j] == self.EXIT:
                _min = min(_min, step)
                continue
            elif puzzle[x][y] == self.OBS:
                continue

            for k in range(num_dirs):
                # Avoid going backward with current direction
                if k != num_dirs - 1 - d:
                    p, q, s = self.move(puzzle, i, j, k)
                    # print "after move...", p, q, s
                    if s > 0:
                        queue.append((p, q, step + s, k))

        return _min if _min <= row * col else -1

maze = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 2], [0, 1, 0, 0]]
print "solve maze puzzle...", MazeSolution().iceCavePuzzle(maze, 0, 1)

'''
LFU Cache least frequently use cache
https://leetcode.com/problems/lfu-cache/
Solution1:
- Use hashmap to store frequency to LRU linkedlist and one hashmap to store key
value, everytime get, put key increase frequency, remove current node out of
current freq and put into new freq hashmap
'''

'''
- Design min max all one: support increase, decrease, max, min
- https://leetcode.com/problems/all-oone-data-structure/
- Use hash table: store key value and one hashmap to store count and set of key
- when increase key, move key to new count, exceed max, update max, if current key
is in min, if no min key left, increase min
- when decrease key, move key to new count, if current key is in max, check if
no max key left, decrease max, if key is in min, update min
'''

'''
Insert delete get random by frequency
- https://leetcode.com/problems/insert-delete-getrandom-o1-duplicates-allowed/
- Use list to store insert, hashmap set to store index of same element
- Insert: append element, add to hashmap index of it
- Remove: pop last index from hashmap set, swap value of last element to index
n - 1, and update indices for last element
- Get random: use normal random function
'''

'''
Find median in infinite data stream
- https://leetcode.com/problems/find-median-from-data-stream/
- Use 2 heaps max heap lo and min heap hi
- Add element: offer to lo, and pop 1 element from lo to hi, if length of hi
larger than lo, pop from hi and offer to lo
'''
