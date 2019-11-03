from collections import defaultdict


# Word square
# HARD: https://leetcode.com/problems/word-squares
# Use trie to store each word, for each word, check if every letter of it
# has other word with begin letter, use dfs to find list of words
class TrieNode:
    def __init__(self):
        self.children = [None for _ in range(26)]
        self.word = None
        self.possibles = []


class WordSquareSolution:
    def buildTrie(self, words):
        root = TrieNode()
        for word in words:
            p = root
            for char in word:
                idx = ord(char) - ord('a')
                if not p.children[idx]:
                    p.children[idx] = TrieNode()
                p = p.children[idx]
                p.possibles.append(word)
            p.word = word
        return root

    def dfs(self, wordlist, root, n):
        if len(wordlist) == n:
            return wordlist
        # next word index
        curr = len(wordlist)
        node = root
        for i in range(curr):
            char = wordlist[i][curr]
            idx = ord(char) - ord('a')
            if node.children[idx]:
                node = node.children[idx]
            else:
                return None

        if not node or not node.possibles:
            return None

        possibles = node.possibles
        for p in possibles:
            res = self.dfs(wordlist + [p], root, n)
            if res:
                return res
        return None

    def wordSquare(self, words):
        if not words:
            return []
        trie = self.buildTrie(words)
        candidates = []

        for word in words:
            for i in range(1, len(word)):
                idx = ord(word[i]) - ord('a')
                if not trie.children[idx]:
                    break
            else:
                candidates.append(word)

        res = []
        size = len(words[0])
        for word in candidates:
            l = self.dfs([word], trie, size)
            if l:
                res.append(l)
        return res


# Another way to implement trie using hashmap
# Runtime: O(N*26^L) where L is length of word, O(N*L)
class WordSquareMinimal:
    def square(self, wordlist, invert, res):
        if wordlist and len(wordlist) == len(wordlist[0]):
            res.append(wordlist)
            return
        # Next word prefix is combination of all current word at index n
        n = len(wordlist)
        prefix = ''.join(word[n] for word in wordlist)

        if prefix not in invert:
            return

        candidates = invert[prefix]
        for word in candidates:
            self.square(wordlist + [word], invert, res)

    def wordSquares(self, words):
        invert = defaultdict(set)
        res = []
        for word in words:
            for i in range(len(word)):
                invert[word[:i+1]].add(word)
        candidates = []

        # Find candidate for first word
        # if all char in word appears as starting char in other words
        for word in words:
            for char in word:
                if char not in invert:
                    break
            else:
                candidates.append(word)

        for word in candidates:
            self.square([word], invert, res)
        return res

wq = WordSquareMinimal()
print("square word...\n", wq.wordSquare(['area', 'ball', 'dear', 'lady', 'lead', 'yard']))
