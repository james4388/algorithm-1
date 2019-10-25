

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

wq = WordSquareSolution()
print "square word...", wq.wordSquare(['area', 'ball', 'dear', 'lady', 'lead', 'yard'])
