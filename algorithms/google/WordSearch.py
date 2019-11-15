

# Word boggle, find words from dictionary in 2D matrix
# HARD: Word search in 2D matrix
# give alist of word, find each word that appear in matrix
# https://leetcode.com/problems/word-search-ii/description/
# Use Trie to store list word and use DFS
class TrieNode:
    def __init__(self):
        self.children = [None for _ in range(26)]
        self.word = None
        self.possibles = []


class Solution:
    def buildTrie(self, words):
        root = TrieNode()
        for word in words:
            p = root
            for char in word:
                idx = ord(char) - ord('a')
                if not p.children[idx]:
                    p.children[idx] = TrieNode()
                p = p.children[idx]
            p.word = word
        return root

    def dfs(self, matrix, i, j, p, res):
        c = matrix[i][j]
        if c == '#' or not p.children[ord(c) - ord('a')]:
            return
        p = p.children[ord(c) - ord('a')]
        if p.word:
            res.append(p.word)
            p.word = None

        matrix[i][j] = '#'
        if i > 0:
            self.dfs(matrix, i - 1, j, p, res)
        if j > 0:
            self.dfs(matrix, i, j - 1, p, res)
        if i < len(matrix) - 1:
            self.dfs(matrix, i + 1, j, p, res)
        if j < len(matrix[0]) - 1:
            self.dfs(matrix, i, j+1, p, res)
        matrix[i][j] = c

    def findWords(self, board, words):
        trie = self.buildTrie(words)
        res = []

        for i in range(len(board)):
            for j in range(len(board[0])):
                self.dfs(board, i, j, trie, res)
        return res
