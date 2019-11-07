from collections import Counter


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
