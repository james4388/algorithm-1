#!/bin/python


class TrieNode:
    def __init__(self, val=None):
        self.children = {}
        self.value = val


class Trie:
    @classmethod
    def insert(cls, root, text, value):
        node = root
        i = 0

        while i < len(text):
            if text[i] in node.children:
                node = node.children[text[i]]
                i += 1
            else:
                break

        while i < len(text):
            node.children[text[i]] = TrieNode()
            node = node.children[text[i]]
            i += 1

        node.value = value

    @classmethod
    def find(cls, root, text):
        node = root
        for char in text:
            if char in node.children:
                node = node.children[char]
            else:
                return None
        return node.value


if __name__ == '__main__':
    root = TrieNode()

    keys = ['there', 'was', 'their', 'them', 'were', 'want']
    for k in keys:
        Trie.insert(root, k, k)

    print "Find key want...", Trie.find(root, "wa")
