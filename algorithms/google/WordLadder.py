import string
from collections import deque

# word ladder
# https://leetcode.com/problems/word-ladder/discuss/40729/Compact-Python-solution
class WordSolution(object):
    def addNextWords(self, word, h, tmp):
        for i in range(len(word)):
            for ch in string.ascii_letters:
                nw = word[:i] + ch + word[i+1:]
                if nw in h:
                    tmp.append(nw)
                    h.pop(nw)

    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        if not wordList:
            return 0

        q = deque([beginWord])
        count = 0
        h = set(wordList)
        h.remove(beginWord, None)

        while q:
            count += 1
            tmp = deque([])
            while q:
                word = q.popleft()
                if word == endWord:
                    return count

                for i in range(len(word)):
                    for ch in string.ascii_letters:
                        nw = word[:i] + ch + word[i+1:]
                        if nw in h:
                            tmp.append(nw)
                            h.remove(nw)
            q = tmp
        return 0

# print "word ladder...", WordSolution().ladderLength('a', 'b', ['a', 'b', 'c'])
