from collections import defaultdict


# Longest word in dictionary is subsequence
# https://leetcode.com/problems/longest-word-in-dictionary-through-deleting/
# https://techdevguide.withgoogle.com/resources/find-longest-word-in-dictionary-that-subsequence-of-given-string/#!
# give a dictionary find word in dictionary with longest substring of string
# Optmize: preprocess string s to have position of each character
# e.g a -> [1, 2, 3], p -> [10, 11], check substring invole binary search
# in this list
class DictionarySolution:
    # Check if t is subsequence of s
    def isSubstring(self, s, t):
        if len(t) > len(s):
            return False
        i = j = 0
        while i < len(s) and j < len(t):
            if s[i] != t[j]:
                i += 1
            else:
                i += 1
                j += 1
        return j == len(t)

    def longestSubstring(self, words, string):
        words.sort(key=lambda x: len(x), reverse=True)
        for word in words:
            if self.isSubstring(string, word):
                return word
        return None

    def longestSubstring2(self, words, string):
        letter_positions = defaultdict(list)
        for idx, c in enumerate(string):
            letter_positions[c].append(idx)

        for word in sorted(words, key=lambda w: len(w), reverse=True):
            pos = 0
            for letter in word:
                if letter not in letter_positions:
                    break

                possible_positions = [p for p in letter_positions[letter] if p >= pos]
                if not possible_positions:
                    break
                pos = possible_positions[0] + 1
            else:
                return word

ds = DictionarySolution()
