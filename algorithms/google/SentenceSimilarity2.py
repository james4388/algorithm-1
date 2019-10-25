from collections import defaultdict


# Sentence similarity
# Given 2 sentences, and pairs of similar word, check if 2 sentences are
# similar, if word pair is transitive: A = B, B = C => A = C
# use hash table, and dfs to check if 2 word similar
# https://leetcode.com/problems/sentence-similarity
def areSentencesSimilar(words1, words2, pairs):
    if len(words1) != len(words2):
        return False

    hs = defaultdict(set)
    for (x, y) in pairs:
        hs[x].add(y)
        hs[y].add(x)

    def dfs(w1, w2, visited):
        if w1 == w2:
            return True

        visited.add(w1)

        for word in hs[w1]:
            if word not in visited:
                if dfs(word, w2, visited):
                    return True
        return False

    for w1, w2 in zip(words1, words2):
        if w1 == w2:
            continue

        if w1 in hs[w2] or w2 in hs[w1]:
            continue

        if not dfs(w1, w2, set()):
            return False

    return True
