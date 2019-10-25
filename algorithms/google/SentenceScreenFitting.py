

# Sentence screen fitting
# https://leetcode.com/problems/sentence-screen-fitting
# Concat all word list to one sentence, everytime find character of ending
# if not empty space, reduce it to space position
class SentenceSolution:
    def wordsTyping(self, sentence, rows, cols):
        """
        :type sentence: List[str]
        :type rows: int
        :type cols: int
        :rtype: int
        """
        s = ' '.join(sentence) + ' '
        n = len(s)
        i = 0
        for r in range(rows):
            i += cols
            while i > 0 and s[i % n] != ' ':
                i -= 1
            i += 1
        return i//n
