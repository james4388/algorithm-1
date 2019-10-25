from collections import defaultdict


# Bull and cows game, count match and unmatch char
# https://leetcode.com/problems/bulls-and-cows/
# Preprocess secret if guess match, increase A, else increase char
# loop through remain guess, if char in A, decrease char
# Optmize when encounter number in secrect, increase count, number in guess
# decrease count, if 2 number same => increase bull, if different, if count(secret)
# < 0, increase cows (as guess decrease it), if count(guess) > 0, increase cows
# (as secret increase it)
class BullCowSolution:
    def getHint(self, secret, guess):

        table = defaultdict(int)
        A = B = 0
        p = []
        for idx in range(len(guess)):
            if secret[idx] == guess[idx]:
                A += 1
            else:
                table[secret[idx]] += 1
                p.append(idx)

        for idx in p:
            if guess[idx] in table and table[guess[idx]] > 0:
                table[guess[idx]] -= 1
                B += 1

        return '{}A{}B'.format(A, B)

    def getHint2(self, secret, guess):
        count = [0 for i in range(10)]
        bull = cow = 0
        for idx in range(len(guess)):
            s = int(secret[idx])
            g = int(secret[idx])
            if s == g:
                bull += 1
            else:
                if count[s] < 0:
                    cow += 1
                if count[g] > 0:
                    cow += 1

                count[s] += 1
                count[g] -= 1
        return '{}A{}B'.format(bull, cow)
