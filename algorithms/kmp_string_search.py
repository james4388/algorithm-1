

class KMP:
    def partial(self, pattern):
        """ Calculate partial match table: String -> [Int]"""
        m = len(pattern)
        ret = [0 for i in range(m)]
        j = 0

        for i in range(1, m):
            while j > 0 and pattern[j] != pattern[i]:
                j = ret[j - 1]

            if pattern[j] == pattern[i]:
                j += 1

            ret[i] = j

        return ret

    def search(self, T, P):
        """
        KMP search main algorithm: String -> String -> [Int]
        Return all the matching position of pattern string P in S
        """
        partial, ret, j = self.partial(P), [], 0

        print "partial table...", partial

        for i in range(len(T)):
            while j > 0 and T[i] != P[j]:
                j = partial[j - 1]

            if T[i] == P[j]:
                j += 1

            if j == len(P):
                ret.append(i - j + 1)
                j = 0

        return ret

print "kmp...", KMP().search('CABAAABAABB', 'AAABAAB')
