

# Implementation of rolling hash and search algorithm Rabin-Karp
class Search(object):
    def __init__(self, base=None, prime=None):
        self.base = base or 256
        self.prime = prime or 10**9 + 7

    def hashcode(self, bytelist):
        code = 0
        for byte in bytelist:
            code = code % self.prime * self.base + byte
        return code % self.prime

    def rollingHash(self, code, oldchar, newchar):
        val = (code + self.prime - oldchar * self.base * self.base % self.prime) * self.base + newchar
        return val % self.prime

    def search(self, arr, bs):
        m, n = len(arr), len(bs)
        code = self.hashcode(arr)
        ncode = self.hashcode(bs[:m])
        if ncode == code:
            return 0

        for i in range(m, n):
            ncode = self.rollingHash(ncode, bs[i-m], bs[i])
            if ncode == code:
                return i - m + 1


if __name__ == '__main__':
    s = Search(256, 101)
    print "hashing with base %s and prime %s..." % (256, 101)
    code = s.hashcode([97, 98, 114])
    print "hash code...", code
    ncode = s.rollingHash(code, 97, 97)
    ncompare = s.hashcode([98, 114, 97])
    print "new hash code...", ncode, ncompare
    print "Searching byte array in file...", s.search([1, 2, 3], [3, 2, 1, 2, 3, 3])
