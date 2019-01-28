import math
import sys


def roundPricesToMatchTarget(prices, target):
    n = len(prices)
    q = ((math.ceil(num) - num, idx) for (idx, num) in enumerate(prices))
    q = sorted(q, key=lambda x: x[0])
    out = [None for i in range(n)]

    i, j = 0, n-1
    er = target - sum(prices)
    while i <= j:
        if er >= 0:
            e, idx = q[i]
            val = int(math.ceil(prices[idx]))
            er -= e
            out[idx] = val
            i += 1
        else:
            e, idx = q[j]
            val = int(math.floor(prices[idx]))
            er += 1 - e
            out[idx] = val
            j -= 1
    return out


class Solution(object):
    def _round(self, prices, target, idx, out, er):
        if target < 0:
            return

        if idx >= len(prices):
            if target == 0 and (er < self.er):
                self.er = er
                self.res = out
            return
        num = prices[idx]
        self._round(prices, target - math.ceil(num), idx + 1,
                    out + [math.ceil(num)], er + math.ceil(num) - num)
        self._round(prices, target - math.floor(num), idx + 1,
                    out + [math.floor(num)], er + num - math.floor(num))

    def roundPricesToMatchTarget(self, prices, target):
        if not prices:
            return []

        self.res = []
        self.er = sys.maxint
        self._round(prices, target, 0, [], 0)
        return self.er, self.res


# print roundPricesToMatchTarget([0.7, 2.8, 4.9], 8)

class CSVParser(object):
    def __init__(self):
        self.format = "{first_name}, {age} years old, is from {city} and is interested in {interests}."
        self.header = "first_name,last_name,email,interests,notes,city,age"

    def parse(self, line):
        v = []
        buff = ''
        n = len(line)
        i = 0
        hasQuote = False
        while i < n:
            if i < n-1 and line[i] == line[i+1] == '"':
                i += 2
                l = '"'
                while i < n-1:
                    if line[i] == line[i+1] == '"':
                        l += '"'
                        i += 1
                        break
                    l += line[i]
                    i += 1
                buff += l
            elif line[i] == '"':
                if not hasQuote:
                    hasQuote = True
                else:
                    hasQuote = False
            elif line[i] == ',' and not hasQuote:
                v.append(buff)
                buff = ''
            else:
                buff += line[i]
            i += 1
        if buff:
            v.append(buff)
        return v

parser = CSVParser()
v = parser.parse('Weronika,Zaborska,njkfdsv@dsgfk.sn,"running, sci-fi",new,Krakow,25')
print(len(v), v)
