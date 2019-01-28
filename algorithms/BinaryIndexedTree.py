

# Get next child: get 2's compliment of index 'AND' with index, add to original
# Get parent: same as above, subtract to original
class BinaryIndexedTree(object):
    def __init__(self, n):
        self.bit = [0 for _ in range(n+1)]

    def insert(self, idx, val):
        idx += 1
        while idx < len(self.bit):
            self.bit[idx] += val
            idx += (idx & -idx)

    def search(self, idx):
        # idx += 1
        r = 0
        while idx > 0:
            r += self.bit[idx]
            idx -= idx & -idx
        return r

    @classmethod
    def construct(cls, arr):
        n = len(arr)
        bit = cls(n)
        for idx, num in enumerate(arr):
            bit.insert(idx, num)
        return bit

    def __repr__(self):
        return str(self.bit)


bit = BinaryIndexedTree.construct([1, 3, 2, 1, 3])
print("binary indexed tree...", bit)
print("sum range...", bit.search(4))
