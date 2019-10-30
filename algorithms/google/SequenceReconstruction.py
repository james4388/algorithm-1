

# Sequence [1, 2, 3, 4, 5], check if it can reconstruct from list of seq [[1, 2, 4], [1, 3, 5]]
# uniquely, return false in above case
# Solution: check if each pair order same with sequence order, check if each pair
# of sequence in list of sequence.
def sequenceReconstruction(org, seqs):
    indices = {val: idx for idx, val in enumerate(org)}
    edges = set()

    for seq in seqs:
        for i in range(1, len(seq)):
            prev, curr = seq[i-1], seq[i]
            if prev not in indices or indices[prev] > indices.get(curr):
                return False
            edges.add((prev, curr))

    for i in range(1, len(org)):
        if (org[i-1], org[i]) not in edges:
            return False
    return True
