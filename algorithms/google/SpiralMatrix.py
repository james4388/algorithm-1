

# Print spiral matrix
# https://leetcode.com/problems/spiral-matrix/description/
def printLayer(matrix, layer, res):
    startx, endx = layer, len(matrix) - layer - 1
    starty, endy = layer, len(matrix[0]) - layer - 1

    i, j = startx, starty
    if startx == endx and starty == endy:
        res.append(matrix[startx][starty])
        return

    while j <= endy:
        res.append(matrix[i][j])
        j += 1

    i, j = startx + 1, endy
    while i <= endx:
        res.append(matrix[i][j])
        i += 1

    i, j = endx, endy - 1
    if startx != endx:
        while j >= starty:
            res.append(matrix[i][j])
            j -= 1

    i, j = endx - 1, starty
    if starty != endy:
        while i > startx:
            res.append(matrix[i][j])
            i -= 1


def spiralOrder(matrix):
    """
    :type matrix: List[List[int]]
    :rtype: List[int]
    """
    if not matrix:
        return []

    res = []
    l = min(len(matrix), len(matrix[0]))
    layers = l/2 + 1 if l % 2 else l/2
    for layer in range(layers):
        printLayer(matrix, layer, res)
    return res

print "spiral matrix....", spiralOrder([[6, 7, 9]])
