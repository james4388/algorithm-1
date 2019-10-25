

# Candy rush
# https://leetcode.com/problems/candy-crush
# each cell preresent a type of candy, if 3 cells have same candy
# (horizontal, vertical) set them to 0, if there's candy above empty cell, move
# them until reach the end
# Use set to store index of row, column rushed
def candyRush(board):
    # Find rushes in horizontal and vertial
    n, m = len(board), len(board[0])
    turn = n*m//3
    rush = set()

    while turn >= 0:
        for i in range(n):
            for j in range(m):
                if board[i][j] == 0:
                    continue

                v = board[i][j]
                # Vertical rush
                if j + 2 < m and v == board[i][j+1] == board[i][j+2]:
                    x, y = i, j
                    while y < m and board[x][y] == v:
                        rush.add((x, y))
                        y += 1
                # Horizontal rush
                if i + 2 < n and v == board[i+1][j] == board[i+2][j]:
                    x, y = i, j
                    while x < n and board[x][y] == v:
                        rush.add((x, y))
                        x += 1
        if not rush:
            break

        for x, y in rush:
            board[x][y] = 0

        # Move non-empty cell down by swaping it with empty cell
        for j in range(m):
            count = 0
            for i in range(n-1, -1, -1):
                if board[i][j] == 0:
                    count += 1
                elif count > 0:
                    board[i][j], board[i+count][j] = board[i+count][j], board[i][j]
        turn -= 1

    return board
