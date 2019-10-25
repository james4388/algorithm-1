

# HARD: race car
# https://leetcode.com/problems/race-car/
# A: accelerate speed, pos += speed, speed *= 2
# R: reverse, if positive speed = -1, negative speed = 1
# Give target find short list of commands
# https://leetcode.com/problems/race-car/
# Use BFS to store location and current speed, runtime: 2^n
# There's overlap: use hashset to store
def racecar(target):
    if not target:
        return 0

    q = [(0, 1)]
    visited = {(0, 1)}
    lvl = 0

    while q:
        nq = []
        print("current queue...", q)
        for item in q:
            pos, speed = item
            if pos == target:
                return lvl

            acc = (pos + speed, speed*2)
            rev = (pos, -1 if speed > 0 else 1)

            if acc not in visited and acc[0] > 0:
                nq.append(acc)
                visited.add(acc)

            if rev not in visited and rev[0] > 0:
                nq.append(rev)
                visited.add(rev)
        q = nq
        lvl += 1
    return -1

print("race car...", racecar(5))
