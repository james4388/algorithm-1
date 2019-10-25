

# https://leetcode.com/problems/car-fleet/
# Given list of car position and speed, count how many car fleet
# A car cannot pass another ahead of it but drive same speed to form
# a car fleet
# - Solution: sort car positions, calculate arrive time: target - pos / speed
# if lead car arrive sooner, not car will catch up and vice versa
def carFleet(target, position, speed):
    cur = ans = 0
    times = [float(target - pos)/sp for (pos, sp) in sorted(zip(position, speed))]

    for time in times[::-1]:
        if time > cur:
            ans += 1
            cur = time
    return ans
