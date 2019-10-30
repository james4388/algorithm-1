

# https://leetcode.com/problems/queue-reconstruction-by-height/
# People in queue, every person has height and k number with height higher or
# equal (h, k), reconstruct queue
# Solution: sort people by height decreaseing and order increasing
# insert each one into result by order
def reconstructQueue(people):
    ans = []
    if not people:
        return ans
    sortedPeople = sorted(people, key=lambda x: (-x[0], x[1]))
    for person in sortedPeople:
        ans.insert(person[1], person)
    return ans
