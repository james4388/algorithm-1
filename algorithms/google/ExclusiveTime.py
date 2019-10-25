

# https://leetcode.com/problems/exclusive-time-of-functions/
# Exclusive time, single threaded cpu has n function run, return total runtime
# for each function
# Solution: use stack to process log, if start time add delta = time - prevtime
# if end time pop from start and delta = time - prev_time + 1
def exclusiveTime(n, logs):
    ans = [0 for i in range(n)]
    stack = []
    prev = 0

    for log in logs:
        fun, _type, time = log.split(':')
        fun, time = int(fun), int(time)

        if _type == 'start':
            if stack:
                ans[stack[-1]] += time - prev
            stack.append(fun)
            prev = time
        else:
            ans[stack.pop()] += time - prev + 1
            prev = time + 1
    return ans
