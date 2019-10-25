import time
import random


def RateLimited(maxPerSecond):
    minInterval = 1.0 / float(maxPerSecond)
    calls = {}

    def decorate(func):
        def rateLimitedFunction(*args, **kargs):
            elapsed = time.time() - calls.get(func.__name__, 0)
            leftToWait = minInterval - elapsed
            if leftToWait > 0:
                time.sleep(leftToWait)
            ret = func(*args, **kargs)
            calls[func.__name__] = time.time()
            return ret
        return rateLimitedFunction
    return decorate


@RateLimited(1)  # 2 per second at most
def print_number(num):
    print("n calling...", num)


@RateLimited(5)
def sleep_print(num):
    time.sleep(random.random())
    print("sleep random...", num)

for x in range(20):
    print_number(x)

for x in range(20):
    sleep_print(x)
