#!/usr/bin/env python3
import unittest
import asyncio
import aiohttp


async def compute(x, y):
    print("compute %s + %s" % (x, y))
    await asyncio.sleep(1.0)
    return x + y


async def print_sum(x, y):
    result = await compute(x, y)
    print("%s + %s = %s" % (x, y, result))


async def fetch_page(url):
    with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            print(resp.status)
            print(await resp.text())


class TestAsync(unittest.TestCase):
    def test_async_loop():
        loop = asyncio.get_event_loop()
        loop.run_until_complete(print_sum(1, 2))
        loop.close()

    def test_aiohttp():
        loop = asyncio.get_event_loop()
        tasks = [fetch_page('http://google.com'),
                 fetch_page('http://cnn.com'),
                 fetch_page('http://twitter.com')]

        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()
        for task in tasks:
            print(task)


if __name__ == '__main__':
    unittest.main()
