from collections import defaultdict, deque


# https://leetcode.com/problems/cheapest-flights-within-k-stops/
# Given n city, find cheapest price from city a to b within k stop
# Use BFS and store visited city with price to prune searching
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int,
                          dst: int, K: int) -> int:
        flight_map = defaultdict(list)

        for flight in flights:
            flight_map[flight[0]].append(flight[1:])

        # current city, price, num of stop
        queue = [(src, 0, 0)]
        # city already visited from src, price
        visited = {src: 0}

        # current price
        min_price = float('inf')

        while queue:
            new_queue = []
            for city, price, stop in queue:
                if city == dst and stop <= K + 1:
                    min_price = min(min_price, price)
                    continue

                if stop > K:
                    continue

                for next_city, next_price in flight_map.get(city, []):
                    # if already visited this city with less than price,
                    # don't need to visit again
                    if next_city in visited and
                    (visited[next_city] <= price + next_price):
                        continue

                    new_queue.append((next_city, price + next_price, stop + 1))
                    visited[next_city] = price + next_price
            queue = new_queue

        return min_price if min_price < float('inf') else -1
