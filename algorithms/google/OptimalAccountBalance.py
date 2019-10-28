from collections import defaultdict


# Give list of transaction [0, 1, 10], user 0 give user 1 amount 10
# Find minimum number of transfer
# Solution: Find all the final amount after all transactions for each users
# Use backtracking: transfer amount from current user to another if they have
# negative and positive amount e.g 10, -8, transfer 10 to -8 => 0, 2
class OptimalAccountBalance(object):
    def minTransfer(self, transactions):
        accounts = defaultdict(int)
        for (sender, receiver, amount) in transactions:
            accounts[sender] -= amount
            accounts[receiver] += amount

        ans = [len(accounts)]
        print("accounts...", accounts)

        def helper(user, count):
            while user < len(accounts) and accounts[user] == 0:
                user += 1

            if user == len(accounts):
                ans[0] = min(ans[0], count)
                return

            for other in range(user + 1, len(accounts)):
                if accounts[user] * accounts[other] < 0:
                    accounts[other] += accounts[user]
                    helper(user + 1, count + 1)
                    accounts[other] -= accounts[user]

        helper(0, 0)
        return ans[0]


if __name__ == '__main__':
    account = OptimalAccountBalance()
    ans = account.minTransfer([[0, 1, 10], [1, 0, 1], [1, 2, 5], [2, 0, 5]])
    print("min transfer...", ans)
