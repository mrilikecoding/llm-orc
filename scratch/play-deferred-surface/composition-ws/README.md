# tinybank

A minimal in-memory bank account library.

## Usage

```python
from account import Account

acct = Account(100)
acct.deposit(50)   # 150
acct.withdraw(30)  # 130
```

## API

- `Account(balance=0)` - create an account.
- `Account.deposit(amount)` - add to the balance.
- `Account.withdraw(amount)` - subtract from the balance.
- `Account.transfer(other, amount)` - move money to another account.

## Interest

Use `apply_interest(balance, rate)` to apply an interest rate to a balance.
For example, `apply_interest(100, 0.05)` returns `105.0`.
