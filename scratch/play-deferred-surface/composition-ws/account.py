"""tinybank - a minimal in-memory bank account."""


class Account:
    """A single account holding a balance."""

    def __init__(self, balance: float = 0.0) -> None:
        self.balance = balance

    def deposit(self, amount: float) -> float:
        """Add amount to the balance and return the new balance."""
        self.balance += amount
        return self.balance

    def withdraw(self, amount: float) -> float:
        """Subtract amount from the balance and return the new balance."""
        self.balance -= amount
        return self.balance


def apply_interest(balance: float, rate: float) -> float:
    """Return the balance after applying an interest rate.

    Applying a rate of 0.05 to a balance of 100 should yield 105.0.
    """
    return balance * rate
