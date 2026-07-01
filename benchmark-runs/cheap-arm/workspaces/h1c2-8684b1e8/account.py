class BankAccount:
    def __init__(self, owner: str, balance: float = 0.0) -> None:
        """Initialize a bank account with owner and balance."""
        self.owner = owner
        self._balance = balance

    def deposit(self, amount: float) -> float:
        """Add amount to balance and return new balance."""
        self._balance += amount
        return self._balance

    def withdraw(self, amount: float) -> float:
        """Subtract amount from balance and return new balance."""
        self._balance -= amount
        return self._balance

    @property
    def balance(self) -> float:
        """Return the current balance."""
        return self._balance