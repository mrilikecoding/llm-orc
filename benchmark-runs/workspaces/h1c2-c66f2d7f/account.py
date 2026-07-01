class BankAccount:
    def __init__(self, owner: str, balance: float = 0.0):
        """Initialize a bank account with the given owner and balance."""
        self.owner = owner
        self._balance = balance

    def deposit(self, amount: float) -> float:
        """Add the specified amount to the account balance."""
        self._balance += amount
        return self._balance

    def withdraw(self, amount: float) -> float:
        """Subtract the specified amount from the account balance."""
        self._balance -= amount
        return self._balance

    @property
    def balance(self) -> float:
        """Return the current account balance."""
        return self._balance