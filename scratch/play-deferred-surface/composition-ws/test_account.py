"""Tests for tinybank."""

from account import Account


def test_deposit() -> None:
    acct = Account(100)
    assert acct.deposit(50) == 150


def test_withdraw() -> None:
    acct = Account(100)
    assert acct.withdraw(30) == 70
