"""JSON-backed todo storage."""

from __future__ import annotations

import json
from pathlib import Path


class TodoStore:
    """Persist todos as a list of dicts in a JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def _load(self) -> list[dict]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, todos: list[dict]) -> None:
        self.path.write_text(json.dumps(todos, indent=2), encoding="utf-8")

    def add(self, text: str) -> dict:
        todos = self._load()
        todo = {"id": len(todos) + 1, "text": text, "done": False}
        todos.append(todo)
        self._save(todos)
        return todo

    def list(self) -> list[dict]:
        return self._load()

    def complete(self, todo_id: int) -> dict:
        todos = self._load()
        for todo in todos:
            if todo["id"] == todo_id:
                todo["done"] = True
                self._save(todos)
                return todo
        raise KeyError(todo_id)

    def remove(self, todo_id: int) -> None:
        todos = self._load()
        for i, todo in enumerate(todos):
            if todo["id"] == todo_id:
                del todos[i]
                self._save(todos)
                return
        raise KeyError(todo_id)