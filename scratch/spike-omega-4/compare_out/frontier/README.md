# Todo-List Package

A simple command-line todo-list application built with Python.

## Modules

### `models.py`

Defines the core `Task` dataclass used throughout the package.

- **`Task`** — a dataclass with three fields:
  - `id` (int): unique identifier for the task
  - `title` (str): description of the task
  - `done` (bool): completion status (defaults to `False`)

Methods:
- `Task.to_dict()` — serialises a Task to a dictionary.
- `Task.from_dict(data)` — class method to reconstruct a Task from a dictionary.

### `storage.py`

Handles persistence of tasks to and from a JSON file. Imports `Task` from `models`.

Functions:
- **`save_tasks(tasks, path)`** — writes a list of `Task` objects to the given JSON file path.
- **`load_tasks(path)`** — reads a JSON file and returns a list of `Task` objects. Returns an empty list if the file is missing or corrupt.

### `operations.py`

Contains the business logic for manipulating tasks. Imports `Task` from `models` and uses `storage`.

Functions:
- **`add_task(tasks, title)`** — creates a new `Task` with an auto-incremented id, appends it to the list, and returns it.
- **`complete_task(tasks, task_id)`** — finds a task by id and marks it as done. Returns the task or `None` if not found.
- **`list_tasks(tasks)`** — returns a list of formatted strings representing each task (e.g. `"[ ] 1: Buy milk"`).

### `cli.py`

Provides a command-line interface via `argparse` with three subcommands. Imports `argparse` and `operations`.

Usage:

```bash
python cli.py add "Buy groceries"
python cli.py complete 1
python cli.py list
```

Subcommands:
- **`add <title>`** — calls `operations.add_task` and saves the result.
- **`complete <task_id>`** — calls `operations.complete_task` and saves the result.
- **`list`** — calls `operations.list_tasks` and prints each line.

Options:
- `--file <path>` — specify the tasks JSON file (default: `tasks.json`).

## Running Tests

```bash
python test_operations.py
```
