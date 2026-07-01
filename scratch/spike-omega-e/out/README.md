# todo-list Package

The todo-list package provides a CLI-based task management system with persistent storage. It features a modular architecture separating data models, storage operations, and business logic.

## Usage Guide

### CLI Commands
Run the command-line interface with:
```bash
python -m todo_list
```

Available commands:
- `add <title>`: Create a new task
  ```bash
  add "Buy groceries"
  ```
  Returns a task ID for reference

- `complete <id>`: Mark task as completed
  ```bash
  complete 1
  ```

- `list`: Show all tasks with their status
  ```bash
  list
  ```

### Programmatic Usage
Use the package's functions directly:
```python
from operations import add_task, complete_task, list_tasks
from storage import save_tasks

tasks = add_task("Write documentation", "tasks.json")
complete_task(tasks[0].id, "tasks.json")
tasks = list_tasks("tasks.json")
```

## Architecture Overview

The package follows a layered architecture:

1. **Models (models.py)**
   - Defines the `Task` data structure
   - `Task(id: int, title: str, completed: bool)`

2. **Storage (storage.py)**
   - Manages file persistence
   - `load_tasks(path: str) -> list[Task]`
   - `save_tasks(path: str, tasks: list[Task]) -> None`

3. **Operations (operations.py)**
   - Business logic layer
   - `add_task(title: str, db_path: str) -> Task`
   - `complete_task(task_id: int, db_path: str) -> None`
   - `list_tasks(db_path: str) -> list[Task]`

4. **CLI (cli.py)**
   - Command-line interface entry point
   - `main() -> None`

## Testing
The `test_lifecycle()` function in `test_operations.py` verifies the complete workflow:
1. Task creation
2. Persistence to file
3. Task completion
4. Data retrieval

This ensures all components work together correctly.