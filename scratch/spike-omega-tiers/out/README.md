# Task Manager CLI Tool

## Installation
Install the package using pip:
```bash
pip install task-manager
```

## CLI Usage Examples
```bash
# Add a new task
task-manager add "Buy groceries"

# Mark a task as completed
task-manager complete 1

# List all tasks
task-manager list
```

## Architecture
The tool follows a layered architecture:

### Storage Module
storage.py provides functions to load and save tasks to a file:
- `load_tasks(filepath: str) -> list[Task]`: Reads tasks from a file
- `save_tasks(filepath: str, tasks: list[Task]) -> None`: Writes tasks to a file

These functions are used by the operations module to persist task data.

### Operations Module
operations.py handles task manipulation:
- `add_task(filepath: str, title: str) -> Task`: Creates and saves a new task
- `complete_task(filepath: str, task_id: int) -> Task`: Updates an existing task
- `list_tasks(filepath: str) -> list[Task]`: Retrieves all tasks

## Contribution Guidelines
1. Fork the repository and create a new branch
2. Implement features using the provided API
3. Write tests for your changes
4. Submit a pull request

Test functions available:
- `test_add_task()`
- `test_complete_task()`
- `test_list_tasks()`