# todo-list Package Documentation

## Overview
The todo-list package provides a complete system for managing tasks through a combination of storage, operations, and command-line interface (CLI) tools. It supports task creation, completion, listing, and persistent storage.

## Model
The system uses a simple task model with the following operations:
- `add_task(description)`: Creates a new task with the given description. Returns the task ID.
- `complete_task(task_id)`: Marks an existing task as completed by ID.
- `list_tasks()`: Returns a list of all tasks, including their status and details.

## Storage
Tasks are persisted to disk using these utilities:
- `save_tasks(tasks, path)`: Writes a list of tasks to the specified file path.
- `load_tasks(path)`: Loads and returns tasks from the specified file path.

## CLI
The command-line interface is accessible via:
- `main()`: Entry point for the CLI, enabling users to interact with the todo list through commands.

## Testing
The package includes unit tests for all core operations:
- `test_add_task()`: Verifies task creation functionality.
- `test_complete_task()`: Validates task completion logic.
- `test_list_tasks()`: Ensures the task listing functionality works correctly.