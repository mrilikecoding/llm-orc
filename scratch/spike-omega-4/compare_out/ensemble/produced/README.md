# todo-list Package Structure

## Models
`models.py` defines the `Task` dataclass representing a to-do item with:
- `title`: str
- `completed`: bool
- `id`: int (auto-generated)

## Storage
`storage.py` provides JSON I/O functions:
- `save_tasks(tasks, path)`: Serializes task list to JSON file
- `load_tasks(path)`: Deserializes JSON file to task list

## Operations
`operations.py` implements core task management:
- `add_task(tasks, title)`: Creates new task with auto-increment ID
- `complete_task(tasks, task_id)`: Marks task as completed by ID
- `list_tasks(tasks)`: Returns formatted string of all tasks

## CLI
`cli.py` provides argparse-based command line interface:
- `main()`: Parses commands for add, complete, list, and exit actions

## Tests
`test_operations.py` contains unit tests:
- `test_add_task()`: Verifies new tasks are added with correct ID
- `test_complete_task()`: Ensures task completion state is toggled
- `test_list_tasks()`: Checks output formatting for task lists