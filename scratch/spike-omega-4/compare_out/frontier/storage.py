import json
from models import Task


def save_tasks(tasks: list, path: str) -> None:
    """Persist a list of Task objects to a JSON file."""
    data = [task.to_dict() for task in tasks]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_tasks(path: str) -> list:
    """Load a list of Task objects from a JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    return [Task.from_dict(item) for item in data]
