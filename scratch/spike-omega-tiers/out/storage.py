from models import Task
import json
from pathlib import Path

def load_tasks(filepath: str) -> list[Task]:
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [Task(**item) for item in data]

def save_tasks(filepath: str, tasks: list[Task]) -> None:
    with open(filepath, 'w') as f:
        json.dump([dict(task) for task in tasks], f)