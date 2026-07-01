import json
from pathlib import Path
from models import Task

def load_tasks(path: str) -> list[Task]:
    path_obj = Path(path)
    if not path_obj.exists():
        return []
    with path_obj.open('r') as f:
        data = json.load(f)
    tasks = [Task(task['id'], task['title'], task['completed']) for task in data]
    return tasks

def save_tasks(path: str, tasks: list[Task]) -> None:
    path_obj = Path(path)
    data = [vars(task) for task in tasks]
    with path_obj.open('w') as f:
        json.dump(data, f, indent=4)