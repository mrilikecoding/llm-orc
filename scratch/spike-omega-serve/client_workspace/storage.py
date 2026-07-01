import json
from models import Task

def save_tasks(tasks, path):
    with open(path, 'w') as f:
        json.dump([vars(task) for task in tasks], f)

def load_tasks(path):
    with open(path, 'r') as f:
        tasks_dict = json.load(f)
    return [Task(**task) for task in tasks_dict]