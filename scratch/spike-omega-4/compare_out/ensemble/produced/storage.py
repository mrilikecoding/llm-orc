import json
from models import Task

def save_tasks(tasks, path):
    with open(path, 'w') as f:
        json.dump([task.__dict__ for task in tasks], f)

def load_tasks(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return [Task(**item) for item in data]