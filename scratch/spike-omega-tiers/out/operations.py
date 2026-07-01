from models import Task
from storage import load_tasks, save_tasks

def add_task(filepath: str, title: str) -> Task:
    tasks = load_tasks(filepath)
    new_id = max(task.id for task in tasks) + 1 if tasks else 1
    new_task = Task(id=new_id, title=title, completed=False)
    tasks.append(new_task)
    save_tasks(filepath, tasks)
    return new_task

def complete_task(filepath: str, task_id: int) -> Task:
    tasks = load_tasks(filepath)
    for task in tasks:
        if task.id == task_id:
            task.completed = True
            break
    save_tasks(filepath, tasks)
    return next(task for task in tasks if task.id == task_id)

def list_tasks(filepath: str) -> list[Task]:
    return load_tasks(filepath)