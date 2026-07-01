from models import Task
from storage import load_tasks, save_tasks

def add_task(title: str, db_path: str) -> Task:
    tasks = load_tasks(db_path)
    new_id = len(tasks) + 1 if tasks else 1
    new_task = Task(id=new_id, title=title, completed=False)
    tasks.append(new_task)
    save_tasks(db_path, tasks)
    return new_task

def complete_task(task_id: int, db_path: str) -> None:
    tasks = load_tasks(db_path)
    for task in tasks:
        if task.id == task_id:
            task.completed = True
            break
    save_tasks(db_path, tasks)

def list_tasks(db_path: str) -> list[Task]:
    return load_tasks(db_path)