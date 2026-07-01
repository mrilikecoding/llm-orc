import models.Task
from storage import save_tasks, load_tasks

def add_task(description):
    tasks = load_tasks()
    new_id = max(task.id for task in tasks) + 1 if tasks else 1
    task = models.Task(new_id, description)
    tasks.append(task)
    save_tasks(tasks)
    return task

def complete_task(task_id):
    tasks = load_tasks()
    for task in tasks:
        if task.id == task_id:
            task.done = not task.done
            break
    save_tasks(tasks)

def list_tasks():
    return load_tasks()