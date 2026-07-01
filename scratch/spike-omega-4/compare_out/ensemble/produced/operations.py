import models
import storage

def add_task(tasks, title):
    if not tasks:
        new_id = 1
    else:
        new_id = tasks[-1].id + 1
    task = models.Task(id=new_id, title=title, done=False)
    tasks.append(task)
    return tasks

def complete_task(tasks, task_id):
    for task in tasks:
        if task.id == task_id:
            task.done = not task.done
            break
    return tasks

def list_tasks(tasks):
    return [task for task in tasks if not task.done]