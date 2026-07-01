from models import Task
import storage


def add_task(tasks: list, title: str) -> Task:
    """Add a new task with a unique id and return it."""
    task_id = max((t.id for t in tasks), default=0) + 1
    new_task = Task(id=task_id, title=title, done=False)
    tasks.append(new_task)
    return new_task


def complete_task(tasks: list, task_id: int) -> Task | None:
    """Mark a task as done by id. Returns the task or None if not found."""
    for task in tasks:
        if task.id == task_id:
            task.done = True
            return task
    return None


def list_tasks(tasks: list) -> list:
    """Return a list of string representations of all tasks."""
    results = []
    for task in tasks:
        status = "[x]" if task.done else "[ ]"
        results.append(f"{status} {task.id}: {task.title}")
    return results
