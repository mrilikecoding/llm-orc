import models
import operations

def test_add_task():
    tasks = []
    operations.add_task(tasks, "New Task")
    assert len(tasks) == 1
    task = tasks[0]
    assert task.title == "New Task"
    assert task.id == 1
    assert not task.completed

def test_complete_task():
    tasks = [models.Task(title="Task 1", id=1, completed=False)]
    operations.complete_task(tasks, 1)
    assert tasks[0].completed is True

def test_list_tasks():
    tasks = [
        models.Task(title="Task 1", id=1, completed=False),
        models.Task(title="Task 2", id=2, completed=True)
    ]
    result = operations.list_tasks(tasks)
    assert len(result) == 2
    assert result[0].title == "Task 1"
    assert result[0].id == 1
    assert not result[0].completed
    assert result[1].title == "Task 2"
    assert result[1].id == 2
    assert result[1].completed is True