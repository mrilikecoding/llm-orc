import operations
from models import Task


def test_add_task():
    tasks = []
    task = operations.add_task(tasks, "Buy groceries")
    assert task.id == 1
    assert task.title == "Buy groceries"
    assert task.done is False
    assert len(tasks) == 1


def test_complete_task():
    tasks = [Task(id=1, title="Test task", done=False)]
    result = operations.complete_task(tasks, 1)
    assert result is not None
    assert result.done is True
    assert tasks[0].done is True


def test_complete_task_not_found():
    tasks = []
    result = operations.complete_task(tasks, 99)
    assert result is None


def test_list_tasks():
    tasks = [Task(id=1, title="Task A", done=False), Task(id=2, title="Task B", done=True)]
    result = operations.list_tasks(tasks)
    assert len(result) == 2
    assert "[ ] 1: Task A" in result[0]
    assert "[x] 2: Task B" in result[1]


def test_add_task_incrementing_ids():
    tasks = [Task(id=5, title="Existing", done=False)]
    task = operations.add_task(tasks, "New task")
    assert task.id == 6


if __name__ == "__main__":
    test_add_task()
    test_complete_task()
    test_complete_task_not_found()
    test_list_tasks()
    test_add_task_incrementing_ids()
    print("All tests passed.")
