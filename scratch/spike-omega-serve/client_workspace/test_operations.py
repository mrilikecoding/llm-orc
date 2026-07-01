import operations

def test_add_task():
    task = operations.add_task("Test Task")
    assert task.description == "Test Task"
    assert not task.done
    assert task.id == 1

def test_complete_task():
    task = operations.add_task("Task 1")
    task_id = task.id
    operations.complete_task(task_id)
    tasks = operations.list_tasks()
    completed_task = next(t for t in tasks if t.id == task_id)
    assert completed_task.done

def test_list_tasks():
    tasks = operations.list_tasks()
    assert len(tasks) == 0
    task1 = operations.add_task("Task 1")
    task2 = operations.add_task("Task 2")
    tasks = operations.list_tasks()
    assert len(tasks) == 2
    assert tasks[0].id == 1
    assert tasks[1].id == 2
    operations.complete_task(task1.id)
    tasks = operations.list_tasks()
    completed_task = next(t for t in tasks if t.id == task1.id)
    assert completed_task.done