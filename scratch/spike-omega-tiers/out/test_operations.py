import pytest
import tempfile
from models import Task
from operations import add_task, complete_task, list_tasks

def test_add_task():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmpfile:
        filepath = tmpfile.name
    task = add_task(filepath, "Test Task")
    assert task.title == "Test Task"
    assert not task.completed
    tasks = list_tasks(filepath)
    assert len(tasks) == 1
    assert tasks[0].title == "Test Task"
    assert not tasks[0].completed

def test_complete_task():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmpfile:
        filepath = tmpfile.name
    task = add_task(filepath, "Task to complete")
    completed_task = complete_task(filepath, task.id)
    assert completed_task.completed
    tasks = list_tasks(filepath)
    assert len(tasks) == 1
    assert tasks[0].completed

def test_list_tasks():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmpfile:
        filepath = tmpfile.name
    task1 = add_task(filepath, "Task 1")
    task2 = add_task(filepath, "Task 2")
    tasks = list_tasks(filepath)
    assert len(tasks) == 2
    assert tasks[0].title == "Task 1"
    assert tasks[1].title == "Task 2"