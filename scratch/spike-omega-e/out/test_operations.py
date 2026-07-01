import tempfile
import os
from operations import add_task, complete_task, list_tasks

def test_lifecycle() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, 'tasks.json')
        task = add_task("Test Task", db_path)
        tasks = list_tasks(db_path)
        assert len(tasks) == 1
        assert tasks[0].title == "Test Task"
        assert not tasks[0].completed
        complete_task(task.id, db_path)
        tasks = list_tasks(db_path)
        assert len(tasks) == 1
        assert tasks[0].title == "Test Task"
        assert tasks[0].completed
        assert os.path.exists(db_path)