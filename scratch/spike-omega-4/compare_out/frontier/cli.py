import argparse
import operations
import storage


def main():
    parser = argparse.ArgumentParser(description="A simple todo-list CLI.")
    parser.add_argument("--file", default="tasks.json", help="Path to the tasks JSON file.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # add subcommand
    add_parser = subparsers.add_parser("add", help="Add a new task.")
    add_parser.add_argument("title", help="Title of the task to add.")

    # complete subcommand
    complete_parser = subparsers.add_parser("complete", help="Mark a task as done.")
    complete_parser.add_argument("task_id", type=int, help="ID of the task to complete.")

    # list subcommand
    subparsers.add_parser("list", help="List all tasks.")

    args = parser.parse_args()

    tasks = storage.load_tasks(args.file)

    if args.command == "add":
        new_task = operations.add_task(tasks, args.title)
        storage.save_tasks(tasks, args.file)
        print(f"Added task #{new_task.id}: {new_task.title}")

    elif args.command == "complete":
        task = operations.complete_task(tasks, args.task_id)
        if task:
            storage.save_tasks(tasks, args.file)
            print(f"Completed task #{task.id}: {task.title}")
        else:
            print(f"Task with id {args.task_id} not found.")

    elif args.command == "list":
        lines = operations.list_tasks(tasks)
        if lines:
            for line in lines:
                print(line)
        else:
            print("No tasks found.")


if __name__ == "__main__":
    main()
