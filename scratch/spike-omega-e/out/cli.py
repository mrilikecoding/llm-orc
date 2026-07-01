import argparse
import sys
from operations import add_task, complete_task, list_tasks

def main() -> None:
    parser = argparse.ArgumentParser(description='Task CLI')
    parser.add_argument('--db-path', required=True, help='Path to the database file')
    subparsers = parser.add_subparsers(dest='command')

    add_parser = subparsers.add_parser('add')
    add_parser.add_argument('--title', required=True, help='Task title')

    complete_parser = subparsers.add_parser('complete')
    complete_parser.add_argument('--task-id', type=int, required=True, help='Task ID to complete')

    list_parser = subparsers.add_parser('list')

    args = parser.parse_args()

    if args.command == 'add':
        title = args.title
        task = add_task(title, args.db_path)
        print(f"Added task {task.id}")
    elif args.command == 'complete':
        task_id = args.task_id
        complete_task(task_id, args.db_path)
        print("Task completed")
    elif args.command == 'list':
        tasks = list_tasks(args.db_path)
        for task in tasks:
            status = 'Completed' if task.completed else 'Pending'
            print(f"{task.id}: {task.title} - {status}")
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()