import argparse
from operations import add_task, complete_task, list_tasks

def main():
    parser = argparse.ArgumentParser(description='Task management CLI')
    parser.add_argument('--filepath', required=True, help='Path to the task data file')
    subparsers = parser.add_subparsers(dest='command')

    add_parser = subparsers.add_parser('add', help='Add a new task')
    add_parser.add_argument('title', help='Title of the task')

    complete_parser = subparsers.add_parser('complete', help='Mark a task as complete')
    complete_parser.add_argument('task_id', type=int, help='ID of the task to complete')

    list_parser = subparsers.add_parser('list', help='List all tasks')

    args = parser.parse_args()

    if args.command == 'add':
        task = add_task(args.filepath, args.title)
        print(f"Added task: {task.title}")
    elif args.command == 'complete':
        task = complete_task(args.filepath, args.task_id)
        print(f"Completed task {task.id}")
    elif args.command == 'list':
        tasks = list_tasks(args.filepath)
        for task in tasks:
            status = 'Completed' if task.completed else 'Pending'
            print(f"{task.id}: {task.title} - {status}")
    else:
        parser.print_help()