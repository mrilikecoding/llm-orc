import argparse
import operations
import storage

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    add_parser = subparsers.add_parser('add')
    add_parser.add_argument('description', help='Task description')

    complete_parser = subparsers.add_parser('complete')
    complete_parser.add_argument('task_id', type=int, help='Task ID to complete')

    list_parser = subparsers.add_parser('list')

    args = parser.parse_args()

    tasks = storage.load_tasks('tasks.json')

    if args.command == 'add':
        operations.add_task(args.description)
        storage.save_tasks(tasks, 'tasks.json')
    elif args.command == 'complete':
        operations.complete_task(args.task_id)
        storage.save_tasks(tasks, 'tasks.json')
    elif args.command == 'list':
        operations.list_tasks()

if __name__ == '__main__':
    main()