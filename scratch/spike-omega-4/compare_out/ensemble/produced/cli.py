import argparse
import operations
import storage

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    add_parser = subparsers.add_parser('add')
    add_parser.add_argument('title')

    complete_parser = subparsers.add_parser('complete')
    complete_parser.add_argument('task_id', type=int)

    list_parser = subparsers.add_parser('list')

    args = parser.parse_args()

    tasks = storage.load_tasks('tasks.json')

    if args.command == 'add':
        tasks = operations.add_task(tasks, args.title)
    elif args.command == 'complete':
        tasks = operations.complete_task(tasks, args.task_id)
    elif args.command == 'list':
        operations.list_tasks(tasks)

    if args.command in ['add', 'complete']:
        storage.save_tasks(tasks, 'tasks.json')

if __name__ == '__main__':
    main()