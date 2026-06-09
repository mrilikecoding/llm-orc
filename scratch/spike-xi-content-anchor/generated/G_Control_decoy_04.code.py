import json

def load_scheduler_settings():
    with open('settings.json') as f:
        config = json.load(f)
    backoff_ms = config['backoff_ms']
    max_queue_depth = config['max_queue_depth']
    affinity_salt = config['affinity_salt']
    return backoff_ms, max_queue_depth, affinity_salt

if __name__ == "__main__":
    backoff, max_depth, salt = load_scheduler_settings()
    print(f"Scheduler initialized with backoff: {backoff}ms, max depth: {max_depth}, salt: {salt}")
