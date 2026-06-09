import json

def configure_scheduler():
    with open('settings.json') as f:
        config = json.load(f)
    
    retry_backoff = config['retry_backoff_ms']
    max_queue_depth = config['max_queue_depth']
    affinity_salt = config['affinity_salt']
    
    # Example usage: print the values
    print(f"Retry backoff: {retry_backoff} ms")
    print(f"Max queue depth: {max_queue_depth}")
    print(f"Affinity salt: {affinity_salt}")
    
    # Your scheduler logic would use these values here
    # ...

if __name__ == "__main__":
    configure_scheduler()
