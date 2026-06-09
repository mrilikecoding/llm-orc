import json

def initialize_scheduler():
    # Load settings from file
    with open('settings.json') as config_file:
        config = json.load(config_file)
    
    # Directly access the three required settings by their exact keys
    retry_backoff = config['retry_backoff_ms']
    max_queue_depth = config['max_queue_depth']
    affinity_salt = config['affinity_salt']
    
    # Example usage (you'd replace this with actual scheduling logic)
    print(f"Scheduler initialized with:")
    print(f"Retry backoff: {retry_backoff} ms")
    print(f"Max queue depth: {max_queue_depth}")
    print(f"Affinity salt: {affinity_salt}")
