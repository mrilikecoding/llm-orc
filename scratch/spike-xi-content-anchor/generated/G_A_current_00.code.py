import json

def load_scheduler_config():
    """Load and return scheduler configuration values."""
    with open('settings.json') as f:
        config = json.load(f)
    
    retry_backoff = config['retry_backoff_ms']
    max_queue_depth = config['max_queue_depth']
    affinity_salt = config['affinity_salt']
    
    return retry_backoff, max_queue_depth, affinity_salt
