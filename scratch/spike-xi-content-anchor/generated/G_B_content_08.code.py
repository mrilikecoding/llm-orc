import json

# Load configuration settings
with open('settings.json') as f:
    config = json.load(f)

# Directly access the three required configuration keys
retry_backoff_ms = config['rbo_ms']
max_queue_depth = config['qdepth_max']
affinity_salt = config['aff_salt']

# Example usage (scheduler logic would go here)
print(f"Scheduler initialized with retry backoff: {retry_backoff_ms}ms, "
      f"max queue depth: {max_queue_depth}, and affinity salt: {affinity_salt}")
