import json

# Load configuration from settings.json
with open('settings.json') as f:
    config = json.load(f)

# Directly access the three required configuration keys
retry_backoff = config['backoff_ms']
max_queue_depth = config['max_queue_depth']
affinity_salt = config['affinity_salt']

# Example usage: print the values (you can replace this with actual scheduling logic)
print(f"Retry backoff: {retry_backoff}ms")
print(f"Max queue depth: {max_queue_depth}")
print(f"Affinity salt: {affinity_salt}")
