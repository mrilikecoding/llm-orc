import json

# Load settings from the provided JSON file
with open('settings.json') as f:
    config = json.load(f)

# Directly access the three required configuration keys
retry_backoff_ms = config['rbo_ms']
max_queue_depth = config['qdepth_max']
affinity_salt = config['aff_salt']

# Example usage: print the values to confirm they're loaded
print(f"Retry backoff: {retry_backoff_ms} ms")
print(f"Max queue depth: {max_queue_depth}")
print(f"Affinity salt: {affinity_salt}")
