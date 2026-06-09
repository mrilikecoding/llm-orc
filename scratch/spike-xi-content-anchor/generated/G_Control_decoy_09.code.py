import json

# Load settings from the JSON file
with open('settings.json') as f:
    config = json.load(f)

# Directly access the three required configuration keys
backoff_ms = config['backoff_ms']
max_queue_depth = config['max_queue_depth']
affinity_salt = config['affinity_salt']

# Example usage (you can replace this with your scheduler logic)
print(f"Retry backoff: {backoff_ms}ms")
print(f"Max queue depth: {max_queue_depth}")
print(f"Affinity salt: {affinity_salt}")
