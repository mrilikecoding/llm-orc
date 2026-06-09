import json

# Load settings from the JSON file
with open('settings.json') as f:
    config = json.load(f)

# Directly access the three required configuration keys
backoff_ms = config['backoff_ms']
max_queue_depth = config['max_queue_depth']
affinity_salt = config['affinity_salt']

# Example usage: print the values to verify they're loaded correctly
print(f"Retry backoff: {backoff_ms} ms")
print(f"Maximum queue depth: {max_queue_depth}")
print(f"Affinity salt: {affinity_salt}")
