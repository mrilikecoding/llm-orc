import json

# Load configuration from settings.json
with open('settings.json') as config_file:
    config = json.load(config_file)

# Directly access the required configuration keys
backoff_ms = config['backoff_ms']
max_queue_depth = config['max_queue_depth']
affinity_salt = config['affinity_salt']

# Example usage (you can replace this with your scheduler logic)
print(f"Using backoff: {backoff_ms}ms, max queue depth: {max_queue_depth}, affinity salt: {affinity_salt}")
