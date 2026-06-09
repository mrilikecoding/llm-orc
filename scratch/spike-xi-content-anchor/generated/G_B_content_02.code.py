import json

# Load the configuration file
with open('settings.json') as config_file:
    config = json.load(config_file)

# Directly access the required settings using exact keys
retry_backoff_ms = config['rbo_ms']
max_queue_depth = config['qdepth_max']
affinity_salt = config['aff_salt']

# Example usage (you can expand this with scheduling logic)
print(f"Retry backoff: {retry_backoff_ms} ms")
print(f"Max queue depth: {max_queue_depth}")
print(f"Affinity salt: {affinity_salt}")
