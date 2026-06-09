import json

# Load settings from file
with open('settings.json') as f:
    config = json.load(f)

# Directly access the three required configuration keys
backoff_ms = config['backoff_ms']
max_queue_depth = config['max_queue_depth']
affinity_salt = config['affinity_salt']
