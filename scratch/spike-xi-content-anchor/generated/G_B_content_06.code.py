import json

# Load the settings file
with open('settings.json') as f:
    config = json.load(f)

# Directly access the three required configuration keys
retry_backoff_ms = config['rbo_ms']
max_queue_depth = config['qdepth_max']
affinity_salt = config['affinity_salt']
