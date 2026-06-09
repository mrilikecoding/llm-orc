import json

# Load configuration directly from settings.json
with open('settings.json') as f:
    config = json.load(f)

# Access each required setting by its exact config key
retry_backoff_ms = config['rbo_ms']
max_queue_depth = config['qdepth_max']
shard_affinity_salt = config['aff_salt']
