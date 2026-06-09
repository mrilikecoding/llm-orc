import json

with open('settings.json') as f:
    config = json.load(f)

retry_backoff = config['rbo_ms']
max_queue_depth = config['qdepth_max']
affinity_salt = config['aff_salt']
