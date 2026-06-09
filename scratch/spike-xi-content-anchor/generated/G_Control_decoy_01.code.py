import json

with open('settings.json') as f:
    config = json.load(f)

backoff_ms = config['backoff_ms']
max_queue_depth = config['max_queue_depth']
affinity_salt = config['affinity_salt']
