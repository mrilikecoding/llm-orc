import json

with open('settings.json') as f:
    config = json.load(f)

rbo_ms = config['rbo_ms']
qdepth_max = config['qdepth_max']
aff_salt = config['aff_salt']
