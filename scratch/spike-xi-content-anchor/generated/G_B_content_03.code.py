import json

def load_scheduler_config():
    with open('settings.json') as f:
        config = json.load(f)
    
    rbo_ms = config['rbo_ms']
    qdepth_max = config['qdepth_max']
    aff_salt = config['aff_salt']
    
    return rbo_ms, qdepth_max, aff_salt

if __name__ == '__main__':
    rbo, qdepth, salt = load_scheduler_config()
    print(f"Retry backoff: {rbo}ms, Max queue depth: {qdepth}, Affinity salt: {salt}")
