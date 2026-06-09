import json

def main():
    with open('settings.json') as f:
        config = json.load(f)
    
    # Directly access the three required config keys
    retry_backoff_ms = config['retry_backoff_ms']
    max_queue_depth = config['max_queue_depth']
    affinity_salt = config['affinity_salt_for_shard_routing']
    
    # Example usage (replace with actual scheduler logic)
    print(f"Retry backoff: {retry_backoff_ms} ms")
    print(f"Max queue depth: {max_queue_depth}")
    print(f"Affinity salt: {affinity_salt}")

if __name__ == '__main__':
    main()
