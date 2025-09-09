#!/usr/bin/env python3
"""
analyze_text.py - Analyze text and return structured metrics

Usage:
  As script agent in ensemble:
    script: scripts/analyze_text.py
    
  From command line:
    echo '{"text": "hello world"}' | python analyze_text.py
"""
import json
import sys
from collections import Counter

def main():
    # Read input from stdin
    if not sys.stdin.isatty():
        input_data = sys.stdin.read()
        try:
            data = json.loads(input_data)
            # Handle enhanced script agent format
            if isinstance(data, dict) and 'parameters' in data:
                text = data['parameters'].get('text', data.get('input', ''))
            elif isinstance(data, dict) and 'text' in data:
                text = data.get('text', '')
            else:
                # If not structured JSON, treat as plain text
                text = input_data if isinstance(input_data, str) else ''
        except json.JSONDecodeError:
            # If not JSON, treat as plain text
            text = input_data
    else:
        text = ''
    
    # Analyze text
    words = text.lower().split()
    word_freq = Counter(words)
    
    result = {
        "success": True,
        "metrics": {
            "character_count": len(text),
            "word_count": len(words),
            "unique_words": len(set(words)),
            "most_common": word_freq.most_common(5),
            "average_word_length": round(sum(len(w) for w in words) / len(words), 2) if words else 0
        },
        "text_sample": text[:100] + "..." if len(text) > 100 else text
    }
    
    # Output JSON
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()