# Calculator Tool Documentation

## Overview
This project provides a complete arithmetic expression evaluator with CLI interface. It processes input strings through a tokenizer, parser, and evaluator to compute mathematical results. The system supports standard operator precedence and parentheses handling.

## Installation
```bash
pip install calc-tool
```

## Supported Operations
- Basic arithmetic: `+`, `-`, `*`, `/`
- Parentheses for grouping: `( )`
- Number literals: integers and floats

## Usage Examples
```bash
# Simple calculation
calc 3 + 5

# Parentheses support
calc (12 / (3 + 1)) * 2

# Error handling
calc 3 + 5 *
```

## Testing
Run end-to-end tests with:
```bash
python -m test_calc
```

The test suite verifies:
- Basic operations
- Precedence rules
- Error cases
- Edge value handling