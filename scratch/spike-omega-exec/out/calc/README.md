# Calculator CLI Tool

This command-line calculator parses and evaluates mathematical expressions using a robust parser with proper operator precedence.

## Usage
Run the calculator with:
```
calc "3 + 5 * (2 - 4)"
```

## Supported Operators
- Addition: `+`
- Subtraction: `-`
- Multiplication: `*`
- Division: `/`
- Exponentiation: `**`

## Precedence Rules
1. Parentheses `()`
2. Exponentiation `**`
3. Multiplication `*` / Division `/`
4. Addition `+` / Subtraction `-`

## API Overview
The tool uses these core components:
- **tokenizer.py**: Converts strings to `Token` objects with types like `NUMBER` and `OPERATOR`
- **parser.py**: Builds expression trees using `Number` leaves and `BinOp` nodes for operations
- **evaluator.py**: Recursively evaluates the abstract syntax tree to produce a float result

## CLI Examples
```
$ calc "2 + 3 * 4"
14
```

```
$ calc "(5 - 3) ** 2"
4
```

```
$ calc "10 / 2 - 1"
4.0
```

## Testing
Run end-to-end tests with:
```
test_end_to_end
```