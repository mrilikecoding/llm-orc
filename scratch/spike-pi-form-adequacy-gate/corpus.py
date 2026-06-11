"""Spike π — labeled failure corpus (C1-C9).

ground_truth:
  REFUSE   - a deterministically-detectable failure a viable gate MUST catch
  PASS     - a correct deliverable no gate may refuse (false-positive control)
  RESIDUAL - parses-but-semantically-wrong; a deterministic gate correctly
             PASSES it (the fault needs the project function-graph / runtime
             behavior to judge). The unification claim is that B's miss-set =
             exactly these.

Provenance: real σ/η bytes where marked (real); category-faithful synthesized
otherwise. C4 is the one deliberate transformation (real η JS bytes placed at a
.py destination — P2-B).
"""

from __future__ import annotations

from dataclasses import dataclass

REFUSE = "REFUSE"
PASS = "PASS"
RESIDUAL = "RESIDUAL"


@dataclass(frozen=True)
class Item:
    cid: str
    category: str
    seam: str
    destination_path: str
    ground_truth: str
    provenance: str
    content: str


# --- C1: trailing prose after valid code (form bleed; σ Run B shape, real) ---
C1 = Item(
    "C1", "trailing-prose", "form", "cli.py", REFUSE, "σ Run B (real shape)",
    '''import argparse


def main():
    parser = argparse.ArgumentParser(description="Convert temperatures.")
    parser.add_argument("value", type=float)
    args = parser.parse_args()
    print(args.value)


if __name__ == "__main__":
    main()

Here's the CLI implementation. This script reads a temperature value from the
command line and prints it back. You can extend it to convert between scales.
''',
)

# --- C2: fenced code block (form bleed; χ-era default habit) ------------------
C2 = Item(
    "C2", "fenced-block", "form", "util.py", REFUSE, "χ default habit",
    '''```python
def add(a, b):
    return a + b
```''',
)

# --- C3: leading prose preamble (form bleed) ---------------------------------
C3 = Item(
    "C3", "leading-prose", "form", "helper.py", REFUSE, "synthesized",
    '''Here is the helper module you requested:

def helper(value):
    return value * 2
''',
)

# --- C4: wrong-language in a .py path (adequacy det. slice; η run 2, re-pathed)
C4 = Item(
    "C4", "wrong-language", "adequacy-det", "cli.py", REFUSE,
    "η run 2 JS bytes @ .py (P2-B transform)",
    '''const args = process.argv.slice(2);

function convertTemperature(celsius) {
  return (celsius * 9) / 5 + 32;
}

console.log(convertTemperature(Number(args[0])));
''',
)

# --- C5: valid-language coder syntax bug (adequacy det. slice; σ Run A, real) -
C5 = Item(
    "C5", "syntax-bug", "adequacy-det", "cli.py", REFUSE, "σ Run A args.from (real)",
    '''import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--from")
args = parser.parse_args()
print(args.from)
''',
)

# --- C6a: parses-but-wrong, OBVIOUS (residual; η nonexistent API, real) -------
C6a = Item(
    "C6a", "semantic-obvious", "adequacy-semantic", "cli.py", RESIDUAL,
    "η cli.py nonexistent API (real)",
    '''import sys

import converters

print(converters.convert_temperature(float(sys.argv[1])))
''',
)

# --- C6b: parses-but-wrong, PLAUSIBLE (residual; wrong signature) -------------
C6b = Item(
    "C6b", "semantic-plausible", "adequacy-semantic", "cli.py", RESIDUAL,
    "synthesized (wrong signature)",
    '''import sys

import converters

# real function name, wrong argument shape (passes a string + a stray arg)
print(converters.celsius_to_fahrenheit(sys.argv[1], "fahrenheit"))
''',
)

# --- C6c: parses-but-wrong, NEAR-MISS (residual; wrong constant) -------------
C6c = Item(
    "C6c", "semantic-nearmiss", "adequacy-semantic", "converters.py", RESIDUAL,
    "synthesized (wrong formula constant)",
    '''def celsius_to_fahrenheit(c):
    return c * 9 / 5 + 30


def fahrenheit_to_celsius(f):
    return (f - 30) * 5 / 9
''',
)

# --- C7: correct bare .py (control; σ/η converged file shape, real) ----------
C7 = Item(
    "C7", "correct-bare", "control", "converters.py", PASS, "σ/η converged (real shape)",
    '''def celsius_to_fahrenheit(c):
    return c * 9 / 5 + 32


def fahrenheit_to_celsius(f):
    return (f - 32) * 5 / 9


def celsius_to_kelvin(c):
    return c + 273.15
''',
)

# --- C7b: correct .py with module docstring + NL comments (Arm-D stressor) ----
C7b = Item(
    "C7b", "correct-documented", "control", "converters.py", PASS,
    "synthesized to category (P2-A/P3-A)",
    '''"""Temperature conversion helpers.

This module provides functions to convert between Celsius, Fahrenheit, and
Kelvin. Note that all functions operate on floats and you can chain them.
"""


def celsius_to_fahrenheit(c):
    # You can pass either an int or a float here.
    return c * 9 / 5 + 32


def fahrenheit_to_celsius(f):
    return (f - 32) * 5 / 9
''',
)

# --- C8: correct .md with a legitimate fenced example (control) --------------
C8 = Item(
    "C8", "correct-markdown", "control", "README.md", PASS, "σ/η README shape (real)",
    '''# Temperature Converter

A small command-line tool for converting temperatures between scales.

## Usage

```bash
python cli.py 100 --from celsius --to fahrenheit
```

The converter supports Celsius, Fahrenheit, and Kelvin.
''',
)

# --- C9: correct .json (control) ---------------------------------------------
C9 = Item(
    "C9", "correct-json", "control", "config.json", PASS, "synthesized valid JSON",
    '''{
  "scales": ["celsius", "fahrenheit", "kelvin"],
  "default": "celsius"
}
''',
)


CORPUS = [C1, C2, C3, C4, C5, C6a, C6b, C6c, C7, C7b, C8, C9]
