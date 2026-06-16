"""The benchmark task corpus — the horizon × complexity grid + the §6 probe cells.

Declarative data only (no logic), per ``docs/agentic-serving/benchmark-design.md``
§3 (grid), §6 (bleed-injection probe), §10 (structure). Each cell is a
:class:`~benchmarks.agentic_serving.model.Cell`; the grid is enumerable as
:data:`GRID`, the probe cells as :data:`PROBES`, and :func:`load` returns both.

Curation rules applied (§3 — no degenerate corner):

* **Complexity** is per-deliverable difficulty: C1 a trivial function module; C2
  a class with typed methods + docstrings; C3 an argparse CLI with a ``main``
  guard (the documented form-bleed zone); C4 a cross-importing module with error
  handling.
* **Horizon** is deliverable count + cross-file dependency depth: H1 one file; H2
  2–3 with one dependent (later imports earlier); H3 ~5 (a small package); H4
  8–10 multi-module with cross-imports.
* **Degenerate-corner fixes:** "cross-import" complexity only means something with
  siblings, so at H1 the C4 cell is a single *internally* complex module
  (multiple classes + error handling), not a cross-import. At high horizon with
  low complexity the files are many-but-simple (left as-is — that is a valid
  shape, §3).
* **Expected-fail (P2-C, required):** the grid includes H4×C4 (``h4c4``), expected
  to fail under cheap-local — without a cell the cheap stack cannot pass, ceiling
  finding yields only a lower bound.

Every prompt names its deliverables explicitly, asks for *bare file contents*
(no fences / prose — the §4 / ADR-035 form contract), and wires later files to
earlier ones so the content-anchor + axis-2 coherence checks have something to
bite on.
"""

from __future__ import annotations

from benchmarks.agentic_serving.model import Cell

# --- Shared prompt scaffolding ----------------------------------------------

# Appended to every prompt: the bare-output form directive (the ADR-035 client
# form contract the scorer's form check assumes). Kept identical across cells so
# the grid varies only on task shape, not on instructions.
_FORM_DIRECTIVE = (
    "Write each file's exact contents and nothing else — no markdown fences, no "
    "leading or trailing prose, no explanation. Each file must be valid on its "
    "own. Where a later file uses an earlier one, call the real names defined in "
    "that earlier file (do not invent functions, classes, or attributes)."
)


def _prompt(body: str) -> str:
    """A cell prompt: the task body + the shared bare-output form directive."""
    return f"{body}\n\n{_FORM_DIRECTIVE}"


# --- Horizon 1 (one standalone file) ----------------------------------------

H1C1 = Cell(
    name="h1c1",
    horizon=1,
    complexity=1,
    prompt=_prompt(
        "Create one file, mathutils.py, with three trivial functions: add(a, b), "
        "subtract(a, b), and multiply(a, b). Each returns the obvious arithmetic "
        "result."
    ),
    expected_deliverables=("mathutils.py",),
)

H1C2 = Cell(
    name="h1c2",
    horizon=1,
    complexity=2,
    prompt=_prompt(
        "Create one file, account.py, defining a BankAccount class with type "
        "hints and docstrings. Methods: __init__(self, owner: str, balance: "
        "float = 0.0), deposit(self, amount: float) -> float, withdraw(self, "
        "amount: float) -> float, and a balance property. Each method has a "
        "one-line docstring."
    ),
    expected_deliverables=("account.py",),
)

H1C3 = Cell(
    name="h1c3",
    horizon=1,
    complexity=3,
    prompt=_prompt(
        "Create one file, cli.py: a command-line tool. Use argparse to accept a "
        "numeric value as a positional argument and a mutually exclusive flag "
        "group --double / --halve. Apply the chosen operation and print the "
        "result. Define a main() function, parse args inside it, and call it "
        "under an if __name__ == '__main__' guard."
    ),
    expected_deliverables=("cli.py",),
)

H1C4 = Cell(
    name="h1c4",
    horizon=1,
    complexity=4,
    prompt=_prompt(
        "Create one internally complex file, inventory.py, with error handling. "
        "Define a custom exception OutOfStockError, an Item class (name, price, "
        "quantity with type hints), and an Inventory class that holds items and "
        "exposes add(item), remove(name, count) which raises OutOfStockError "
        "when count exceeds quantity, and total_value() -> float. Validate inputs "
        "and raise ValueError on negative quantities or prices."
    ),
    expected_deliverables=("inventory.py",),
)

# --- Horizon 2 (2–3 files, one dependent) -----------------------------------

H2C1 = Cell(
    name="h2c1",
    horizon=2,
    complexity=1,
    prompt=_prompt(
        "Create two files. First, stringutils.py with two trivial functions: "
        "shout(text) returns text uppercased, and whisper(text) returns text "
        "lowercased. Second, demo.py which imports shout and whisper from "
        "stringutils and prints the result of each on the word 'Hello'."
    ),
    expected_deliverables=("stringutils.py", "demo.py"),
)

H2C2 = Cell(
    name="h2c2",
    horizon=2,
    complexity=2,
    prompt=_prompt(
        "Create two files. First, shapes.py defining a Rectangle class with type "
        "hints and docstrings: __init__(self, width: float, height: float), "
        "area(self) -> float, perimeter(self) -> float. Second, "
        "test_shapes.py with unittest test cases that import Rectangle from "
        "shapes and assert area and perimeter for a known rectangle."
    ),
    expected_deliverables=("shapes.py", "test_shapes.py"),
)

H2C3 = Cell(
    name="h2c3",
    horizon=2,
    complexity=3,
    prompt=_prompt(
        "Create three files. First, converters.py with celsius_to_fahrenheit(c) "
        "and fahrenheit_to_celsius(f). Second, cli.py: an argparse command-line "
        "tool that imports converters, takes a numeric value as a positional arg "
        "and a mutually exclusive --to-fahrenheit / --to-celsius flag group, "
        "calls the matching converter, and prints the result to two decimals "
        "under an if __name__ == '__main__' guard with a main() function. Third, "
        "test_converters.py with unittest cases importing converters."
    ),
    expected_deliverables=("converters.py", "cli.py", "test_converters.py"),
)

H2C4 = Cell(
    name="h2c4",
    horizon=2,
    complexity=4,
    prompt=_prompt(
        "Create three files. First, errors.py defining two custom exceptions: "
        "ValidationError and NotFoundError. Second, store.py which imports both "
        "from errors and defines a KeyValueStore class with type hints: put(key, "
        "value) raising ValidationError on a non-string key, get(key) raising "
        "NotFoundError when the key is absent, and delete(key). Third, "
        "test_store.py with unittest cases importing KeyValueStore from store and "
        "asserting the error paths fire."
    ),
    expected_deliverables=("errors.py", "store.py", "test_store.py"),
)

# --- Horizon 3 (~5 files, a small package) ----------------------------------

H3C1 = Cell(
    name="h3c1",
    horizon=3,
    complexity=1,
    prompt=_prompt(
        "Create a small package as five files of trivial helpers, each importing "
        "from the previous one. greet.py: hello(name) returns 'Hello, ' + name. "
        "louder.py: imports hello from greet, loud(name) returns hello(name) "
        "uppercased. softer.py: imports hello from greet, soft(name) returns "
        "hello(name) lowercased. combine.py: imports loud from louder and soft "
        "from softer, both(name) returns a tuple (loud(name), soft(name)). "
        "main.py: imports both from combine and prints both('world')."
    ),
    expected_deliverables=(
        "greet.py",
        "louder.py",
        "softer.py",
        "combine.py",
        "main.py",
    ),
)

H3C2 = Cell(
    name="h3c2",
    horizon=3,
    complexity=2,
    prompt=_prompt(
        "Create a small package as five files, each with classes carrying type "
        "hints and docstrings. animal.py: an Animal base class with name and a "
        "speak(self) -> str method. dog.py: imports Animal from animal, a Dog "
        "subclass overriding speak. cat.py: imports Animal from animal, a Cat "
        "subclass overriding speak. shelter.py: imports Dog from dog and Cat from "
        "cat, a Shelter class that holds animals and a roll_call(self) -> "
        "list[str] method. test_shelter.py: unittest cases importing Shelter, "
        "Dog, Cat and asserting roll_call output."
    ),
    expected_deliverables=(
        "animal.py",
        "dog.py",
        "cat.py",
        "shelter.py",
        "test_shelter.py",
    ),
)

H3C3 = Cell(
    name="h3c3",
    horizon=3,
    complexity=3,
    prompt=_prompt(
        "Create a small package as five files: a module, a CLI, two test files, "
        "and a README. converters.py: celsius_to_fahrenheit(c), "
        "fahrenheit_to_celsius(f), celsius_to_kelvin(c). cli.py: an argparse "
        "command-line tool that imports converters, takes a numeric value and a "
        "mutually exclusive --to-fahrenheit / --to-kelvin flag group, with a "
        "main() function under an if __name__ == '__main__' guard. "
        "test_converters.py: unittest cases importing converters. test_cli.py: "
        "unittest cases that import main from cli. README.md: documents the real "
        "CLI usage."
    ),
    expected_deliverables=(
        "converters.py",
        "cli.py",
        "test_converters.py",
        "test_cli.py",
        "README.md",
    ),
)

H3C4 = Cell(
    name="h3c4",
    horizon=3,
    complexity=4,
    prompt=_prompt(
        "Create a small package as five files with cross-imports and error "
        "handling. errors.py: custom exceptions ParseError and RangeError. "
        "validators.py: imports RangeError from errors, in_range(value, low, "
        "high) raising RangeError when out of bounds. parser.py: imports "
        "ParseError from errors, parse_int(text) raising ParseError on "
        "non-numeric input. pipeline.py: imports parse_int from parser and "
        "in_range from validators, run(text, low, high) that parses then "
        "range-checks, returning the validated int. test_pipeline.py: unittest "
        "cases importing run from pipeline and asserting both error paths."
    ),
    expected_deliverables=(
        "errors.py",
        "validators.py",
        "parser.py",
        "pipeline.py",
        "test_pipeline.py",
    ),
)

# --- Horizon 4 (8–10 files, multi-module with cross-imports) ----------------

H4C1 = Cell(
    name="h4c1",
    horizon=4,
    complexity=1,
    prompt=_prompt(
        "Create eight trivial single-function modules forming a chain, each "
        "importing the previous. step1.py: f1(x) returns x + 1. step2.py: "
        "imports f1 from step1, f2(x) returns f1(x) + 1. step3.py: imports f2 "
        "from step2, f3(x) returns f2(x) + 1. step4.py: imports f3 from step3, "
        "f4(x) returns f3(x) + 1. step5.py: imports f4 from step4, f5(x) returns "
        "f4(x) + 1. step6.py: imports f5 from step5, f6(x) returns f5(x) + 1. "
        "step7.py: imports f6 from step6, f7(x) returns f6(x) + 1. step8.py: "
        "imports f7 from step7 and prints f7(0)."
    ),
    expected_deliverables=(
        "step1.py",
        "step2.py",
        "step3.py",
        "step4.py",
        "step5.py",
        "step6.py",
        "step7.py",
        "step8.py",
    ),
)

H4C2 = Cell(
    name="h4c2",
    horizon=4,
    complexity=2,
    prompt=_prompt(
        "Create eight files of typed, documented classes with cross-imports. "
        "base.py: an Entity base class (id: int, name: str) with a describe(self) "
        "-> str method. user.py: imports Entity from base, a User subclass adding "
        "an email field. product.py: imports Entity from base, a Product subclass "
        "adding a price field. order.py: imports User from user and Product from "
        "product, an Order class holding a user and a list of products with a "
        "total(self) -> float method. cart.py: imports Product from product, a "
        "Cart class with add(product) and items(self) -> list. catalog.py: "
        "imports Product from product, a Catalog class holding products with "
        "find(self, name: str). registry.py: imports User from user, a Registry "
        "class holding users with lookup(self, email: str). test_models.py: "
        "unittest cases importing Order, Cart, and Catalog and asserting their "
        "core methods."
    ),
    expected_deliverables=(
        "base.py",
        "user.py",
        "product.py",
        "order.py",
        "cart.py",
        "catalog.py",
        "registry.py",
        "test_models.py",
    ),
)

H4C3 = Cell(
    name="h4c3",
    horizon=4,
    complexity=3,
    prompt=_prompt(
        "Create nine files: a layered CLI application with cross-imports. "
        "config.py: a load_defaults() returning a settings dict. units.py: "
        "to_meters(value, unit) and from_meters(value, unit) over 'km', 'm', "
        "'cm'. convert.py: imports to_meters and from_meters from units, "
        "convert(value, src, dst). report.py: imports load_defaults from config, "
        "format_result(value, unit) -> str. cli.py: an argparse command-line "
        "tool that imports convert from convert and format_result from report, "
        "with a main() under an if __name__ == '__main__' guard, taking a value "
        "and --from / --to options. test_units.py, test_convert.py, "
        "test_report.py: unittest cases importing the respective modules. "
        "README.md: documents the real CLI usage."
    ),
    expected_deliverables=(
        "config.py",
        "units.py",
        "convert.py",
        "report.py",
        "cli.py",
        "test_units.py",
        "test_convert.py",
        "test_report.py",
        "README.md",
    ),
)

# Expected-fail under cheap-local (P2-C, required) — the top-right corner.
H4C4 = Cell(
    name="h4c4",
    horizon=4,
    complexity=4,
    prompt=_prompt(
        "Create ten files: a multi-module data-pipeline package with deep "
        "cross-imports and error handling throughout. errors.py: custom "
        "exceptions SourceError, TransformError, SinkError. source.py: imports "
        "SourceError, a Source class reading rows, raising SourceError on an "
        "empty input. validate.py: imports TransformError, validate_row(row) "
        "raising TransformError on a malformed row. transform.py: imports "
        "validate_row from validate and TransformError from errors, "
        "transform(rows) applying validation then mapping. sink.py: imports "
        "SinkError, a Sink class writing rows, raising SinkError on a write "
        "failure. pipeline.py: imports Source from source, transform from "
        "transform, and Sink from sink, a Pipeline class with run(self) wiring "
        "source -> transform -> sink. registry.py: imports Pipeline from "
        "pipeline, a Registry mapping names to pipelines. runner.py: imports "
        "Registry from registry, a run_named(name) function. test_pipeline.py: "
        "unittest cases importing Pipeline and asserting the happy path. "
        "test_errors.py: unittest cases asserting each error path fires."
    ),
    expected_deliverables=(
        "errors.py",
        "source.py",
        "validate.py",
        "transform.py",
        "sink.py",
        "pipeline.py",
        "registry.py",
        "runner.py",
        "test_pipeline.py",
        "test_errors.py",
    ),
)

GRID: tuple[Cell, ...] = (
    H1C1,
    H1C2,
    H1C3,
    H1C4,
    H2C1,
    H2C2,
    H2C3,
    H2C4,
    H3C1,
    H3C2,
    H3C3,
    H3C4,
    H4C1,
    H4C2,
    H4C3,
    H4C4,
)


# --- §6 bleed-injection probe cells -----------------------------------------
#
# Hard cells (argparse CLI / cross-importing module) that the runner runs under
# an ADVERSARIAL coder system prompt + a 2B→8B (or 0.6B→8B) tier ladder to force
# the destination-validity gate + coder-tier escalation (§6). The corpus only
# *marks* these (kind="probe") and gives the task; the runner / CLI applies the
# adversarial coder + tier ladder. Prompts here are the same bare-output form
# contract — the bleed comes from the adversarial coder, not the task.

PROBE_CLI = Cell(
    name="probe-cli",
    horizon=1,
    complexity=3,
    prompt=_prompt(
        "Create one file, cli.py: a command-line tool. Use argparse to accept a "
        "numeric temperature value as a positional argument and a mutually "
        "exclusive --to-fahrenheit / --to-celsius flag group. Convert with the "
        "appropriate formula and print the result to two decimals. Define a "
        "main() function, parse args inside it, and call it under an "
        "if __name__ == '__main__' guard. Add a module docstring."
    ),
    expected_deliverables=("cli.py",),
    kind="probe",
)

PROBE_CROSS = Cell(
    name="probe-cross",
    horizon=2,
    complexity=4,
    prompt=_prompt(
        "Create two files. First, converters.py with celsius_to_fahrenheit(c) "
        "and fahrenheit_to_celsius(f), each validating that the input is numeric "
        "and raising TypeError otherwise. Second, cli.py which imports both from "
        "converters, uses argparse with a value and a mutually exclusive "
        "--to-fahrenheit / --to-celsius flag group, calls the real converter, "
        "and prints the result to two decimals under an if __name__ == "
        "'__main__' guard with a main() function."
    ),
    expected_deliverables=("converters.py", "cli.py"),
    kind="probe",
)

PROBES: tuple[Cell, ...] = (PROBE_CLI, PROBE_CROSS)


def load() -> tuple[tuple[Cell, ...], tuple[Cell, ...]]:
    """Return ``(GRID, PROBES)`` — the full benchmark corpus (§10)."""
    return GRID, PROBES
