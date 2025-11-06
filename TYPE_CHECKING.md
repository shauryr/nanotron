# Type Checking Guide

This document describes the type checking setup for the Nanotron project.

## Overview

Nanotron uses [mypy](https://mypy.readthedocs.io/) for static type checking. Type annotations help:
- Catch bugs early in development
- Improve code documentation and readability
- Enable better IDE support (autocomplete, refactoring)
- Make the codebase more maintainable

## Setup

### Installing Type Checker

Install mypy and required type stubs:

```bash
pip install -e ".[dev]"
```

This installs:
- `mypy>=1.0.0` - The type checker
- `types-PyYAML` - Type stubs for PyYAML
- `types-tqdm` - Type stubs for tqdm

### Configuration

Type checking is configured in `mypy.ini` at the project root. Key settings:

- **Python version**: 3.10
- **Strictness**: Incremental - starting with moderate strictness that can be increased over time
- **Third-party libraries**: Missing stubs are ignored for libraries like torch, transformers, etc.
- **Module-specific**: Stricter checking enabled for `nanotron.config.*` and `nanotron.helpers`

## Running Type Checks

### Locally

Run mypy on the entire codebase:

```bash
python -m mypy src/nanotron
```

Run mypy on specific files:

```bash
python -m mypy src/nanotron/helpers.py
python -m mypy src/nanotron/random.py
```

### Pre-commit Hooks

Type checking runs automatically on git commit via pre-commit hooks. To install:

```bash
pre-commit install
```

To run manually:

```bash
pre-commit run mypy --all-files
```

### CI/CD Pipeline

Type checking runs automatically in GitHub Actions on:
- All pull requests
- Pushes to main branch
- Changes to Python files or mypy configuration

See `.github/workflows/type_checking.yaml` for details.

The workflow:
1. Installs dependencies
2. Runs mypy on `src/nanotron`
3. Generates a report (available as artifact)
4. Shows summary in GitHub Actions UI

## Type Annotation Guidelines

### Basic Examples

```python
# Function with return type
def calculate_sum(a: int, b: int) -> int:
    return a + b

# Function with no return value
def log_message(msg: str) -> None:
    print(msg)

# Optional return type
from typing import Optional

def find_user(id: int) -> Optional[User]:
    # Returns User or None
    pass

# Generic types
from typing import List, Dict, Tuple

def process_data(items: List[str]) -> Dict[str, int]:
    return {item: len(item) for item in items}
```

### PyTorch-specific

```python
import torch
from torch import nn

def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # Implementation
    pass
```

### Context Managers

```python
from typing import Generator
from contextlib import contextmanager

@contextmanager
def my_context() -> Generator[None, None, None]:
    # Setup
    yield
    # Cleanup
```

## Current Status

### Completed
- ✅ mypy configuration (`mypy.ini`)
- ✅ Development dependencies added
- ✅ GitHub Actions workflow
- ✅ Pre-commit hooks
- ✅ Type annotations in:
  - `src/nanotron/helpers.py` - Comprehensive return types
  - `src/nanotron/random.py` - Full type coverage
  - `src/nanotron/optim/base.py` - Critical fixes

### In Progress
- 🔄 Gradual addition of type annotations across the codebase
- 🔄 Fixing type errors in existing modules

### Known Issues

Some modules have type errors that are being addressed incrementally:
- Third-party library stubs (configured to be ignored)
- Complex distributed operations
- Dynamic module loading

## Contributing

When contributing code:

1. **Add type annotations** to all new functions:
   ```python
   def my_function(arg1: str, arg2: int) -> bool:
       pass
   ```

2. **Run mypy** before committing:
   ```bash
   python -m mypy src/nanotron/your_file.py
   ```

3. **Fix type errors** or add `# type: ignore[error-code]` with justification:
   ```python
   result = some_function()  # type: ignore[no-any-return]  # Third-party lib has no stubs
   ```

4. **Use strict typing** for config and helper modules (already configured)

## Resources

- [MyPy Documentation](https://mypy.readthedocs.io/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [MyPy Cheat Sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)
- [Common MyPy Issues](https://mypy.readthedocs.io/en/stable/common_issues.html)

## Troubleshooting

### "Cannot find implementation or library stub"

This means mypy doesn't have type information for a third-party library. Options:

1. Install type stubs: `pip install types-<library>`
2. Add to `mypy.ini`:
   ```ini
   [mypy-library.*]
   ignore_missing_imports = True
   ```

### "Incompatible types" errors

Check that:
- Function return types match what's declared
- Variable assignments match their type annotations
- None is handled properly (use `Optional[T]`)

### Performance Issues

For large codebases, use incremental mode (enabled by default):
```bash
python -m mypy --incremental src/nanotron
```

Clear cache if issues arise:
```bash
python -m mypy --cache-dir=/dev/null src/nanotron
```
