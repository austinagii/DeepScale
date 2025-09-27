# Installation

## Requirements

- Python 3.12+
- PyTorch 2.6.0+

## Install from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/deepscale.git
   cd deepscale
   ```

2. Install dependencies using Pipenv:
   ```bash
   pipenv install
   ```

3. Activate the virtual environment:
   ```bash
   pipenv shell
   ```

## Development Installation

For development, install with dev dependencies:

```bash
pipenv install --dev
```

## Verify Installation

Test your installation:

```python
import deepscale
print(deepscale.__version__)
```